import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from ne_flow.flax_utils import ModuleDict, TrainState, nonpytree_field
from ne_flow.models import (
    GCChunkCritic,  # ### MODIFIED: Import new Critic
    GCFlowActor,
    GCUnet,  # ### MODIFIED: Import new Actor
    GCValue,
)


class NE_without_temporal_ensemble(flax.struct.PyTreeNode):
    """
    Hierarchical Implicit Q-Learning with Flow Matching & Action Chunking.

    ### MODIFIED Architecture:
    - Low-Level Policy: GCUnet (1D Temporal U-Net)
    - Critic: GCChunkCritic (1D Temporal CNN with FiLM)
    - High-Level Policy: MLP Flow Matching (Unchanged)
    - Value: MLP IQL Value (Unchanged)
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # === Stateful Inference Fields (non-PyTree) ===
    _state: dict = nonpytree_field(default_factory=dict)

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """IQL Expectile Loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    @staticmethod
    def compute_flow_matching_sq_diff(
        network, module_name, obs, goal, target_x, rng, params
    ):
        """
        Core Flow Matching Logic.
        ### MODIFIED: Supports both 2D [B, D] (High Actor) and 3D [B, H, A] (Low Actor) inputs.
        """
        batch_size = obs.shape[0]
        # target_x shape can be [B, Dim] or [B, Horizon, Dim]

        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # 1. Sample Noise x0 matching target shape
        x_0 = jax.random.normal(x_rng, target_x.shape)
        x_1 = target_x

        # Sample t [B, 1]
        t = jax.random.uniform(t_rng, (batch_size, 1))

        # Handle broadcasting for 3D tensors (Low Actor)
        # If x is [B, H, A], t needs to be [B, 1, 1] for broadcasting
        if target_x.ndim == 3:
            t_broadcast = t[:, :, None]
        else:
            t_broadcast = t

        # 2. Linear Interpolation (Optimal Transport Path)
        x_t = (1 - t_broadcast) * x_0 + t_broadcast * x_1
        target_velocity = x_1 - x_0

        # 3. Predict Velocity
        # GCUnet expects: (obs, goal, actions, times)
        pred_velocity = network.select(module_name)(obs, goal, x_t, t, params=params)

        # 4. Return Squared Difference (No Mean, No Weighting)
        return jnp.square(pred_velocity - target_velocity)

    # --- 1. IQL Value Engine ---
    def value_loss(self, batch, grad_params):
        """
        IQL V-Loss: Expectile Regression.
        Objective: V(s, g) -> Expectile(Q(s, a_chunk, g))
        """
        # ### MODIFIED: Removed flattening.
        # Critic now expects [B, H, A], and batch["actions"] is naturally [B, H, A]
        actions = batch["actions"]

        # Target Q (Dual Q, no gradient)
        # GCChunkCritic input: (obs, goal, actions_3d)
        q1, q2 = self.network.select("target_critic")(
            batch["observations"], batch["value_goals"], actions
        )
        q_target = jnp.minimum(q1, q2)

        # Current V (Single V, with gradient)
        v = self.network.select("value")(
            batch["observations"], batch["value_goals"], params=grad_params
        )

        adv = q_target - v
        value_loss = self.expectile_loss(adv, adv, self.config["expectile"]).mean()

        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
        }

    def critic_loss(self, batch, grad_params):
        """
        IQL Q-Loss: MSE.
        Objective: Q(s, a_chunk, g) -> r_sum + gamma^H * V(s', g)
        """
        # ### MODIFIED: Removed flattening.
        if self.config["action_chunking"]:
            actions = batch["actions"]  # [B, H, A]
            discount = self.config["discount"] ** self.config["horizon_length"]
            # Rewards are summed here
            rewards = jnp.sum(
                batch["rewards"]
                * (
                    self.config["discount"] ** jnp.arange(self.config["horizon_length"])
                ),
                axis=1,
            )
            # Next observation is H steps ahead
            next_obs = batch["next_observations"]
        else:
            # Fallback (though config usually enforces chunking with this agent)
            actions = batch["actions"]
            discount = self.config["discount"]
            rewards = batch["rewards"]
            next_obs = batch["next_observations"]

        # V-Target
        next_v = self.network.select("value")(next_obs, batch["value_goals"])

        # TD Target
        target_q = rewards + discount * batch["masks"] * next_v
        target_q = jax.lax.stop_gradient(target_q)

        # Current Q (Dual Q) via GCChunkCritic
        q1, q2 = self.network.select("critic")(
            batch["observations"], batch["value_goals"], actions, params=grad_params
        )

        loss1 = jnp.mean((q1 - target_q) ** 2)
        loss2 = jnp.mean((q2 - target_q) ** 2)
        critic_loss = loss1 + loss2

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": 0.5 * (q1 + q2).mean(),
        }

    # --- 2. Hierarchical Policy Learning ---
    def high_actor_loss(self, batch, grad_params, rng):
        """
        High-Level Policy: Predicts Subgoal w. (MLP Flow Matching - Unchanged)
        """
        # 1. Calculate Advantage
        obs_concat = jnp.concatenate(
            [batch["observations"], batch["high_actor_targets"]], axis=0
        )
        goals_concat = jnp.concatenate(
            [batch["high_actor_goals"], batch["high_actor_goals"]], axis=0
        )

        v_all = self.network.select("value")(obs_concat, goals_concat)
        v_curr, v_next = jnp.split(v_all, 2, axis=0)

        adv = v_next - v_curr

        # 2. Calculate AWR Weights
        weights = jnp.exp(adv * self.config["high_awr_temp"])
        weights = jnp.clip(weights, max=100.0)
        weights = jax.lax.stop_gradient(weights)

        # Get Squared Diff [B, Obs_Dim]
        sq_diff = self.compute_flow_matching_sq_diff(
            self.network,
            "high_actor",
            batch["observations"],
            batch["high_actor_goals"],
            batch["high_actor_targets"],
            rng,
            grad_params,
        )

        loss_per_sample = jnp.mean(sq_diff, axis=-1)  # Reduce -> [B]
        loss = jnp.mean(weights * loss_per_sample)

        return loss, {
            "high_actor_loss": loss,
            "high_adv_mean": adv.mean(),
            "high_weights": weights.mean(),
        }

    def low_actor_loss(self, batch, grad_params, rng):
        """
        Low-Level Policy: Predicts Action Chunk a_t:t+h using GCUnet.
        """
        # ### MODIFIED: Actions are [B, H, A], no reshaping
        actions = batch["actions"]

        # 1. Calculate Advantage
        v = self.network.select("value")(
            batch["observations"], batch["low_actor_goals"]
        )
        # GCChunkCritic: Q(s, a_chunk, w)
        q1, q2 = self.network.select("critic")(
            batch["observations"], batch["low_actor_goals"], actions
        )
        q = jnp.minimum(q1, q2)

        adv = q - v

        # 2. Calculate AWR Weights [B]
        weights = jnp.exp(adv * self.config["low_awr_temp"])
        weights = jnp.clip(weights, max=100.0)
        weights = jax.lax.stop_gradient(weights)

        # 3. Get Squared Diff [B, H, A]
        # GCUnet returns output same shape as input
        sq_diff = self.compute_flow_matching_sq_diff(
            self.network,
            "low_actor",
            batch["observations"],
            batch["low_actor_goals"],
            actions,
            rng,
            grad_params,
        )

        # 4. Mean over action dimensions: [B, H, A] -> [B, H]
        loss_per_step = jnp.mean(sq_diff, axis=-1)

        # 5. Apply Valid Mask and AWR Weights
        # step_weights: [B, H]
        step_weights = batch["valid"] * weights[:, None]
        loss = jnp.mean(step_weights * loss_per_step)

        return loss, {
            "low_actor_loss": loss,
            "low_adv_mean": adv.mean(),
            "low_weights": weights.mean(),
        }

    # --- 3. Training Loop Boilerplate (Unchanged) ---
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        rng, high_rng, low_rng = jax.random.split(rng, 3)

        info = {}

        v_loss, v_info = self.value_loss(batch, grad_params)
        for k, v in v_info.items():
            info[f"value/{k}"] = v

        c_loss, c_info = self.critic_loss(batch, grad_params)
        for k, v in c_info.items():
            info[f"critic/{k}"] = v

        h_loss, h_info = self.high_actor_loss(batch, grad_params, high_rng)
        for k, v in h_info.items():
            info[f"high_actor/{k}"] = v

        l_loss, l_info = self.low_actor_loss(batch, grad_params, low_rng)
        for k, v in l_info.items():
            info[f"low_actor/{k}"] = v

        total = v_loss + c_loss + h_loss + l_loss
        return total, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "critic")
        return self.replace(network=new_network, rng=new_rng), info

    # --- 4. Stateful Inference Interface (Unchanged logic, just internal buffer shape) ---
    def sample_actions(self, observations, goals, seed, temperature=1.0):
        if seed is None:
            raise ValueError("Seed required.")
        obs = jnp.expand_dims(observations, 0)
        goal = jnp.expand_dims(goals, 0)

        if not self._state:
            new_state = self.reset_inference_state(
                self.config["obs_dim"],
                self.config["action_dim"],
                self.config["horizon_length"],
            )
            self._state.update(new_state)

        state = self._state

        if state["prev_goal"] is not None:
            goal_changed = (goal.shape != state["prev_goal"].shape) or (
                not jnp.allclose(goal, state["prev_goal"])
            )
            if goal_changed:
                new_state = self.reset_inference_state(
                    self.config["obs_dim"],
                    self.config["action_dim"],
                    self.config["horizon_length"],
                )
                self._state.clear()
                self._state.update(new_state)
                state = self._state

        state["prev_goal"] = goal

        traj_horizon = self.config["horizon_length"]
        subgoal_horizon = self.config["subgoal_horizon"]
        update_subgoal = (state["high_step_counter"] % subgoal_horizon) == 0

        rng, high_rng, low_rng = jax.random.split(seed, 3)

        if update_subgoal:
            state["current_subgoal"] = self.sample_high_actions(
                observations=obs, goals=goal, rng=high_rng
            )
            state["high_step_counter"] = 0

        state["high_step_counter"] += 1

        low_interval = self.config["low_actor_update_interval"]
        update_low = (state["low_step_counter"] % low_interval) == 0

        if update_low:
            # ### MODIFIED: sample_low_actions now returns [1, H, A]
            # No need to reshape from flat
            action_chunk = self.sample_low_actions(
                observations=obs, subgoals=state["current_subgoal"], rng=low_rng
            )
            # Remove batch dim: [1, H, A] -> [H, A]
            state["low_action_chunk"] = action_chunk[0]

        chunk_idx = state["low_step_counter"] % low_interval
        chunk_idx = jnp.minimum(chunk_idx, traj_horizon - 1)
        current_action = state["low_action_chunk"][chunk_idx]

        state["low_step_counter"] += 1
        return current_action

    def reset_inference_state(self, obs_dim, action_dim, horizon):
        return {
            "current_subgoal": jnp.zeros((1, obs_dim)),
            "prev_goal": None,
            "high_step_counter": 0,
            "low_action_chunk": jnp.zeros((horizon, action_dim)),
            "low_step_counter": 0,
        }

    # --- 5. Cascaded Best-of-N Sampling ---
    def sample_flow_actions(
        self, module_name, obs, goal, out_shape_tuple, num_samples, rng
    ):
        """
        Helper: Generate N samples.
        ### MODIFIED: Handles batch merging for UNet/CNN compatibility.

        Args:
            out_shape_tuple: Tuple defining shape of x (excluding Batch).
                             e.g., (Dim,) for High Actor, (Horizon, Dim) for Low Actor.
        """
        batch_size = obs.shape[0]

        # 1. Expand inputs: [B, D] -> [B, N, D] -> [B*N, D]
        # We merge dimensions because UNet expects [Batch, Time, Channel]
        obs_rep = jnp.repeat(obs, num_samples, axis=0)  # [B*N, O]
        goal_rep = jnp.repeat(goal, num_samples, axis=0)  # [B*N, G]

        # 2. Initial Noise
        # x_shape: [B*N, *out_shape_tuple]
        total_samples = batch_size * num_samples
        x = jax.random.normal(rng, (total_samples, *out_shape_tuple))

        # 3. Euler Integration
        steps = self.config["flow_steps"]
        dt = 1.0 / steps
        for i in range(steps):
            t_val = i / steps
            # Time: [B*N, 1]
            t = jnp.full((total_samples, 1), t_val)

            vel = self.network.select(module_name)(obs_rep, goal_rep, x, t)
            x = x + vel * dt

        # 4. Reshape back: [B*N, ...] -> [B, N, ...]
        final_shape = (batch_size, num_samples, *out_shape_tuple)
        return x.reshape(final_shape)

    @jax.jit
    def sample_high_actions(self, observations, goals, rng=None):
        """High Actor (MLP) - Largely unchanged, but using updated helper."""
        rng = rng if rng is not None else self.rng
        rng, high_rng = jax.random.split(rng)

        N = self.config["high_num_samples"]
        K = self.config.get("high_top_k", 4)
        subgoal_dim = observations.shape[-1]

        # Output shape for high actor is just (Dim,)
        candidate_subgoals = self.sample_flow_actions(
            "high_actor", observations, goals, (subgoal_dim,), N, high_rng
        )  # [B, N, subgoal_dim]

        # Re-flatten for Value function scoring (MLP Value)
        flat_goal_input = jnp.repeat(goals, N, axis=0)  # [B*N, G]
        flat_candidates = candidate_subgoals.reshape(-1, subgoal_dim)

        # V(subgoal, goal) ? Original code used V(subgoal, goal) for scoring w?
        # Logic: Score = V(w, g). Yes.

        flat_scores = self.network.select("value")(flat_candidates, flat_goal_input)
        scores = flat_scores.reshape(observations.shape[0], N)

        top_k_scores, top_k_indices = jax.lax.top_k(scores, K)

        top_k_subgoals = candidate_subgoals[
            jnp.arange(candidate_subgoals.shape[0])[:, None], top_k_indices
        ]
        weights = jax.nn.softmax(top_k_scores, axis=-1)[:, :, None]
        weighted_avg_subgoal = jnp.sum(top_k_subgoals * weights, axis=1)

        return weighted_avg_subgoal

    @jax.jit
    def sample_low_actions(self, observations, subgoals, rng=None):
        """
        Low Actor (GCUnet).
        ### MODIFIED: Handles 3D action structure [B, H, A].
        """
        rng = rng if rng is not None else self.rng
        rng, low_rng = jax.random.split(rng)

        N = self.config["low_num_samples"]
        K = self.config.get("low_top_k", 4)

        action_dim = self.config["action_dim"]
        horizon = self.config["horizon_length"]

        # Output shape for Low Actor is (Horizon, Action_Dim)
        candidate_actions = self.sample_flow_actions(
            "low_actor", observations, subgoals, (horizon, action_dim), N, low_rng
        )  # [B, N, H, A]

        candidate_actions = jnp.clip(candidate_actions, -1.0, 1.0)

        # Evaluate with GCChunkCritic: Q(s, a_chunk, w)
        # Flatten B and N for scoring
        flat_obs = jnp.repeat(observations, N, axis=0)  # [B*N, O]
        flat_subgoals = jnp.repeat(subgoals, N, axis=0)  # [B*N, G]
        flat_actions = candidate_actions.reshape(-1, horizon, action_dim)  # [B*N, H, A]

        # Critic expects [Batch, Horizon, Dim]
        q1, q2 = self.network.select("critic")(flat_obs, flat_subgoals, flat_actions)
        q_min = jnp.minimum(q1, q2)  # [B*N]

        q_scores = q_min.reshape(observations.shape[0], N)

        top_k_scores, top_k_indices = jax.lax.top_k(q_scores, K)

        # Gather top-k actions: [B, K, H, A]
        top_k_actions = candidate_actions[
            jnp.arange(candidate_actions.shape[0])[:, None], top_k_indices
        ]

        weights = jax.nn.softmax(top_k_scores, axis=-1)  # [B, K]
        weights = weights[:, :, None, None]  # [B, K, 1, 1] for broadcasting

        weighted_avg_action = jnp.sum(top_k_actions * weights, axis=1)  # [B, H, A]

        return weighted_avg_action

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        obs_dim = ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]

        # ### MODIFIED: Setup dimensions for 3D tensors
        horizon = config["horizon_length"]
        action_chunk_shape = (1, horizon, action_dim)

        # Placeholders for initialization
        ex_goals = ex_observations
        ex_subgoals = ex_observations
        ex_time = jnp.zeros((1, 1))

        # Dummy action inputs
        ex_actions_chunk = jnp.zeros(action_chunk_shape)

        # Networks
        value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=False,
            gc_encoder=None,
        )

        # ### MODIFIED: Critic -> GCChunkCritic
        critic_def = GCChunkCritic(
            hidden_dims=config["value_hidden_dims"],
            conv_dims=config["critic_conv_dims"],
            layer_norm=config["layer_norm"],
            ensemble=True,
            gc_encoder=None,
        )

        high_actor_def = GCFlowActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=obs_dim,
            layer_norm=config["layer_norm"],
            gc_encoder=None,
        )

        # ### MODIFIED: Low Actor -> GCUnet
        low_actor_def = GCUnet(
            size_channel=action_dim,
            size_emb_transport=config["unet_time_emb_dim"],
            size_channel_hidden=config["unet_hidden_dims"],
            period_min=1.0,
            period_max=1000.0,
            size_kernel=config["unet_kernel_size"],
            size_group_norm=8,  # Good default
        )

        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            # Critic input: (obs, goal, action_chunk_3d)
            critic=(critic_def, (ex_observations, ex_goals, ex_actions_chunk)),
            target_critic=(
                copy.deepcopy(critic_def),
                (ex_observations, ex_goals, ex_actions_chunk),
            ),
            high_actor=(
                high_actor_def,
                (ex_observations, ex_goals, ex_subgoals, ex_time),
            ),
            # Low Actor input: (obs, goal, action_chunk_3d, time)
            low_actor=(
                low_actor_def,
                (ex_observations, ex_subgoals, ex_actions_chunk, ex_time),
            ),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_critic"] = params["modules_critic"]

        config["obs_dim"] = obs_dim
        config["action_dim"] = action_dim

        inference_state = {
            "current_subgoal": jnp.zeros((1, obs_dim)),
            "prev_goal": None,
            "high_step_counter": 0,
            "low_action_chunk": jnp.zeros((horizon, action_dim)),
            "low_step_counter": 0,
        }

        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(**config),
            _state=inference_state,
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="neflow_notemporal",
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),  # For High Actor
            value_hidden_dims=(512, 512, 512),  # For Value and Critic MLP Heads
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            # IQL Params
            expectile=0.9,
            # Hierarchical Params
            subgoal_steps=25,
            low_top_k=1,
            high_top_k=2,
            discrete=False,
            # Dataset Params
            dataset_class="HGCChunkDataset",
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,
            # Flow / AWR Params
            flow_steps=10,
            high_awr_temp=3.0,
            low_awr_temp=3.0,
            # Chunking Params
            action_chunking=True,
            horizon_length=16,  # ### MODIFIED: Need longer horizon for UNet (must be div by 2^downsamples)
            temporal_decay=0.1,
            subgoal_horizon=16,  # Sync with horizon usually
            low_actor_update_interval=16,  # Sync with horizon usually
            # Inference Params
            high_num_samples=32,
            low_num_samples=32,
            # Misc
            encoder=None,
            frame_stack=ml_collections.config_dict.placeholder(int),
            # ### MODIFIED: New Params for UNet and ChunkCritic
            critic_conv_dims=(128, 256),
            unet_hidden_dims=(128, 256),
            unet_kernel_size=3,
            unet_time_emb_dim=128,
        )
    )
    return config
