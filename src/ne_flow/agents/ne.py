import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from ne_flow.flax_utils import ModuleDict, TrainState, nonpytree_field
from ne_flow.models import (
    GCFlowActor,
    GCValue,
)


class NE_Agent(flax.struct.PyTreeNode):
    """
    Hierarchical Implicit Q-Learning with Flow Matching & Action Chunking.

    Features:
    - IQL Value Engine: Single V, Dual Q, Goal-Conditioned.
    - Action Chunking: Low-level policy predicts action sequences.
    - Cascaded Best-of-N: Hierarchical sampling and ranking.
    - Stateful Inference: Internal buffer management for seamless action chunking.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # === Stateful Inference Fields (non-PyTree) ===
    # All mutable state is encapsulated in a dict to bypass FrozenInstanceError.
    # WARNING: This dict is modified in-place during non-JIT inference.
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
        Core Flow Matching Logic: Samples noise, interpolates, and computes squared error.
        Returns element-wise squared difference without reduction.
        Output shape: [Batch, Dim] (Dim can be flat action chunk size)
        """
        batch_size = obs.shape[0]
        x_dim = target_x.shape[-1]

        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # 1. Sample Noise x0 and Time t
        x_0 = jax.random.normal(x_rng, (batch_size, x_dim))
        x_1 = target_x
        t = jax.random.uniform(t_rng, (batch_size, 1))

        # 2. Linear Interpolation (Optimal Transport Path)
        x_t = (1 - t) * x_0 + t * x_1
        target_velocity = x_1 - x_0

        # 3. Predict Velocity
        pred_velocity = network.select(module_name)(obs, goal, x_t, t, params=params)

        # 4. Return Squared Difference (No Mean, No Weighting)
        return jnp.square(pred_velocity - target_velocity)

    # --- 1. IQL Value Engine (Single V, Dual Q) ---
    def value_loss(self, batch, grad_params):
        """
        IQL V-Loss: Expectile Regression.
        Objective: V(s, g) -> Expectile(Q(s, a_chunk, g))
        """
        # Chunk handling: flatten actions if they are chunks
        if self.config["action_chunking"]:
            # [B, H, A] -> [B, H*A]
            actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            actions = batch["actions"]

        # Target Q (Dual Q, no gradient)
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
        if self.config["action_chunking"]:
            actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
            discount = self.config["discount"] ** self.config["horizon_length"]
            # Rewards are summed here if batch has time dim
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
            actions = batch["actions"]
            discount = self.config["discount"]
            rewards = batch["rewards"]
            next_obs = batch["next_observations"]

        # V-Target (Single V network, stop gradient implied by being distinct module logic or explicit)
        # In IQL, we use the V network to bootstrap to avoid OOD queries.
        next_v = self.network.select("value")(next_obs, batch["value_goals"])

        # TD Target
        target_q = rewards + discount * batch["masks"] * next_v
        target_q = jax.lax.stop_gradient(target_q)

        # Current Q (Dual Q)
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

    # --- 2. Hierarchical Policy Learning (Weighted Flow Matching) ---
    def high_actor_loss(self, batch, grad_params, rng):
        """
        High-Level Policy: Predicts Subgoal w.
        Advantage: A = V(w_data, g) - V(s, g)
        """
        # 1. Calculate Advantage
        # V(s, g) - Baseline
        v_curr = self.network.select("value")(
            batch["observations"], batch["high_actor_goals"]
        )
        # V(w_data, g) - Quality of the data sample
        # batch['high_actor_targets'] contains the actual subgoals from dataset
        v_next = self.network.select("value")(
            batch["high_actor_targets"], batch["high_actor_goals"]
        )

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
        Low-Level Policy: Predicts Action Chunk a_t:t+h.
        Advantage: A = Q(s, a_chunk, w) - V(s, w)
        """
        if self.config["action_chunking"]:
            actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            actions = batch["actions"]

        # 1. Calculate Advantage
        # V(s, w) - Baseline
        v = self.network.select("value")(
            batch["observations"], batch["low_actor_goals"]
        )
        # Q(s, a, w) - Quality of the action chunk
        q1, q2 = self.network.select("critic")(
            batch["observations"], batch["low_actor_goals"], actions
        )
        q = jnp.minimum(q1, q2)

        adv = q - v

        # 2. Calculate AWR Weights [B]
        weights = jnp.exp(adv * self.config["low_awr_temp"])
        weights = jnp.clip(weights, max=100.0)
        weights = jax.lax.stop_gradient(weights)

        # 1. Get Squared Diff [B, H*A]
        sq_diff = self.compute_flow_matching_sq_diff(
            self.network,
            "low_actor",
            batch["observations"],
            batch["low_actor_goals"],
            actions,
            rng,
            grad_params,
        )

        # 2. Reshape to recover time horizon: [B, H*A] -> [B, H, A]
        batch_size = sq_diff.shape[0]
        horizon = batch["valid"].shape[1]
        action_dim = sq_diff.shape[-1] // horizon

        sq_diff_reshaped = jnp.reshape(sq_diff, (batch_size, horizon, action_dim))

        # 3. Mean over action dimensions first: [B, H, A] -> [B, H]
        loss_per_step = jnp.mean(sq_diff_reshaped, axis=-1)

        # 4. Apply Valid Mask and AWR Weights
        # step_weights: [B, H]
        step_weights = batch["valid"] * weights[:, None]
        loss = jnp.mean(step_weights * loss_per_step)

        return loss, {
            "low_actor_loss": loss,
            "low_adv_mean": adv.mean(),
            "low_weights": weights.mean(),
        }

    # --- 3. Training Loop Boilerplate ---
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        rng, high_rng, low_rng = jax.random.split(rng, 3)

        info = {}

        # Value Update
        v_loss, v_info = self.value_loss(batch, grad_params)
        for k, v in v_info.items():
            info[f"value/{k}"] = v

        # Critic Update
        c_loss, c_info = self.critic_loss(batch, grad_params)
        for k, v in c_info.items():
            info[f"critic/{k}"] = v

        # High Actor Update
        h_loss, h_info = self.high_actor_loss(batch, grad_params, high_rng)
        for k, v in h_info.items():
            info[f"high_actor/{k}"] = v

        # Low Actor Update
        l_loss, l_info = self.low_actor_loss(batch, grad_params, low_rng)
        for k, v in l_info.items():
            info[f"low_actor/{k}"] = v

        total = v_loss + c_loss + h_loss + l_loss
        return total, info

    def target_update(self, network, module_name):
        """Update target network (only needed for Critic in this setup)."""
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

    # --- 4. Stateful Inference Interface ---
    def sample_actions(
        self,
        observations,
        goals,
        seed,
        temperature=1.0,  # Kept for API compatibility, unused
    ):
        """
        Stateful hierarchical action sampling (non-JIT).

        ⚠️ SIDE EFFECTS WARNING:
        - This method modifies internal agent state in-place (via _state dict).
        - Each agent instance should be used by only one environment/thread.
        - Internal buffers are automatically reset when `goals` change.

        Args:
            observations: [obs_dim] Single observation vector (no batch dim).
            goals: [goal_dim] Single goal vector (no batch dim).
            seed: PRNG seed for deterministic sampling (required).
            temperature: Unused, kept for API compatibility.

        Returns:
            actions: [action_dim] Single action vector.
        """
        # Validate seed
        if seed is None:
            raise ValueError(
                "seed must be provided for reproducibility in FlowHIQLAgent.sample_actions"
            )

        # ===== 1. Handle Input Dimensions (Single Environment Only) =====
        # Input: [obs_dim] -> Internal: [1, obs_dim]
        obs = jnp.expand_dims(observations, 0)
        goal = jnp.expand_dims(goals, 0)

        # Get mutable state dict
        state = self._state

        # ===== 2. Task Change Detection & Auto-Reset =====
        # Since this method is NOT jitted, we can safely use Python control flow.
        if state.get("prev_goal") is not None:
            # Check for goal change: shape mismatch or value difference
            prev_goal = state["prev_goal"]
            goal_changed = (goal.shape != prev_goal.shape) or (
                not jnp.allclose(goal, prev_goal)
            )
            if goal_changed:
                # Task has changed! Reset internal state.
                horizon = self.config["horizon_length"]
                action_dim = self.config["action_dim"]
                obs_dim = self.config["obs_dim"]
                state["action_chunk_buffer"] = jnp.zeros((1, horizon, action_dim))
                state["chunk_step_idx"] = 0
                state["current_subgoal"] = jnp.zeros((1, obs_dim))

        # Update previous goal reference
        state["prev_goal"] = goal

        # ===== 3. Replanning Check =====
        needs_replan = state["chunk_step_idx"] >= self.config["horizon_length"]

        if needs_replan:
            # Generate new subgoal and action chunk
            rng, high_rng, low_rng = jax.random.split(seed, 3)

            # Sample new subgoal
            state["current_subgoal"] = self.sample_high_actions(
                observations=obs, goals=goal, rng=high_rng
            )  # [1, obs_dim]

            # Sample new action chunk
            action_chunk_flat = self.sample_low_actions(
                observations=obs, subgoals=state["current_subgoal"], rng=low_rng
            )  # [1, H*A]

            # Reshape and store: [1, H*A] -> [1, H, A]
            batch_size = 1
            horizon = self.config["horizon_length"]
            action_dim = self.config["action_dim"]
            state["action_chunk_buffer"] = jnp.reshape(
                action_chunk_flat, (batch_size, horizon, action_dim)
            )

            # Reset chunk pointer
            state["chunk_step_idx"] = 0

        # ===== 4. Execute Current Step =====
        # Extract current action: [1, A]
        current_action = state["action_chunk_buffer"][:, state["chunk_step_idx"], :]

        # Advance pointer
        state["chunk_step_idx"] += 1

        # Return [A] without batch dimension
        return current_action[0]

    # --- 5. Cascaded Best-of-N Sampling (JIT-compiled) ---
    @partial(jax.jit, static_argnames=("module_name", "out_dim", "num_samples"))
    def sample_flow_actions(self, module_name, obs, goal, out_dim, num_samples, rng):
        """Helper: Generate N samples from a conditional flow model."""
        # Setup dimensions for broadcasting
        batch_size = obs.shape[0]
        obs_rep = jnp.repeat(obs[:, None, :], num_samples, axis=1)
        goal_rep = jnp.repeat(goal[:, None, :], num_samples, axis=1)

        # Initial Noise x0
        x = jax.random.normal(rng, (batch_size, num_samples, out_dim))

        # Euler Integration
        steps = self.config["flow_steps"]
        dt = 1.0 / steps
        for i in range(steps):
            t_val = i / steps
            t = jnp.full((batch_size, num_samples, 1), t_val)

            vel = self.network.select(module_name)(obs_rep, goal_rep, x, t)
            x = x + vel * dt

        return x

    @jax.jit
    def sample_high_actions(self, observations, goals, rng=None):
        """
        Sample subgoals using Best-of-N.
        Returns: [B, subgoal_dim]
        """
        rng = rng if rng is not None else self.rng
        rng, high_rng = jax.random.split(rng)

        N = self.config["high_num_samples"]
        K = self.config.get("high_top_k", 4)
        subgoal_dim = observations.shape[-1]

        candidate_subgoals = self.sample_flow_actions(
            "high_actor", observations, goals, subgoal_dim, N, high_rng
        )  # [B, N, subgoal_dim]

        # Evaluate with V(w, g)
        flat_obs = candidate_subgoals.reshape(-1, subgoal_dim)
        flat_goals = jnp.repeat(goals[:, None, :], N, axis=1).reshape(
            -1, goals.shape[-1]
        )

        flat_scores = self.network.select("value")(flat_obs, flat_goals)
        scores = flat_scores.reshape(observations.shape[0], N)

        top_k_scores, top_k_indices = jax.lax.top_k(scores, K)  # [B, K]
        # Gather top-k subgoals: [B, K, subgoal_dim]
        top_k_subgoals = candidate_subgoals[
            jnp.arange(candidate_subgoals.shape[0])[:, None], top_k_indices
        ]
        weights = jax.nn.softmax(top_k_scores, axis=-1)  # [B, K]
        weights = weights[:, :, None]  # [B, K, 1]
        weighted_avg_subgoal = jnp.sum(
            top_k_subgoals * weights, axis=1
        )  # [B, subgoal_dim]

        return weighted_avg_subgoal

    @jax.jit
    def sample_low_actions(self, observations, subgoals, rng=None):
        """
        Sample action chunks using Best-of-N.
        Returns: [B, H*A]
        """
        rng = rng if rng is not None else self.rng
        rng, low_rng = jax.random.split(rng)

        N = self.config["low_num_samples"]
        K = self.config.get("low_top_k", 4)
        action_dim = self.config["action_dim"] * self.config["horizon_length"]

        candidate_actions = self.sample_flow_actions(
            "low_actor", observations, subgoals, action_dim, N, low_rng
        )  # [B, N, H*A]
        candidate_actions = jnp.clip(candidate_actions, -1.0, 1.0)

        # Evaluate with Q(s, a, w)
        flat_obs = jnp.repeat(observations[:, None, :], N, axis=1).reshape(
            -1, observations.shape[-1]
        )
        flat_subgoals = jnp.repeat(subgoals[:, None, :], N, axis=1).reshape(
            -1, subgoals.shape[-1]
        )
        flat_actions = candidate_actions.reshape(-1, action_dim)

        q1, q2 = self.network.select("critic")(flat_obs, flat_subgoals, flat_actions)
        flat_q = (q1 + q2) / 2
        q_scores = flat_q.reshape(observations.shape[0], N)

        top_k_scores, top_k_indices = jax.lax.top_k(q_scores, K)  # [B, K]
        # Gather top-k actions: [B, K, H*A]
        top_k_actions = candidate_actions[
            jnp.arange(candidate_actions.shape[0])[:, None], top_k_indices
        ]
        weights = jax.nn.softmax(top_k_scores, axis=-1)  # [B, K]
        weights = weights[:, :, None]  # [B, K, 1]
        weighted_avg_action = jnp.sum(top_k_actions * weights, axis=1)  # [B, H*A]

        return weighted_avg_action

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        obs_dim = ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]
        # Action Chunk dim
        full_action_dim = action_dim * (
            config["horizon_length"] if config["action_chunking"] else 1
        )

        # Placeholders for initialization
        ex_goals = ex_observations
        ex_subgoals = ex_observations
        ex_full_actions = jnp.zeros((1, full_action_dim))
        ex_time = jnp.zeros((1, 1))

        # Networks
        value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=False,
            gc_encoder=None,
        )
        critic_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
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
        low_actor_def = GCFlowActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["layer_norm"],
            gc_encoder=None,
        )

        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            critic=(critic_def, (ex_observations, ex_goals, ex_full_actions)),
            target_critic=(
                copy.deepcopy(critic_def),
                (ex_observations, ex_goals, ex_full_actions),
            ),
            high_actor=(
                high_actor_def,
                (ex_observations, ex_goals, ex_subgoals, ex_time),
            ),
            low_actor=(
                low_actor_def,
                (ex_observations, ex_subgoals, ex_full_actions, ex_time),
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

        # Store dimensions for inference buffer initialization
        config["obs_dim"] = obs_dim
        config["action_dim"] = action_dim  # Raw action dim

        # Initialize stateful inference buffers in a mutable dict
        horizon = config["horizon_length"]
        batch_size = 1  # Single environment assumption

        inference_state = {
            "action_chunk_buffer": jnp.zeros((batch_size, horizon, action_dim)),
            "chunk_step_idx": 0,
            "current_subgoal": jnp.zeros((batch_size, obs_dim)),
            "prev_goal": None,
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
            agent_name="neflow",
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            # IQL Params
            expectile=0.9,
            # Hierarchical Params
            subgoal_steps=10,
            low_top_k=4,
            high_top_k=4,
            discrete=False,  # unused
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
            horizon_length=5,
            # Inference Params
            high_num_samples=32,
            low_num_samples=32,
            # Misc
            encoder=None,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
