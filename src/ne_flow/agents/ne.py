import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from ne_flow.flax_utils import ModuleDict, TrainState, nonpytree_field
from ne_flow.models import (
    GCUnet,
    GCValue,
)


class NE_Agent(flax.struct.PyTreeNode):
    """
    Features:
    - No DDPG: Purely offline RL via Advantage-Weighted Flow Matching.
    - IQL Value Engine: Single V, Dual Q, Goal-Conditioned.
    - Action Chunking: Low-level policy predicts action sequences.
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
        Core Flow Matching Logic for GCUnet (动作序列版).

        Args:
            target_x: [Batch, Horizon, Action_dim]

        Returns:
            [Batch, Horizon, Action_dim] (element-wise squared diff)
        """
        batch_size = obs.shape[0]
        horizon = target_x.shape[1]
        action_dim = target_x.shape[2]

        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # 1. Sample noise and time
        x_0 = jax.random.normal(x_rng, (batch_size, horizon, action_dim))
        x_1 = target_x
        t = jax.random.uniform(t_rng, (batch_size, 1))  # [B, 1]

        # 2. Linear Interpolation
        # 广播: [B, 1] -> [B, 1, 1] 可以与 [B, H, A] 广播
        t_broadcast = t[:, :, None]  # [B, 1, 1]
        x_t = (1 - t_broadcast) * x_0 + t_broadcast * x_1
        target_velocity = x_1 - x_0  # [B, H, A]

        # 3. Predict velocity using GCUnet
        # GCUnet: (observations[B, obs_dim], goals[B, goal_dim], actions[B, H, A], times[B, 1])
        pred_velocity = network.select(module_name)(
            obs,
            goal,
            x_t,  # [B, H, A]
            t,  # [B, 1]
            params=params,
        )  # [B, H, A]

        # 4. Return element-wise squared diff
        return jnp.square(pred_velocity - target_velocity)  # [B, H, A]

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

    # --- 2. Weighted Flow Matching ---
    def actor_loss(self, batch, grad_params, rng):
        """
        Low-Level Policy: Predicts Action Chunk a_t:t+h.
        Advantage: A = Q(s, a_chunk, g) - V(s, g)
        """
        # 直接使用序列结构 [B, H, A]
        actions = batch["actions"]

        # 1. Calculate Advantage (Critic needs flat actions)
        # V(s, g)
        v = self.network.select("value")(
            batch["observations"], batch["actor_goals"]
        )  # [B]

        # 为 Critic 展平动作: [B, H, A] -> [B, H*A]
        actions_flat = jnp.reshape(actions, (actions.shape[0], -1))

        # Q(s, a, g)
        q1, q2 = self.network.select("critic")(
            batch["observations"], batch["actor_goals"], actions_flat
        )  # [B]
        q = jnp.minimum(q1, q2)

        adv = q - v  # [B]

        # 2. AWR weights [B]
        weights = jnp.exp(adv * self.config["awr_temp"])
        weights = jnp.clip(weights, max=100.0)
        weights = jax.lax.stop_gradient(weights)  # [B]

        # 3. Get Squared Diff [B, H, A] from GCUnet
        sq_diff = self.compute_flow_matching_sq_diff(
            self.network,
            "actor",
            batch["observations"],
            batch["actor_goals"],
            actions,  # 直接传入 [B, H, A]
            rng,
            grad_params,
        )  # [B, H, A]

        # 4. Apply validity mask per time step
        # [B, H, A] -> [B, H]
        loss_per_step = jnp.mean(sq_diff, axis=-1)  # 在动作维度上平均

        # step_weights: [B, H]
        step_weights = batch["valid"] * weights[:, None]

        # 5. Final loss
        masked_loss = step_weights * loss_per_step  # [B, H]
        loss = jnp.mean(masked_loss)  # 全局平均

        return loss, {
            "actor_loss": loss,
            "adv_mean": adv.mean(),
            "weights": weights.mean(),
            "loss_per_step_mean": loss_per_step.mean(),
        }

    # --- 3. Training Loop Boilerplate ---
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        rng, actor_rng = jax.random.split(rng)

        info = {}

        # Value Update
        v_loss, v_info = self.value_loss(batch, grad_params)
        for k, v in v_info.items():
            info[f"value/{k}"] = v

        # Critic Update
        c_loss, c_info = self.critic_loss(batch, grad_params)
        for k, v in c_info.items():
            info[f"critic/{k}"] = v

        # Low Actor Update
        l_loss, l_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in l_info.items():
            info[f"actor/{k}"] = v

        total = v_loss + c_loss + l_loss
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
        Stateful action chunk sampling (non-JIT).

        ⚠️ SIDE EFFECTS WARNING:
        - This method modifies internal agent state in-place (via _state dict).
        - Each agent instance should be used by only one environment/thread.
        - Internal buffers are automatically reset when `goals` change.

        Args:
            observations: [obs_dim] Single observation vector (no batch dim).
            goals: [goal_dim] Single goal vector (no batch dim).
            seed: rng for deterministic sampling (required).
            temperature: Unused, kept for API compatibility.

        Returns:
            actions: [action_dim] Single action vector.
        """

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
                state["action_chunk_buffer"] = jnp.zeros((1, horizon, action_dim))
                state["chunk_step_idx"] = 0

        # Update previous goal reference
        state["prev_goal"] = goal

        # ===== 3. Replanning Check =====
        needs_replan = state["chunk_step_idx"] >= self.config["horizon_length"]

        if needs_replan:
            # Sample new action chunk
            action_chunk_flat = self.sample_low_actions(
                observations=obs, goals=goal, rng=seed
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

    # --- 5. Best-of-N Sampling ---
    def sample_flow_actions(
        self, module_name, obs, goal, horizon, action_dim, num_samples, rng
    ):
        """Generate N samples from a conditional flow model for action chunks."""
        batch_size = obs.shape[0]

        # Repeat for samples: [B, D] -> [B, N, D]
        obs_rep = jnp.repeat(obs[:, None, :], num_samples, axis=1)  # [B, N, obs_dim]
        goal_rep = jnp.repeat(goal[:, None, :], num_samples, axis=1)  # [B, N, goal_dim]

        # Initialize action sequence: [B, N, H, A]
        x = jax.random.normal(rng, (batch_size, num_samples, horizon, action_dim))

        steps = self.config["flow_steps"]
        dt = 1.0 / steps

        for i in range(steps):
            t_val = i / steps
            # Create time tensor: [B, N, 1]
            t = jnp.full((batch_size, num_samples, 1), t_val)

            # reshape: [B, N, ...] -> [B*N, ...]
            obs_flat = obs_rep.reshape(-1, obs_rep.shape[-1])
            goal_flat = goal_rep.reshape(-1, goal_rep.shape[-1])
            x_flat = x.reshape(-1, *x.shape[-2:])  # [B*N, H, A]
            t_flat = t.reshape(-1, 1)  # [B*N, 1]

            vel_flat = self.network.select(module_name)(
                obs_flat, goal_flat, x_flat, t_flat
            )

            # Reshape back: [B*N, H, A] -> [B, N, H, A]
            vel = vel_flat.reshape(batch_size, num_samples, horizon, action_dim)

            # Update: [B, N, H, A]
            x = x + vel * dt

        # Reshape to flat for Q evaluation: [B, N, H, A] -> [B, N, H*A]
        return jnp.reshape(x, (batch_size, num_samples, -1))

    @jax.jit
    def sample_low_actions(self, observations, goals, rng):
        """
        Low-Level Action Chunk Sampling given Goals.
        Returns: [B, H*A] (flattened for consistency with critic)
        """
        N = self.config["low_num_samples"]
        horizon = self.config["horizon_length"]
        action_dim = self.config["action_dim"]

        # Get samples: [B, N, H*A]
        candidate_actions_flat = self.sample_flow_actions(
            "actor", observations, goals, horizon, action_dim, N, rng
        )
        candidate_actions_flat = jnp.clip(candidate_actions_flat, -1.0, 1.0)

        # Q evaluation (needs flattening)
        batch_size = observations.shape[0]
        obs_dim = observations.shape[-1]
        goal_dim = goals.shape[-1]

        flat_obs = jnp.repeat(observations[:, None, :], N, axis=1).reshape(-1, obs_dim)
        flat_subgoals = jnp.repeat(goals[:, None, :], N, axis=1).reshape(-1, goal_dim)
        flat_actions = candidate_actions_flat.reshape(-1, horizon * action_dim)

        q1, q2 = self.network.select("critic")(flat_obs, flat_subgoals, flat_actions)
        flat_q = (q1 + q2) / 2
        q_scores = flat_q.reshape(batch_size, N)  # [B, N]

        # Select Best Action Chunk
        best_idx = jnp.argmax(q_scores, axis=1)  # [B]
        best_actions = candidate_actions_flat[
            jnp.arange(len(best_idx)), best_idx
        ]  # [B, H*A]

        return best_actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        action_dim = ex_actions.shape[-1]
        obs_dim = ex_observations.shape[-1]
        horizon = config["horizon_length"] if config["action_chunking"] else 1

        # Placeholders for initialization
        ex_goals = ex_observations
        # 1. Critic 需要展平的动作: [1, H*A]
        full_action_dim_flat = action_dim * horizon
        ex_actions_flat = jnp.zeros((1, full_action_dim_flat))
        # 2. Actor 需要序列结构: [1, H, A]
        ex_actions_seq = jnp.zeros((1, horizon, action_dim))
        # 3. Time 保持不变: [1, 1]
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
        actor_def = GCUnet(
            size_channel=action_dim,  # 输入通道数 = 动作维度
            size_emb_transport=32,  # TODO: 配置化
            size_channel_hidden=[64, 128, 256],
            period_min=0.002,
            period_max=10.0,
            size_kernel=3,
            size_group_norm=8,
        )

        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            # Critic 使用展平动作
            critic=(critic_def, (ex_observations, ex_goals, ex_actions_flat)),
            target_critic=(
                copy.deepcopy(critic_def),
                (ex_observations, ex_goals, ex_actions_flat),
            ),
            # Actor 使用序列动作
            actor=(
                actor_def,
                (ex_observations, ex_goals, ex_actions_seq, ex_time),
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
        config["goal_dim"] = obs_dim  # Assuming goal dim = obs dim
        config["action_dim"] = action_dim  # Raw action dim
        config["horizon"] = horizon

        # Initialize stateful inference buffers in a mutable dict
        batch_size = 1  # Single environment assumption
        inference_state = {
            "action_chunk_buffer": jnp.zeros((batch_size, horizon, action_dim)),
            "chunk_step_idx": 0,
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
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            # IQL Params
            expectile=0.9,  # IQL Expectile (0.7-0.9 is standard)
            subgoal_steps=20,  # unused
            discrete=False,  # unused
            # Dataset Params
            dataset_class="GCChunkDataset",
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            # Flow / AWR Params
            flow_steps=10,  # ODE Integration steps
            awr_temp=3.0,  # AWR 温度 (越大区分度越高，但也越不稳定)
            # Chunking Params
            action_chunking=True,
            horizon_length=8,  # Chunk size
            # Inference Params (Best-of-N)
            low_num_samples=32,  # Samples for action chunk
            # Misc
            encoder=None,
            frame_stack=ml_collections.config_dict.placeholder(int),  # unused
        )
    )
    return config
