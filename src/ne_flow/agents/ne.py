import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from ne_flow.flax_utils import ModuleDict, TrainState, nonpytree_field
from ne_flow.models import GCChunkCritic, GCGeometricValue, GCUnet


class NE_Agent(flax.struct.PyTreeNode):
    """
    Features:
    - No DDPG: Purely offline RL via Advantage-Weighted Flow Matching.
    - IQL Value Engine: Single V, Dual Q, Goal-Conditioned.
    - Action Chunking: Low-level policy predicts action sequences [H, A].
    - Conv+FiLM Critic: Handles temporal structure of action chunks.
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
        target_x: [Batch, Horizon, Action_dim]
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
        t_broadcast = t[:, :, None]  # [B, 1, 1]
        x_t = (1 - t_broadcast) * x_0 + t_broadcast * x_1
        target_velocity = x_1 - x_0  # [B, H, A]

        # 3. Predict velocity
        pred_velocity = network.select(module_name)(
            obs, goal, x_t, t, params=params
        )  # [B, H, A]

        return jnp.square(pred_velocity - target_velocity)

    # --- 1. IQL Value Engine (Single V, Dual Q) ---
    def value_loss(self, batch, grad_params):
        """
        IQL V-Loss: Expectile Regression.
        Objective: V(s, g) -> Expectile(Q(s, a_chunk, g))
        """
        # [Modify] 直接使用序列动作 [B, H, A]，GCChunkCritic 会处理
        actions = batch["actions"]

        # Target Q (Dual Q, no gradient)
        # 输入: obs[B, D], goal[B, D], actions[B, H, A]
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
            "v_max": v.max(),
            "v_min": v.min(),
        }

    def critic_loss(self, batch, grad_params):
        """
        IQL Q-Loss.
        Objective: Q(s, a_chunk, g) -> r_sum + gamma^H * V(s', g)
        """
        # [Modify] 保持动作维度 [B, H, A]
        actions = batch["actions"]

        if self.config["action_chunking"]:
            discount = self.config["discount"] ** self.config["horizon_length"]
            # Rewards summed over horizon
            rewards = jnp.sum(
                batch["rewards"]
                * (
                    self.config["discount"] ** jnp.arange(self.config["horizon_length"])
                ),
                axis=1,
            )
            next_obs = batch["next_observations"]  # s_{t+H}
        else:
            # 兼容非 chunking 模式 (H=1)
            discount = self.config["discount"]
            rewards = batch["rewards"]
            next_obs = batch["next_observations"]

        # V-Target
        next_v = self.network.select("value")(next_obs, batch["value_goals"])

        # TD Target
        target_q = rewards + discount * batch["masks"] * next_v
        target_q = jax.lax.stop_gradient(target_q)

        # Current Q (Dual Q) with Conv+FiLM
        # 输入: obs[B, D], goal[B, D], actions[B, H, A]
        q1, q2 = self.network.select("critic")(
            batch["observations"], batch["value_goals"], actions, params=grad_params
        )

        loss1 = jnp.mean((q1 - target_q) ** 2)
        loss2 = jnp.mean((q2 - target_q) ** 2)
        critic_loss = loss1 + loss2

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": 0.5 * (q1 + q2).mean(),
            "q_min": jnp.minimum(q1, q2).min(),
            "q_max": jnp.maximum(q1, q2).max(),
        }

    # --- 2. Weighted Flow Matching ---
    def actor_loss(self, batch, grad_params, rng):
        """
        Low-Level Policy: Predicts Action Chunk a_t:t+h.
        Advantage: A = Q(s, a_chunk, g) - V(s, g)
        """
        actions = batch["actions"]  # [B, H, A]

        # 1. Calculate Advantage
        v = self.network.select("value")(
            batch["observations"], batch["actor_goals"]
        )  # [B]

        # [Modify] 直接传入 3D 动作计算 Q
        q1, q2 = self.network.select("critic")(
            batch["observations"], batch["actor_goals"], actions
        )  # [B]
        q = jnp.minimum(q1, q2)

        adv = q - v  # [B]

        # 2. AWR weights
        weights = jnp.exp(adv * self.config["awr_temp"])
        weights = jnp.clip(weights, max=100.0)
        weights = jax.lax.stop_gradient(weights)  # [B]

        # 3. Get Squared Diff [B, H, A]
        sq_diff = self.compute_flow_matching_sq_diff(
            self.network,
            "actor",
            batch["observations"],
            batch["actor_goals"],
            actions,
            rng,
            grad_params,
        )  # [B, H, A]

        # 4. Loss calculation (masking valid steps)
        loss_per_step = jnp.mean(sq_diff, axis=-1)  # [B, H]
        step_weights = batch["valid"] * weights[:, None]
        masked_loss = step_weights * loss_per_step  # [B, H]
        loss = jnp.mean(masked_loss)  # 全局平均

        return loss, {
            "actor_loss": loss,
            "adv_mean": adv.mean(),
            "weights": weights.mean(),
            "loss_per_step_mean": loss_per_step.mean(),
        }

    # --- 3. Training Loop Boilerplate (Unchanged) ---
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        rng, actor_rng = jax.random.split(rng)
        info = {}

        v_loss, v_info = self.value_loss(batch, grad_params)
        for k, v in v_info.items():
            info[f"value/{k}"] = v

        c_loss, c_info = self.critic_loss(batch, grad_params)
        for k, v in c_info.items():
            info[f"critic/{k}"] = v

        l_loss, l_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in l_info.items():
            info[f"actor/{k}"] = v

        return v_loss + c_loss + l_loss, info

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

    # --- 4. Stateful Inference Interface ---
    def sample_actions(self, observations, goals, seed, temperature=1.0):
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
        # Input: [D] -> [1, D]
        obs = jnp.expand_dims(observations, 0)
        goal = jnp.expand_dims(goals, 0)
        state = self._state

        # Auto-Reset Logic
        if state.get("prev_goal") is not None:
            prev_goal = state["prev_goal"]
            goal_changed = (goal.shape != prev_goal.shape) or (
                not jnp.allclose(goal, prev_goal)
            )
            if goal_changed:
                state["chunk_step_idx"] = self.config["horizon_length"]  # Force replan

        state["prev_goal"] = goal

        # Replanning
        if state.get("chunk_step_idx", 999) >= self.config["horizon_length"]:
            # Sample: Returns [1, H, A]
            action_chunk = self.sample_low_actions(obs, goal, seed)
            state["action_chunk_buffer"] = action_chunk
            state["chunk_step_idx"] = 0

        # Execute
        # Buffer: [1, H, A] -> Current: [1, A]
        current_action = state["action_chunk_buffer"][:, state["chunk_step_idx"], :]
        state["chunk_step_idx"] += 1

        # Return [A] without batch dimension
        return current_action[0]

    # --- 5. Best-of-N Sampling ---
    def sample_flow_actions(
        self, module_name, obs, goal, horizon, action_dim, num_samples, rng
    ):
        """
        Generate N samples.
        Returns: [B, N, H, A] (NO FLATTENING)
        """
        batch_size = obs.shape[0]

        # Repeat for samples: [B, D] -> [B, N, D]
        obs_rep = jnp.repeat(obs[:, None, :], num_samples, axis=1)
        goal_rep = jnp.repeat(goal[:, None, :], num_samples, axis=1)

        # Initialize action sequence: [B, N, H, A]
        x = jax.random.normal(rng, (batch_size, num_samples, horizon, action_dim))

        steps = self.config["flow_steps"]
        dt = 1.0 / steps

        for i in range(steps):
            t_val = i / steps
            t = jnp.full((batch_size, num_samples, 1), t_val)

            # Flatten batch for network: [B*N, ...]
            obs_flat = obs_rep.reshape(-1, obs_rep.shape[-1])
            goal_flat = goal_rep.reshape(-1, goal_rep.shape[-1])
            x_flat = x.reshape(-1, horizon, action_dim)  # [B*N, H, A]
            t_flat = t.reshape(-1, 1)

            vel_flat = self.network.select(module_name)(
                obs_flat, goal_flat, x_flat, t_flat
            )  # Output: [B*N, H, A]

            # Reshape back: [B*N, H, A] -> [B, N, H, A]
            vel = vel_flat.reshape(batch_size, num_samples, horizon, action_dim)
            x = x + vel * dt

        # [Modify] Return structured [B, N, H, A]
        return x

    @jax.jit
    def sample_low_actions(self, observations, goals, rng):
        """
        Low-Level Sampling.
        Returns: [B, H, A] (Structured Action Chunk)
        """
        N = self.config["low_num_samples"]
        horizon = self.config["horizon_length"]
        action_dim = self.config["action_dim"]

        # 1. Get Candidates: [B, N, H, A]
        candidates = self.sample_flow_actions(
            "actor", observations, goals, horizon, action_dim, N, rng
        )
        candidates = jnp.clip(candidates, -1.0, 1.0)

        # 2. Q Evaluation
        batch_size = observations.shape[0]
        obs_dim = observations.shape[-1]
        goal_dim = goals.shape[-1]

        # Flatten batch and samples: [B*N, ...]
        flat_obs = jnp.repeat(observations[:, None, :], N, axis=1).reshape(-1, obs_dim)
        flat_goals = jnp.repeat(goals[:, None, :], N, axis=1).reshape(-1, goal_dim)

        # [Modify] Flatten candidates to [B*N, H, A] for Critic
        flat_actions = candidates.reshape(-1, horizon, action_dim)

        # Critic takes structured actions
        q1, q2 = self.network.select("critic")(flat_obs, flat_goals, flat_actions)
        flat_q = (q1 + q2) / 2  # [B*N]
        q_scores = flat_q.reshape(batch_size, N)

        # 3. Select Best
        best_idx = jnp.argmax(q_scores, axis=1)  # [B]

        # Indexing in [B, N, H, A]
        # jnp.arange(B) chooses the batch, best_idx chooses the sample
        best_actions = candidates[jnp.arange(batch_size), best_idx]  # [B, H, A]

        return best_actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        action_dim = ex_actions.shape[-1]
        obs_dim = ex_observations.shape[-1]
        horizon = config["horizon_length"] if config["action_chunking"] else 1

        ex_goals = ex_observations

        # [Modify] 占位符不再是扁平的，而是 [1, H, A]
        ex_actions_seq = jnp.zeros((1, horizon, action_dim))
        ex_time = jnp.zeros((1, 1))

        # Networks Definition
        # V: MLP (GCValue)
        value_def = GCGeometricValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=False,
            gc_encoder=None,
        )

        # [Modify] Critic: Conv + FiLM (GCChunkCritic)
        critic_def = GCChunkCritic(
            hidden_dims=config["value_hidden_dims"],
            conv_dims=(64, 128, 256),
            layer_norm=config["layer_norm"],
            ensemble=True,
            gc_encoder=None,
        )

        # Actor: Unet
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
            # [Modify] Critic 输入现在是 ex_actions_seq [1, H, A]
            critic=(critic_def, (ex_observations, ex_goals, ex_actions_seq)),
            target_critic=(
                copy.deepcopy(critic_def),
                (ex_observations, ex_goals, ex_actions_seq),
            ),
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

        # Inference Setup
        config["obs_dim"] = obs_dim
        config["goal_dim"] = obs_dim  # Assuming goal dim = obs dim
        config["action_dim"] = action_dim
        config["horizon"] = horizon

        # Initialize stateful inference buffers in a mutable dict
        batch_size = 1
        inference_state = {
            # Buffer 保持 [1, H, A] 结构
            "action_chunk_buffer": jnp.zeros((batch_size, horizon, action_dim)),
            "chunk_step_idx": 999,  # Init to force replan
            "prev_goal": None,
        }

        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(**config),
            _state=inference_state,
        )


def get_config():
    # 保持原有的 config 结构，确保引入了 GCChunkCritic 所需的超参
    config = ml_collections.ConfigDict(
        dict(
            agent_name="neflow",
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512, 512),  # TODO: 待清理
            value_hidden_dims=(256, 256),
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
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal. 为了避免作弊，这里必须是 1.0，其他两项为 0.0
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
