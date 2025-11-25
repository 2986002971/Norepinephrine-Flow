import copy
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
        # 参数校验与维度处理
        if seed is None:
            raise ValueError("Seed required.")
        obs = jnp.expand_dims(observations, 0)  # [1, O]
        goal = jnp.expand_dims(goals, 0)  # [1, G]

        # 初始化状态 (如果为空)
        if not self._state:
            new_state = self.reset_inference_state(
                self.config["obs_dim"],
                self.config["action_dim"],
                self.config["horizon_length"],
            )
            self._state.update(new_state)

        state = self._state

        # ===== 1. 任务变化检测 =====
        if state["prev_goal"] is not None:
            goal_changed = (goal.shape != state["prev_goal"].shape) or (
                not jnp.allclose(goal, state["prev_goal"])
            )
            if goal_changed:
                # 任务改变，完全重置
                new_state = self.reset_inference_state(
                    self.config["obs_dim"],
                    self.config["action_dim"],
                    self.config["horizon_length"],
                )
                self._state.clear()
                self._state.update(new_state)
                state = self._state  # 重新指向新字典

        state["prev_goal"] = goal

        # ===== 2. High-Level Policy (Subgoal) =====
        # 逻辑：Subgoal 不需要每一步都变，通常每 H 步更新一次，或者每一步都更新也可以(更费时)
        # 这里假设保持原来的频率：每 H 步更新一次 Subgoal
        horizon = self.config["horizon_length"]
        update_subgoal = (state["high_step_counter"] % horizon) == 0

        rng, high_rng, low_rng = jax.random.split(seed, 3)

        if update_subgoal:
            state["current_subgoal"] = self.sample_high_actions(
                observations=obs, goals=goal, rng=high_rng
            )  # [1, obs_dim]
            state["high_step_counter"] = 0  # 重置计数器防止溢出

        state["high_step_counter"] += 1

        # ===== 3. Low-Level Policy (Action Chunk) =====
        # 逻辑：为了时间集成，我们需要每一步都预测一个新的 Chunk

        # 预测新的 Chunk [1, H*A]
        action_chunk_flat = self.sample_low_actions(
            observations=obs, subgoals=state["current_subgoal"], rng=low_rng
        )

        # Reshape: [1, H, A] -> [H, A] (去掉 Batch 维)
        action_dim = self.config["action_dim"]
        new_chunk = jnp.reshape(action_chunk_flat, (horizon, action_dim))

        # ===== 4. 计算时间集成 (JIT call) =====
        # 调用纯函数进行 Buffer 更新和平均
        # 注意：这里需要传入 decay_rate，可以放在 config 里
        decay_rate = self.config.get("temporal_decay", 0.1)

        new_buffer, new_ptr, current_action = self.compute_ensemble_action(
            state["ensemble_buffer"],
            state["ensemble_ptr"],
            new_chunk,
            state["episode_step"],
            decay_rate,
        )

        # ===== 5. 更新状态 =====
        state["ensemble_buffer"] = new_buffer
        state["ensemble_ptr"] = new_ptr
        state["episode_step"] += 1

        # 返回当前动作 [A]
        return current_action

    def reset_inference_state(self, obs_dim, action_dim, horizon):
        """重置推理状态，初始化集成 Buffer"""
        return {
            # 高层策略状态
            "current_subgoal": jnp.zeros((1, obs_dim)),
            "prev_goal": None,
            "high_step_counter": 0,  # 用于控制高层策略更新频率
            # 低层策略状态 (时间集成)
            # Buffer 形状: [H, H, A]
            "ensemble_buffer": jnp.zeros((horizon, horizon, action_dim)),
            "ensemble_ptr": 0,
            "episode_step": 0,  # 用于集成算法中的 mask 计算
        }

    @staticmethod
    @jax.jit
    def compute_ensemble_action(
        buffer: jnp.ndarray,
        write_ptr: int,
        new_chunk: jnp.ndarray,
        step_idx: int,
        decay_rate: float,
    ):
        """
        执行单步时间集成 (Temporal Ensemble / Action Aggregation).

        Args:
            buffer: [Horizon, Horizon, Action_Dim] 环形缓冲区
            write_ptr: int 当前写入位置
            new_chunk: [Horizon, Action_Dim] 模型预测出的新动作序列
            step_idx: int 当前episode的步数 (用于处理原本buffer为空的warmup阶段)
            decay_rate: float 指数衰减率

        Returns:
            new_buffer: 更新后的缓冲区
            new_write_ptr: 更新后的指针
            action: [Action_Dim] 当前时间步的合成动作
        """
        horizon = buffer.shape[0]

        # 1. 将新的预测块写入环形缓冲区
        # new_chunk[0] 是对 t 的预测, new_chunk[1] 是对 t+1 的预测...
        new_buffer = buffer.at[write_ptr].set(new_chunk)

        # 2. 收集当前时间步的所有候选动作
        # 我们需要从过去 Horizon 个预测中提取对当前时刻的预测。
        # 假设当前是 t。
        # 刚刚写入的块 (age=0): 位于 write_ptr，它的第 [0] 个元素是对 t 的预测。
        # 上一步写入的块 (age=1): 位于 write_ptr-1，它的第 [1] 个元素是对 t 的预测。
        # ...
        # i 步前写入的块 (age=i): 位于 write_ptr-i，它的第 [i] 个元素是对 t 的预测。

        ages = jnp.arange(horizon)  # [0, 1, ..., H-1]

        # 计算在环形 Buffer 中的读取索引
        read_indices = (write_ptr - ages + horizon) % horizon
        # 在 Chunk 中的动作索引就是 age (因为我们预测的是 future chunk)
        action_indices = ages

        # 提取候选动作: [Horizon, Action_Dim]
        candidates = new_buffer[read_indices, action_indices]

        # 3. 计算权重 (处理 Warmup 和 衰减)
        # 只有 age <= step_idx 的预测才是有效的（避免读取到了 Buffer 初始化时的 0）
        valid_mask = ages <= step_idx

        # 计算指数权重: exp(-k * t)，通常越新鲜的预测(age小，也就是chunk的前端)权重越大
        # 注意：这与你的测试代码逻辑略有不同，你的测试代码似乎给了旧预测更高的权重。
        # 现在的逻辑是：动作块的第0步(最准)权重最高，第H步(最远)权重最低。
        weights = jnp.exp(-decay_rate * ages)
        weights = weights * valid_mask  # 遮罩无效的历史

        # 归一化
        total_weight = jnp.sum(weights) + 1e-6  # 避免除零
        weights = weights / total_weight

        # 4. 加权求和
        final_action = jnp.sum(candidates * weights[:, None], axis=0)

        # 5. 更新指针
        new_write_ptr = (write_ptr + 1) % horizon

        return new_buffer, new_write_ptr, final_action

    # --- 5. Cascaded Best-of-N Sampling (JIT-compiled) ---
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
        q_min = jnp.minimum(q1, q2)
        q_scores = q_min.reshape(observations.shape[0], N)

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
            "current_subgoal": jnp.zeros((batch_size, obs_dim)),
            "prev_goal": None,
            "high_step_counter": 0,
            "ensemble_buffer": jnp.zeros((batch_size, horizon, action_dim)),
            "ensemble_ptr": 0,
            "episode_step": 0,
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
            subgoal_steps=25,
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
            horizon_length=8,
            temporal_decay=0.1,
            # Inference Params
            high_num_samples=32,
            low_num_samples=32,
            # Misc
            encoder=None,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
