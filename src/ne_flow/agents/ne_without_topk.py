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


class NE_without_topk(flax.struct.PyTreeNode):
    """
    NE Flow Agent WITHOUT Top-K Ranking and WITHOUT Temporal Ensemble.

    This is an ablation baseline.
    - Inference: Samples exactly 1 candidate from the Flow model and executes it directly.
    - No Critic/Value evaluation during inference.
    - No Temporal Ensemble (Open-loop execution of chunks).
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
        """
        batch_size = obs.shape[0]
        x_dim = target_x.shape[-1]

        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # 1. Sample Noise x0 and Time t
        x_0 = jax.random.normal(x_rng, (batch_size, x_dim))
        x_1 = target_x
        t = jax.random.uniform(t_rng, (batch_size, 1))

        # 2. Linear Interpolation
        x_t = (1 - t) * x_0 + t * x_1
        target_velocity = x_1 - x_0

        # 3. Predict Velocity
        pred_velocity = network.select(module_name)(obs, goal, x_t, t, params=params)

        # 4. Return Squared Difference
        return jnp.square(pred_velocity - target_velocity)

    # --- 1. IQL Value Engine (Single V, Dual Q) ---
    def value_loss(self, batch, grad_params):
        if self.config["action_chunking"]:
            actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            actions = batch["actions"]

        q1, q2 = self.network.select("target_critic")(
            batch["observations"], batch["value_goals"], actions
        )
        q_target = jnp.minimum(q1, q2)

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
        if self.config["action_chunking"]:
            actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
            discount = self.config["discount"] ** self.config["low_chunk_length"]
            rewards = jnp.sum(
                batch["rewards"]
                * (
                    self.config["discount"] ** jnp.arange(self.config["low_chunk_length"])
                ),
                axis=1,
            )
            next_obs = batch["next_observations"]
        else:
            actions = batch["actions"]
            discount = self.config["discount"]
            rewards = batch["rewards"]
            next_obs = batch["next_observations"]

        next_v = self.network.select("value")(next_obs, batch["value_goals"])

        target_q = rewards + discount * batch["masks"] * next_v
        target_q = jax.lax.stop_gradient(target_q)

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
        obs_concat = jnp.concatenate(
            [batch["observations"], batch["high_actor_targets"]], axis=0
        )
        goals_concat = jnp.concatenate(
            [batch["high_actor_goals"], batch["high_actor_goals"]], axis=0
        )

        v_all = self.network.select("value")(obs_concat, goals_concat)
        v_curr, v_next = jnp.split(v_all, 2, axis=0)

        adv = v_next - v_curr

        weights = jnp.exp(adv * self.config["high_beta"])
        weights = jnp.clip(weights, max=100.0)
        weights = jax.lax.stop_gradient(weights)

        sq_diff = self.compute_flow_matching_sq_diff(
            self.network,
            "high_actor",
            batch["observations"],
            batch["high_actor_goals"],
            batch["high_actor_targets"],
            rng,
            grad_params,
        )

        loss_per_sample = jnp.mean(sq_diff, axis=-1)
        loss = jnp.mean(weights * loss_per_sample)

        return loss, {
            "high_actor_loss": loss,
            "high_adv_mean": adv.mean(),
            "high_weights": weights.mean(),
        }

    def low_actor_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            actions = batch["actions"]

        v = self.network.select("value")(
            batch["observations"], batch["low_actor_goals"]
        )
        q1, q2 = self.network.select("critic")(
            batch["observations"], batch["low_actor_goals"], actions
        )
        q = jnp.minimum(q1, q2)

        adv = q - v

        weights = jnp.exp(adv * self.config["low_beta"])
        weights = jnp.clip(weights, max=100.0)
        weights = jax.lax.stop_gradient(weights)

        sq_diff = self.compute_flow_matching_sq_diff(
            self.network,
            "low_actor",
            batch["observations"],
            batch["low_actor_goals"],
            actions,
            rng,
            grad_params,
        )

        batch_size = sq_diff.shape[0]
        horizon = batch["valid"].shape[1]
        action_dim = sq_diff.shape[-1] // horizon

        sq_diff_reshaped = jnp.reshape(sq_diff, (batch_size, horizon, action_dim))
        loss_per_step = jnp.mean(sq_diff_reshaped, axis=-1)

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

    # --- 4. Stateful Inference Interface ---
    def sample_actions(
        self,
        observations,
        goals,
        seed,
        temperature=1.0,
    ):
        if seed is None:
            raise ValueError("Seed required.")
        obs = jnp.expand_dims(observations, 0)
        goal = jnp.expand_dims(goals, 0)

        if not self._state:
            new_state = self.reset_inference_state(
                self.config["obs_dim"],
                self.config["action_dim"],
                self.config["low_chunk_length"],
            )
            self._state.update(new_state)

        state = self._state

        # Check Goal Change
        if state["prev_goal"] is not None:
            goal_changed = (goal.shape != state["prev_goal"].shape) or (
                not jnp.allclose(goal, state["prev_goal"])
            )
            if goal_changed:
                new_state = self.reset_inference_state(
                    self.config["obs_dim"],
                    self.config["action_dim"],
                    self.config["low_chunk_length"],
                )
                self._state.clear()
                self._state.update(new_state)
                state = self._state

        state["prev_goal"] = goal

        # Hierarchical Logic
        traj_horizon = self.config["low_chunk_length"]
        subgoal_replan_interval = self.config["subgoal_replan_interval"]
        update_subgoal = (state["high_step_counter"] % subgoal_replan_interval) == 0

        rng, high_rng, low_rng = jax.random.split(seed, 3)

        if update_subgoal:
            # DIRECT SAMPLE (No Top-K)
            state["current_subgoal"] = self.sample_high_actions(
                observations=obs, goals=goal, rng=high_rng
            )
            state["high_step_counter"] = 0

        state["high_step_counter"] += 1

        # Low-Level Logic
        low_interval = self.config["low_chunk_replan_interval"]
        update_low = (state["low_step_counter"] % low_interval) == 0

        if update_low:
            # DIRECT SAMPLE (No Top-K)
            action_chunk_flat = self.sample_low_actions(
                observations=obs, subgoals=state["current_subgoal"], rng=low_rng
            )
            action_dim = self.config["action_dim"]
            new_chunk = jnp.reshape(action_chunk_flat, (traj_horizon, action_dim))
            state["low_action_chunk"] = new_chunk

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

    # --- 5. Direct Sampling (No Top-K) ---
    def sample_flow_actions(self, module_name, obs, goal, out_dim, num_samples, rng):
        """Helper: Generate N samples from a conditional flow model."""
        batch_size = obs.shape[0]
        # Although num_samples is likely 1, we keep the dim for consistency
        obs_rep = jnp.repeat(obs[:, None, :], num_samples, axis=1)
        goal_rep = jnp.repeat(goal[:, None, :], num_samples, axis=1)

        # Initial Noise x0 (RANDOM is correct, not zero)
        x = jax.random.normal(rng, (batch_size, num_samples, out_dim))

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
        Directly sample 1 subgoal.
        """
        rng = rng if rng is not None else self.rng
        rng, high_rng = jax.random.split(rng)

        # Force N=1
        N = 1
        subgoal_dim = observations.shape[-1]

        candidate_subgoals = self.sample_flow_actions(
            "high_actor", observations, goals, subgoal_dim, N, high_rng
        )  # [B, 1, subgoal_dim]

        # Squeeze the sample dimension: [B, 1, D] -> [B, D]
        return candidate_subgoals[:, 0, :]

    @jax.jit
    def sample_low_actions(self, observations, subgoals, rng=None):
        """
        Directly sample 1 action chunk.
        """
        rng = rng if rng is not None else self.rng
        rng, low_rng = jax.random.split(rng)

        # Force N=1
        N = 1
        action_dim = self.config["action_dim"] * self.config["low_chunk_length"]

        candidate_actions = self.sample_flow_actions(
            "low_actor", observations, subgoals, action_dim, N, low_rng
        )  # [B, 1, H*A]
        candidate_actions = jnp.clip(candidate_actions, -1.0, 1.0)

        # Squeeze the sample dimension: [B, 1, D] -> [B, D]
        return candidate_actions[:, 0, :]

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        obs_dim = ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]
        full_action_dim = action_dim * (
            config["low_chunk_length"] if config["action_chunking"] else 1
        )

        # Placeholders
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

        config["obs_dim"] = obs_dim
        config["action_dim"] = action_dim

        horizon = config["low_chunk_length"]
        batch_size = 1

        inference_state = {
            "current_subgoal": jnp.zeros((batch_size, obs_dim)),
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
            agent_name="neflow_notopk",
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
            subgoal_replan_interval=4,
            discrete=False,  # unused
            # Top-K Params REMOVED (Implicitly N=1)
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
            high_beta=3.0,
            low_beta=3.0,
            # Chunking Params
            action_chunking=True,
            low_chunk_length=8,
            low_chunk_replan_interval=4,
            # Inference Params
            high_num_samples=1,  # Force to 1
            low_num_samples=1,  # Force to 1
            # Misc
            encoder=None,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
