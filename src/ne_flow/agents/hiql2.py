import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from ne_flow.flax_utils import ModuleDict, TrainState, nonpytree_field
from ne_flow.models import (
    GCActor,
    GCEncoder,
    GCValue,
    encoder_modules,
)


class HIQL2Agent(flax.struct.PyTreeNode):
    """
    Hierarchical IQL (H-IQL) Agent with Unified Representation Space.

    This agent uses a unified, explicit state encoder for all components.
    High-level policy plans in the latent space, and all critics and actors
    operate on these representations. The core value engine is from IQL (Q+V),
    and policies are trained with DDPG+BC.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    # --- 1. IQL Core Value Estimation (in latent space) ---

    def value_loss(self, batch, grad_params):
        """Computes the IQL V-function loss."""
        # Get target Q-value from the target critic network (no gradients).
        q1_t, q2_t = self.network.select("target_critic")(
            batch["observations"], batch["value_goals"], batch["actions"]
        )
        q_t = jnp.minimum(q1_t, q2_t)

        # Compute current V-value. Gradients flow through V-network.
        # N.B.: Using ensemble for V-function as explored in the new source code.
        v1, v2 = self.network.select("value")(
            batch["observations"], batch["value_goals"], params=grad_params
        )

        # Compute expectile loss for both V-functions.
        adv1 = q_t - v1
        adv2 = q_t - v2
        value_loss1 = self.expectile_loss(adv1, adv1, self.config["expectile"]).mean()
        value_loss2 = self.expectile_loss(adv2, adv2, self.config["expectile"]).mean()
        value_loss = value_loss1 + value_loss2

        v_statis = (v1 + v2) / 2
        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v_statis.mean(),
            "v_max": v_statis.max(),
            "v_min": v_statis.min(),
        }

    def critic_loss(self, batch, grad_params):
        """Computes the IQL Q-function (critic) loss."""
        # Compute TD target using the current (non-target) V-function.
        # Average the two V-functions for a more stable target.
        next_v1, next_v2 = self.network.select("value")(
            batch["next_observations"], batch["value_goals"]
        )
        next_v = (next_v1 + next_v2) / 2.0
        target_q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v

        # Compute current Q-values. Gradients flow through Q-network.
        q1, q2 = self.network.select("critic")(
            batch["observations"],
            batch["value_goals"],
            batch["actions"],
            params=grad_params,
        )
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": target_q.mean(),
            "q_max": target_q.max(),
            "q_min": target_q.min(),
        }

    # --- 2. Hierarchical Policy Extraction (in latent space) ---

    def low_actor_loss(self, batch, grad_params):
        """Computes the low-level actor loss using DDPG+BC."""
        # --- 1. DDPG Loss (Gradient Ascent through Actor) ---
        dist = self.network.select("low_actor")(
            batch["observations"], batch["low_actor_goals"], params=grad_params
        )
        pred_actions = jnp.clip(dist.mode(), -1, 1)

        # NOTE: The critic's target is the subgoal 'w_k'.
        q1, q2 = self.network.select("critic")(
            batch["observations"], batch["low_actor_goals"], pred_actions
        )
        q = jnp.minimum(q1, q2)
        q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)

        # --- 2. Advantage-Weighted BC Loss ---
        # 计算数据集动作的 Q 值 (Q(s, a_data))
        q1_data, q2_data = self.network.select("critic")(
            batch["observations"], batch["low_actor_goals"], batch["actions"]
        )
        q_data = (q1_data + q2_data) / 2.0

        # 计算当前状态的 V 值 (V(s))
        v1, v2 = self.network.select("value")(
            batch["observations"], batch["low_actor_goals"]
        )
        v = (v1 + v2) / 2.0

        # 计算优势 A = Q(s, a_data) - V(s)
        adv = q_data - v

        # 计算 AWR 权重: w = exp(A / temperature)
        # 使用 stop_gradient 确保只更新 Actor，不更新产生 A 的 Critic/Value
        weight = jnp.exp(
            adv * self.config["low_awr_temperature"]
        )  # 建议 temperature 设为 0.1 ~ 10 之间，即 alpha
        weight = jnp.minimum(weight, 100.0)  # 裁剪以防数值爆炸
        weight = jax.lax.stop_gradient(weight)

        log_prob = dist.log_prob(batch["actions"])

        # 加权 BC Loss
        bc_loss = -(weight * log_prob).mean()

        # 总 Loss: DDPG项 + 加权BC项
        # 注意：通常使用了加权BC后，DDPG项的系数可以适当调小，或者 BC 的系数(low_bc_alpha)可以稍微调大
        actor_loss = q_loss + self.config["low_bc_alpha"] * bc_loss
        return actor_loss, {
            "loss": actor_loss,
            "q_loss": q_loss,
            "bc_loss": bc_loss,
            "q_mean": q.mean(),
            "adv_mean": adv.mean(),
            "weight_mean": weight.mean(),
        }

    def high_actor_loss(self, batch, grad_params):
        """Computes the high-level actor loss using DDPG + Advantage-Weighted BC."""

        # --- 1. DDPG Loss ---
        dist = self.network.select("high_actor")(
            batch["observations"], batch["high_actor_goals"], params=grad_params
        )
        pred_w_rep = dist.mode()

        # 评估 Actor 生成的子目标的价值
        v1_pred, v2_pred = self.network.select("value")(
            pred_w_rep, batch["high_actor_goals"]
        )
        v_pred = jnp.minimum(v1_pred, v2_pred)
        v_loss = -v_pred.mean() / jax.lax.stop_gradient(jnp.abs(v_pred).mean() + 1e-6)

        # --- 2. Advantage-Weighted BC Loss ---
        # 这里的 "Action" 是数据集中的真实子目标 (high_actor_targets)
        # 我们需要评估这个真实子目标好不好。

        # 数据集子目标的价值 V(s_next, g)
        v1_target, v2_target = self.network.select("value")(
            batch["high_actor_targets"], batch["high_actor_goals"]
        )
        v_target = (v1_target + v2_target) / 2.0

        # 当前状态的价值 V(s, g)
        v1_curr, v2_curr = self.network.select("value")(
            batch["observations"], batch["high_actor_goals"]
        )
        v_curr = (v1_curr + v2_curr) / 2.0

        # 优势 A = V(next) - V(curr)
        # 含义：如果这一步转移让价值升高了，说明这是个好动作，应该大力模仿
        adv = v_target - v_curr

        weight = jnp.exp(adv * self.config["high_awr_temperature"])
        weight = jnp.minimum(weight, 100.0)
        weight = jax.lax.stop_gradient(weight)

        log_prob = dist.log_prob(batch["high_actor_targets"])

        # 加权 BC
        bc_loss = -(weight * log_prob).mean()

        actor_loss = v_loss + self.config["high_bc_alpha"] * bc_loss
        return actor_loss, {
            "loss": actor_loss,
            "v_loss": v_loss,
            "bc_loss": bc_loss,
            "v_mean": v_pred.mean(),
            "adv_mean": adv.mean(),
            "weight_mean": weight.mean(),
        }

    # --- 3. Training and Inference ---

    # The total_loss, target_update, and update functions remain largely the same,
    # just ensuring all four losses are summed. I'm reusing the previous version's structure.
    @jax.jit
    def total_loss(self, batch, grad_params):
        """Compute the total loss."""
        # A single grad_params dict is passed, containing keys for all trainable modules:
        # 'modules_state_encoder', 'modules_value', 'modules_critic', 'modules_low_actor', 'modules_high_actor'
        info = {}

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f"value/{k}"] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params)
        for k, v in low_actor_info.items():
            info[f"low_actor/{k}"] = v

        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params)
        for k, v in high_actor_info.items():
            info[f"high_actor/{k}"] = v

        loss = value_loss + critic_loss + low_actor_loss + high_actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, _ = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params)

        # apply_loss_fn handles calculating gradients for all parameters in grad_params
        # and applying them.
        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "critic")

        return self.replace(network=new_network, rng=new_rng), info

    # The HIQL implementation used latent representations here.
    # We are directly predicting a state, which is simpler but might be
    # challenging for image-based envs.
    # w_rep = high_dist.sample(seed=high_seed)
    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Sample actions hierarchically in the original state space."""
        high_dist = self.network.select("high_actor")(
            observations, goals, temperature=temperature
        )
        subgoals = high_dist.mode()

        low_dist = self.network.select("low_actor")(
            observations, subgoals, temperature=temperature
        )
        actions = low_dist.mode()

        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new agent with the unified and NORMALIZED encoder architecture."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        action_dim = ex_actions.shape[-1]
        obs_dim = ex_observations.shape[-1]
        ex_goals = ex_observations

        # --- Define Encoders based on environment type ---
        gc_encoder = None
        if config["encoder"] is not None:
            # TODO: This setup is currently NOT compatible with image-based envs
            # because the high-level actor predicts a high-dimensional state, which
            # is infeasible. This code is tailored for state-based envs like Pointmaze.
            # For image envs, one would need to revert to a representation-based approach.
            encoder_module = encoder_modules[config["encoder"]]()
            gc_encoder = GCEncoder(concat_encoder=encoder_module)

        # --- Define Core Modules ---
        value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=True,  # Using Twin V-functions for stability
            gc_encoder=copy.deepcopy(gc_encoder),
        )

        critic_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=True,
            gc_encoder=copy.deepcopy(gc_encoder),
        )

        low_actor_def = GCActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            const_std=config["const_std"],
            gc_encoder=copy.deepcopy(gc_encoder),
        )

        high_actor_def = GCActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=obs_dim,  # Predicts a coordinate
            const_std=config["const_std"],
            gc_encoder=copy.deepcopy(gc_encoder),
        )

        # --- Initialize Networks ---
        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(
                copy.deepcopy(critic_def),
                (ex_observations, ex_goals, ex_actions),
            ),
            low_actor=(low_actor_def, (ex_observations, ex_goals)),
            high_actor=(high_actor_def, (ex_observations, ex_goals)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)

        # Using the standard Adam optimizer without gradient clipping for now.
        network_tx = optax.adam(learning_rate=config["lr"])

        # This will create params for all modules defined in `networks`
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_critic"] = params["modules_critic"]

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name="hiql2",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            low_awr_temperature=3.0,  # 低层 AWR 温度
            high_awr_temperature=3.0,  # 高层 AWR 温度 (越大区分度越高，但也越不稳定)
            low_bc_alpha=0.1,  # Low-level BC coefficient in DDPG+BC.
            high_bc_alpha=1.0,  # High-level BC coefficient in DDPG+BC.
            subgoal_steps=25,  # Subgoal steps.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(
                str
            ),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class="HGCDataset",  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(
                int
            ),  # Number of frames to stack.
        )
    )
    return config
