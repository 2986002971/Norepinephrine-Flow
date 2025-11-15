from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from ne_flow.flax_utils import ModuleDict, TrainState, nonpytree_field
from ne_flow.models import (
    GCActor,
    GCDiscreteActor,
    GCEncoder,
    GCValue,
    Identity,
    encoder_modules,
)


class HIQLAblationAgent(flax.struct.PyTreeNode):  # Renamed for clarity
    """
    Hierarchical implicit Q-learning (HIQL) agent - Ablation Version.

    This version removes the goal representation layer 'phi' (goal_rep) and has
    the high-level policy operate directly in the raw observation space.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the IVL value loss."""
        # This function's logic remains the same, as V-functions operate
        # on raw states s and g, and their internal encoders handle the inputs.
        (next_v1_t, next_v2_t) = self.network.select("target_value")(
            batch["next_observations"], batch["value_goals"]
        )
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v_t

        (v1_t, v2_t) = self.network.select("target_value")(
            batch["observations"], batch["value_goals"]
        )
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v1_t
        q2 = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v2_t
        (v1, v2) = self.network.select("value")(
            batch["observations"], batch["value_goals"], params=grad_params
        )
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config["expectile"]).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config["expectile"]).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
        }

    def low_actor_loss(self, batch, grad_params):
        """Compute the low-level actor loss, operating on raw subgoals."""
        # The advantage calculation is unchanged.
        v1, v2 = self.network.select("value")(
            batch["observations"], batch["low_actor_goals"]
        )
        nv1, nv2 = self.network.select("value")(
            batch["next_observations"], batch["low_actor_goals"]
        )
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config["low_alpha"])
        exp_a = jnp.minimum(exp_a, 100.0)

        # CHANGE: No longer compute goal_reps manually. The low_actor's internal
        # GCEncoder will handle the raw subgoal batch["low_actor_goals"].
        # We also remove goal_encoded=True flag.
        dist = self.network.select("low_actor")(
            batch["observations"], batch["low_actor_goals"], params=grad_params
        )

        log_prob = dist.log_prob(batch["actions"])
        actor_loss = -(exp_a * log_prob).mean()

        # ... (logging info remains the same)
        actor_info = {
            "actor_loss": actor_loss,
            "adv": adv.mean(),
            "bc_log_prob": log_prob.mean(),
        }
        if not self.config["discrete"]:
            actor_info.update({"mse": jnp.mean((dist.mode() - batch["actions"]) ** 2)})
        return actor_loss, actor_info

    def high_actor_loss(self, batch, grad_params):
        """Compute the high-level actor loss, predicting raw subgoals."""
        # The advantage calculation is unchanged.
        v1, v2 = self.network.select("value")(
            batch["observations"], batch["high_actor_goals"]
        )
        nv1, nv2 = self.network.select("value")(
            batch["high_actor_targets"], batch["high_actor_goals"]
        )
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config["high_alpha"])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select("high_actor")(
            batch["observations"], batch["high_actor_goals"], params=grad_params
        )

        # CHANGE: The BC target is no longer a representation from goal_rep,
        # but the raw future state itself.
        target = batch["high_actor_targets"]
        log_prob = dist.log_prob(target)

        actor_loss = -(exp_a * log_prob).mean()

        # ... (logging info remains the same, mse is now in raw state space)
        return actor_loss, {
            "actor_loss": actor_loss,
            "adv": adv.mean(),
            "bc_log_prob": log_prob.mean(),
            "mse": jnp.mean((dist.mode() - target) ** 2),
        }

    # total_loss and update methods remain unchanged as they just sum the losses.
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f"value/{k}"] = v
        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params)
        for k, v in low_actor_info.items():
            info[f"low_actor/{k}"] = v
        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params)
        for k, v in high_actor_info.items():
            info[f"high_actor/{k}"] = v
        loss = value_loss + low_actor_loss + high_actor_loss
        return loss, info

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
        self.target_update(new_network, "value")
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions hierarchically, planning in the raw state space."""
        high_seed, low_seed = jax.random.split(seed)

        high_dist = self.network.select("high_actor")(
            observations, goals, temperature=temperature
        )

        # CHANGE: The high-level policy now directly outputs a raw subgoal state.
        raw_subgoal = high_dist.sample(seed=high_seed)

        # CHANGE: The normalization step is removed.
        # goal_reps = goal_reps / jnp.linalg.norm(...)

        # CHANGE: The low-level policy takes the raw subgoal directly.
        # The goal_encoded flag is removed (or implicitly False).
        low_dist = self.network.select("low_actor")(
            observations, raw_subgoal, temperature=temperature
        )
        actions = low_dist.sample(seed=low_seed)

        if not self.config["discrete"]:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent without the goal_rep network."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config["discrete"]:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # CHANGE: Determine the state dimension for the high-level actor's output.
        state_dim = ex_observations.shape[-1]

        # CHANGE: The 'goal_rep' network definition is completely removed.
        # goal_rep_def = ...

        # CHANGE: Define GCEncoders for each component without relying on a shared goal_rep_def.
        if config["encoder"] is not None:
            # For image-based envs
            encoder_module = encoder_modules[config["encoder"]]
            # V and low-level actor need to process two image-like inputs (s, g)
            value_encoder_def = GCEncoder(
                state_encoder=encoder_module(), goal_encoder=encoder_module()
            )
            target_value_def = GCEncoder(
                state_encoder=encoder_module(), goal_encoder=encoder_module()
            )
            low_actor_encoder_def = GCEncoder(
                state_encoder=encoder_module(), goal_encoder=encoder_module()
            )
            # High-level actor concatenates s and g, then processes with one encoder
            high_actor_encoder_def = GCEncoder(concat_encoder=encoder_module())
        else:
            # For state-based envs (like pointmaze)
            # V and policies can just concatenate the raw state vectors.
            value_encoder_def = GCEncoder(
                state_encoder=Identity(), goal_encoder=Identity()
            )
            target_value_encoder_def = GCEncoder(
                state_encoder=Identity(), goal_encoder=Identity()
            )
            low_actor_encoder_def = GCEncoder(
                state_encoder=Identity(), goal_encoder=Identity()
            )
            high_actor_encoder_def = (
                None  # Use actor's internal MLP on concatenated [s, g]
            )

        # --- Define value and actor networks ---
        value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=True,
            gc_encoder=value_encoder_def,
        )
        target_value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=True,
            gc_encoder=target_value_encoder_def,
        )

        if config["discrete"]:
            low_actor_def = GCDiscreteActor(...)  # Definition remains similar
        else:
            low_actor_def = GCActor(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
                const_std=config["const_std"],
                gc_encoder=low_actor_encoder_def,
            )

        # CHANGE: high_actor's action_dim is now state_dim.
        high_actor_def = GCActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=state_dim,  # Predicts a raw state
            const_std=config["const_std"],
            gc_encoder=high_actor_encoder_def,
        )

        # --- Initialize Networks ---
        # CHANGE: network_info no longer contains 'goal_rep'.
        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(target_value_def, (ex_observations, ex_goals)),
            low_actor=(low_actor_def, (ex_observations, ex_goals)),
            high_actor=(high_actor_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_value"] = params["modules_value"]

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name="hiql",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.7,  # IQL expectile.
            low_alpha=3.0,  # Low-level AWR temperature.
            high_alpha=3.0,  # High-level AWR temperature.
            subgoal_steps=25,  # Subgoal steps.
            rep_dim=10,  # Goal representation dimension.
            low_actor_rep_grad=False,  # Whether low-actor gradients flow to goal representation (use True for pixels).
            const_std=True,  # Whether to use constant standard deviation for the actors.
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
