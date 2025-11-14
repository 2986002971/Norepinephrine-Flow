import copy
import functools
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from ne_flow.flax_utils import ModuleDict, TrainState, nonpytree_field
from ne_flow.models import (
    MLP,
    GCActor,
    GCEncoder,
    GCValue,
    Identity,
    LengthNormalize,
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
        # Explicitly encode all states/goals first
        s_rep = self.network.select("state_encoder")(batch["observations"])
        s_rep = jax.lax.stop_gradient(
            s_rep
        )  # TODO: ?Encoder is not trained through V here?
        g_rep = self.network.select("state_encoder")(batch["value_goals"])
        g_rep = jax.lax.stop_gradient(
            g_rep
        )  # Goals should not provide gradients to the encoder here

        q1, q2 = self.network.select("target_critic")(s_rep, g_rep, batch["actions"])
        q = jnp.minimum(q1, q2)

        # Encoder is trained through the 's_rep' input to the value function.
        s_rep_grad = self.network.select("state_encoder")(
            batch["observations"], params=grad_params
        )
        v = self.network.select("value")(s_rep_grad, g_rep, params=grad_params)

        # Loss uses target Q and current V for stability.
        value_loss = self.expectile_loss(q - v, q - v, self.config["expectile"]).mean()

        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    def critic_loss(self, batch, grad_params):
        """Computes the IQL Q-function (critic) loss."""
        # Explicitly encode. Gradients flow through encoders for s and s_next.
        s_rep = self.network.select("state_encoder")(
            batch["observations"], params=grad_params
        )
        s_next_rep = self.network.select("state_encoder")(batch["next_observations"])
        g_rep = self.network.select("state_encoder")(batch["value_goals"])
        s_next_rep, g_rep = jax.lax.stop_gradient((s_next_rep, g_rep))

        # TD target comes from the V-function.
        next_v = self.network.select("value")(s_next_rep, g_rep)
        target_q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v

        # Critic's encoder is Identity, so it takes reps directly.
        q1, q2 = self.network.select("critic")(
            s_rep, g_rep, batch["actions"], params=grad_params
        )
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": target_q.mean(),
            "q_max": target_q.max(),
            "q_min": target_q.min(),
        }

    # --- 2. Hierarchical Policy Extraction (in latent space) ---

    def low_actor_loss(self, batch, grad_params, rng=None):
        """Computes the low-level actor loss using DDPG+BC."""
        # Encode states and subgoals. Gradients flow to encoder through s_rep.
        s_rep = self.network.select("state_encoder")(batch["observations"])
        s_rep = jax.lax.stop_gradient(s_rep)
        w_rep = self.network.select("state_encoder")(batch["low_actor_goals"])
        w_rep = jax.lax.stop_gradient(w_rep)  # Subgoal doesn't train the encoder.

        # --- DDPG Loss Part ---
        dist = self.network.select("low_actor")(s_rep, w_rep, params=grad_params)
        pred_actions = jnp.clip(dist.mode(), -1, 1)

        # Q-value is computed for gradient ascent on the actor. NO grads to critic/encoder here.
        q1, q2 = self.network.select("critic")(s_rep, w_rep, pred_actions)
        q = jnp.minimum(q1, q2)
        q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)

        # --- BC Loss Part ---
        log_prob = dist.log_prob(batch["actions"])
        bc_loss = -(self.config["low_bc_alpha"] * log_prob).mean()

        actor_loss = q_loss + bc_loss
        return actor_loss, {
            "actor_loss": actor_loss,
            "actor_q_loss": q_loss,
            "actor_bc_loss": bc_loss,
            "q_mean": q.mean(),
        }

    def high_actor_loss(self, batch, grad_params, rng=None):
        """Computes the high-level actor loss using DDPG+BC."""
        # Encode states and goals. Gradients flow to encoder through s_rep.
        s_rep = self.network.select("state_encoder")(batch["observations"])
        s_rep = jax.lax.stop_gradient(s_rep)
        g_rep = self.network.select("state_encoder")(batch["high_actor_goals"])
        g_rep = jax.lax.stop_gradient(g_rep)

        # --- DDPG Loss Part ---
        dist = self.network.select("high_actor")(s_rep, g_rep, params=grad_params)
        pred_w_rep = dist.mode()

        # V-value is computed for gradient ascent. NO grads to value_fn/encoder here.
        # We evaluate the proposed subgoal representation's value.
        v = self.network.select("value")(pred_w_rep, g_rep)
        v_loss = -v.mean() / jax.lax.stop_gradient(jnp.abs(v).mean() + 1e-6)

        # --- BC Loss Part ---
        # The BC target is the representation of the actual k-step future state.
        target_w_rep = self.network.select("state_encoder")(batch["high_actor_targets"])
        target_w_rep = jax.lax.stop_gradient(target_w_rep)
        log_prob = dist.log_prob(target_w_rep)
        bc_loss = -(self.config["high_bc_alpha"] * log_prob).mean()

        actor_loss = v_loss + bc_loss
        return actor_loss, {
            "actor_loss": actor_loss,
            "actor_v_loss": v_loss,
            "actor_bc_loss": bc_loss,
            "v_mean": v.mean(),
        }

    # --- 3. Training and Inference ---

    # The total_loss, target_update, and update functions remain largely the same,
    # just ensuring all four losses are summed. I'm reusing the previous version's structure.
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        # A single grad_params dict is passed, containing keys for all trainable modules:
        # 'modules_state_encoder', 'modules_value', 'modules_critic', 'modules_low_actor', 'modules_high_actor'
        info = {}
        rng = rng if rng is not None else self.rng
        rng, low_actor_rng, high_actor_rng = jax.random.split(rng, 3)

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f"value/{k}"] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(
            batch, grad_params, low_actor_rng
        )
        for k, v in low_actor_info.items():
            info[f"low_actor/{k}"] = v

        high_actor_loss, high_actor_info = self.high_actor_loss(
            batch, grad_params, high_actor_rng
        )
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
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

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
        """Sample actions hierarchically in the latent space."""
        high_seed, low_seed = jax.random.split(seed)

        # 1. Encode states and goals into representations. The encoder now normalizes them.
        s_rep = self.network.select("state_encoder")(observations)
        g_rep = self.network.select("state_encoder")(goals)

        # 2. High-level policy proposes a subgoal representation
        high_dist = self.network.select("high_actor")(
            s_rep, g_rep, temperature=temperature
        )
        w_rep = high_dist.sample(seed=high_seed)

        # CRITICAL: Normalize the actor's output during inference to ensure it lies
        # on the same hypersphere as the training targets. This is a robust practice
        # inspired by the HIQL paper.
        w_rep = (
            w_rep
            / jnp.linalg.norm(w_rep, axis=-1, keepdims=True)
            * jnp.sqrt(w_rep.shape[-1])
        )

        # 3. Low-level policy proposes an action to reach that subgoal representation
        low_dist = self.network.select("low_actor")(
            s_rep, w_rep, temperature=temperature
        )
        actions = low_dist.sample(seed=low_seed)

        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new agent with the unified and NORMALIZED encoder architecture."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]
        rep_dim = config["rep_dim"]

        # --- 1. Define the Unified State Encoder with Length Normalization ---
        if config["encoder"] is not None:
            # For image-based envs, we construct an encoder that first uses an
            # IMPALA-style CNN and then normalizes the output.

            # We need to ensure the final output dimension of the base encoder is rep_dim.
            # We can do this by creating a partial with the correct mlp_hidden_dims.
            base_encoder_cls = encoder_modules[config["encoder"]]

            # Get the original mlp_hidden_dims, but replace the last element with rep_dim
            original_mlp_dims = list(
                base_encoder_cls.kwargs.get("mlp_hidden_dims", (512,))
            )
            original_mlp_dims[-1] = rep_dim

            encoder_module_base = functools.partial(
                base_encoder_cls, mlp_hidden_dims=tuple(original_mlp_dims)
            )

            state_encoder_def = nn.Sequential(
                [encoder_module_base(), LengthNormalize()]
            )

        else:  # For state-based envs, use a simple MLP encoder followed by normalization.

            class StateEncoder(nn.Module):
                @nn.compact
                def __call__(self, x):
                    net = nn.Sequential(
                        [
                            MLP(
                                hidden_dims=(*config["value_hidden_dims"], rep_dim),
                                activate_final=False,
                                name="EncoderMLP",
                            ),
                            LengthNormalize(),  # CRITICAL: Add normalization layer at the end
                        ]
                    )
                    return net(x)

            state_encoder_def = StateEncoder()

        # --- 2. Define Core Modules with Identity Encoders ---
        # All modules below will operate on representations of size `rep_dim`

        # Value Function: V(s_rep, g_rep)
        value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            # Critical: Use Identity since we provide reps manually
            gc_encoder=GCEncoder(state_encoder=Identity(), goal_encoder=Identity()),
        )

        # Critic (Q-function): Q(s_rep, a, g_rep)
        critic_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=True,  # Twin Q
            gc_encoder=GCEncoder(state_encoder=Identity(), goal_encoder=Identity()),
        )

        # Low-level Actor: pi^l(a | s_rep, w_rep)
        low_actor_def = GCActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            const_std=config["const_std"],
            gc_encoder=GCEncoder(state_encoder=Identity(), goal_encoder=Identity()),
        )

        # High-level Actor: pi^h(w_rep | s_rep, g_rep)
        high_actor_def = GCActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=rep_dim,  # Predicts a representation
            const_std=config["const_std"],
            gc_encoder=GCEncoder(state_encoder=Identity(), goal_encoder=Identity()),
        )

        # --- 3. Initialize Networks ---
        # Example inputs are now representations
        ex_rep = jnp.zeros((ex_observations.shape[0], rep_dim))

        network_info = dict(
            state_encoder=(state_encoder_def, ex_observations),
            value=(value_def, (ex_rep, ex_rep)),
            critic=(critic_def, (ex_rep, ex_rep, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_rep, ex_rep, ex_actions)),
            low_actor=(low_actor_def, (ex_rep, ex_rep)),
            high_actor=(high_actor_def, (ex_rep, ex_rep)),
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
            low_bc_alpha=3.0,  # Low-level BC coefficient in DDPG+BC.
            high_bc_alpha=3.0,  # High-level BC coefficient in DDPG+BC.
            subgoal_steps=25,  # Subgoal steps.
            rep_dim=10,  # Goal representation dimension.
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
