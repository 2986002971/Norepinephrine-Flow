import jax.numpy as jnp
from jax import lax

# --- Simplified Parameters: INFERENCE_HORIZON = 1, ENSEMBLE_SIZE = TRAJ_ACT_LEN ---
MAX_TIMESTEPS = 20
TRAJ_ACT_LEN = 8  # This is now also the buffer size
ACTION_DIM = 2
DECAY_RATE = 0.2


def get_mock_prediction(planning_step_index):
    return jnp.ones((TRAJ_ACT_LEN, ACTION_DIM)) * (planning_step_index + 1.0)


# --- Simplified Reference (Still Giant Matrix) ---
def giant_matrix_reference(predictions):
    """INFERENCE_HORIZON=1, so we write at every t."""
    num_preds = predictions.shape[0]
    all_time_actions = jnp.zeros(
        (MAX_TIMESTEPS, MAX_TIMESTEPS + TRAJ_ACT_LEN, ACTION_DIM)
    )

    final_actions = []
    for t in range(MAX_TIMESTEPS):
        if t < num_preds:
            all_time_actions = all_time_actions.at[t, t : t + TRAJ_ACT_LEN].set(
                predictions[t]
            )

        # Read column t: predictions[0][t], predictions[1][t-1], ..., predictions[t][0]
        candidates = all_time_actions[:, t, :]
        populated_mask = jnp.any(candidates != 0, axis=-1)
        candidates = candidates[populated_mask]

        if candidates.shape[0] == 0:
            final_action = jnp.zeros((ACTION_DIM,))
        else:
            num_candidates = candidates.shape[0]
            weights = jnp.exp(-DECAY_RATE * jnp.arange(num_candidates))
            weights /= jnp.sum(weights)
            final_action = jnp.sum(candidates * weights[:, None], axis=0)
        final_actions.append(final_action)

    return jnp.stack(final_actions)


# --- Simplified Efficient (Tiny Circular Buffer) ---
def efficient(predictions):
    """Buffer size = TRAJ_ACT_LEN. Each prediction is stored once."""
    num_preds = predictions.shape[0]
    buffer = jnp.zeros(
        (TRAJ_ACT_LEN, TRAJ_ACT_LEN, ACTION_DIM)
    )  # (buffer_pos, traj_idx, action)
    final_actions = []

    for t in range(MAX_TIMESTEPS):
        write_ptr = t % TRAJ_ACT_LEN
        if t < num_preds:
            buffer = buffer.at[write_ptr].set(predictions[t])

        # Collect candidates: for i from 0 to min(t, TRAJ_ACT_LEN-1)
        candidate_list = []
        max_age = min(t, TRAJ_ACT_LEN - 1)
        for i in range(max_age + 1):  # i = age
            buffer_idx = (write_ptr - i + TRAJ_ACT_LEN) % TRAJ_ACT_LEN
            action_idx = i  # Because INFERENCE_HORIZON=1 and j=0
            candidate_list.append(buffer[buffer_idx, action_idx])

        if not candidate_list:
            final_action = jnp.zeros((ACTION_DIM,))
        else:
            candidates = jnp.stack(candidate_list)
            num_candidates = candidates.shape[0]
            # Older predictions (higher i) get higher weight
            weights = jnp.exp(-DECAY_RATE * jnp.arange(num_candidates - 1, -1, -1))
            weights /= jnp.sum(weights)
            final_action = jnp.sum(candidates * weights[:, None], axis=0)

        final_actions.append(final_action)

    return jnp.stack(final_actions)


# --- Simplified Vectorized (lax.scan) ---
def vectorized(predictions):
    """State is just the buffer and write pointer."""
    num_preds = predictions.shape[0]

    def step(carry, t):
        buffer, write_ptr = carry

        # 1. Write current prediction if available
        new_chunk = lax.cond(
            t < num_preds,
            lambda: predictions[t],
            lambda: jnp.zeros((TRAJ_ACT_LEN, ACTION_DIM)),
        )
        buffer = buffer.at[write_ptr].set(new_chunk)

        # 2. Collect candidates for time t
        i_vec = jnp.arange(TRAJ_ACT_LEN)
        max_age = jnp.minimum(t, TRAJ_ACT_LEN - 1)
        valid_mask = i_vec <= max_age  # Only i <= max_age are valid

        # Read from newest (i=0) to oldest (i=max_age)
        buffer_idx_vec = (write_ptr - i_vec + TRAJ_ACT_LEN) % TRAJ_ACT_LEN
        action_idx_vec = i_vec  # Because j=0

        # Vectorized read + masking
        all_candidates = buffer[buffer_idx_vec, action_idx_vec, :]
        candidates = all_candidates * valid_mask[:, None]

        # 3. Weighted average
        num_valid = max_age + 1
        weight_idx = num_valid - 1 - i_vec  # Old -> high weight
        weights_raw = jnp.exp(-DECAY_RATE * weight_idx)
        weights = weights_raw * valid_mask
        weights /= jnp.sum(weights)

        final_action = jnp.sum(candidates * weights[:, None], axis=0)

        # Next state
        next_write_ptr = (t + 1) % TRAJ_ACT_LEN  # Move write pointer forward

        return (buffer, next_write_ptr), final_action

    init_carry = (jnp.zeros((TRAJ_ACT_LEN, TRAJ_ACT_LEN, ACTION_DIM)), 0)
    _, final_actions = lax.scan(step, init_carry, jnp.arange(MAX_TIMESTEPS))
    return final_actions


# --- Test ---
def test_simplified():
    print("\n=== Testing Simplified Temporal Aggregation (INFERENCE_HORIZON=1) ===")

    num_planning_steps = MAX_TIMESTEPS  # One per timestep
    predictions = jnp.stack([get_mock_prediction(k) for k in range(num_planning_steps)])

    print("Running simplified reference...")
    ref_out = giant_matrix_reference(predictions)

    print("Running simplified efficient...")
    eff_out = efficient(predictions)

    print("Running simplified vectorized...")
    vec_out = vectorized(predictions)

    print(f"Reference shape: {ref_out.shape}")
    print(f"Efficient shape: {eff_out.shape}")
    print(f"Vectorized shape: {vec_out.shape}")

    # First 10 timesteps comparison
    print("\nFirst 10 timesteps comparison:")
    for i in range(min(10, MAX_TIMESTEPS)):
        print(
            f"t={i}: ref={ref_out[i, 0]:.7f}, eff={eff_out[i, 0]:.7f}, vec={vec_out[i, 0]:.7f}"
        )

    assert jnp.allclose(ref_out, eff_out, atol=1e-6), "Efficient != Reference"
    assert jnp.allclose(ref_out, vec_out, atol=1e-6), "Vectorized != Reference"
    print("\n✅ All simplified implementations are equivalent!")


if __name__ == "__main__":
    test_simplified()
