import jax.numpy as jnp
from jax import lax

# --- Parameters for the test ---
MAX_TIMESTEPS = 20
ENSEMBLE_SIZE = 4  # E
TRAJ_ACT_LEN = 8  # H
INFERENCE_HORIZON = 2  # H_inf
ACTION_DIM = 2  # D
DECAY_RATE = 0.2  # k


# --- Mock Policy Predictions ---
def get_mock_prediction(planning_step_index):
    """Create a predictable, non-zero chunk."""
    return jnp.ones((TRAJ_ACT_LEN, ACTION_DIM), dtype=jnp.float32) * (
        planning_step_index + 1.0
    )


# --- Implementation 1: Reference (Giant Matrix) ---
def reference_aggregation_loop(predictions):
    """
    Simulates the aggregation over an episode using the large matrix method,
    which is inefficient but easy to verify.
    """
    num_planning_steps = len(predictions)
    all_time_actions = jnp.zeros(
        (MAX_TIMESTEPS, MAX_TIMESTEPS + TRAJ_ACT_LEN, ACTION_DIM)
    )

    final_actions = []
    prediction_idx = 0
    for t in range(MAX_TIMESTEPS):
        # If this is a planning step, a new prediction is made and stored.
        if t % INFERENCE_HORIZON == 0:
            if prediction_idx < num_planning_steps:
                prediction = predictions[prediction_idx]
                all_time_actions = all_time_actions.at[t, t : t + TRAJ_ACT_LEN, :].set(
                    prediction
                )
                prediction_idx += 1

        # Aggregate action for the current time step t
        candidate_slices = all_time_actions[:, t, :]

        # Filter out predictions that haven't been made yet (still zero)
        populated_mask = jnp.any(candidate_slices != 0, axis=-1)
        candidates = candidate_slices[populated_mask]

        if candidates.shape[0] == 0:
            final_action = jnp.zeros((ACTION_DIM,))
        else:
            # Weighted average (older predictions get higher weights)
            num_candidates = candidates.shape[0]
            # `candidates` are ordered oldest to newest, so this weighting is correct.
            weights = jnp.exp(-DECAY_RATE * jnp.arange(num_candidates))
            weights /= jnp.sum(weights)
            weights = jnp.expand_dims(weights, axis=-1)

            final_action = jnp.sum(candidates * weights, axis=0)
        final_actions.append(final_action)

    return jnp.stack(final_actions)


# --- Implementation 2: Efficient (Circular Buffer) ---
def efficient_aggregation_loop(predictions):
    """
    Simulates the aggregation using the memory-efficient circular buffer method.
    """
    num_planning_steps = len(predictions)

    # State for the efficient method
    chunk_history = jnp.zeros((ENSEMBLE_SIZE, TRAJ_ACT_LEN, ACTION_DIM))
    planning_step_counter = 0

    final_actions = []

    for k in range(num_planning_steps):  # k is the planning step index
        # 1. Update history with the new prediction
        new_chunk = predictions[k]
        history_ptr = planning_step_counter % ENSEMBLE_SIZE
        chunk_history = chunk_history.at[history_ptr].set(new_chunk)
        planning_step_counter += 1

        # 2. Generate actions for the next INFERENCE_HORIZON steps
        for j in range(INFERENCE_HORIZON):
            t = k * INFERENCE_HORIZON + j
            if t >= MAX_TIMESTEPS:
                break

            # 3. Collect candidate actions for time t
            candidate_actions_list = []
            num_valid_chunks_in_history = min(planning_step_counter, ENSEMBLE_SIZE)

            for i in range(num_valid_chunks_in_history):  # i is age, 0=newest
                action_idx = i * INFERENCE_HORIZON + j

                if action_idx < TRAJ_ACT_LEN:
                    # Read from the circular buffer
                    chunk_read_ptr = (history_ptr - i + ENSEMBLE_SIZE) % ENSEMBLE_SIZE
                    chunk = chunk_history[chunk_read_ptr]
                    candidate = chunk[action_idx]
                    candidate_actions_list.append(candidate)

            if not candidate_actions_list:
                final_action = jnp.zeros((ACTION_DIM,))
            else:
                # This list is ordered from newest to oldest
                candidates = jnp.stack(candidate_actions_list)

                # 4. Weighted average (older is higher weight)
                num_candidates = candidates.shape[0]
                # Reverse the arange to give higher weights to older actions (at the end of the list)
                reversed_arange = jnp.arange(num_candidates - 1, -1, -1)
                weights = jnp.exp(-DECAY_RATE * reversed_arange)
                weights /= jnp.sum(weights)
                weights = jnp.expand_dims(weights, axis=-1)

                final_action = jnp.sum(candidates * weights, axis=0)

            final_actions.append(final_action)

    return jnp.stack(final_actions)


# --- Implementation 3: Vectorized (lax.scan) ---
def vectorized_efficient_aggregation(predictions):
    num_planning_steps = predictions.shape[0]

    init_carry = {
        "chunk_history": jnp.zeros((ENSEMBLE_SIZE, TRAJ_ACT_LEN, ACTION_DIM)),
        "planning_step_counter": 0,
    }

    def step(carry, t):
        chunk_history = carry["chunk_history"]
        planning_step_counter = carry["planning_step_counter"]

        k = t // INFERENCE_HORIZON
        j = t % INFERENCE_HORIZON

        # 只在规划步写入，非规划步完全跳过写入
        def do_write(ch):
            write_ptr = planning_step_counter % ENSEMBLE_SIZE
            new_chunk = lax.cond(
                k < num_planning_steps,
                lambda: predictions[k],
                lambda: jnp.zeros((TRAJ_ACT_LEN, ACTION_DIM)),
            )
            return ch.at[write_ptr].set(new_chunk)

        chunk_history = lax.cond(j == 0, do_write, lambda ch: ch, chunk_history)

        # 写入后的计数器
        next_planning_step_counter = planning_step_counter + (j == 0)

        # 读取指针基于最新有效预测的位置
        # 当 planning_step_counter > 0 时，最新预测在 (planning_step_counter - 1) % ENSEMBLE_SIZE
        read_ptr = lax.cond(
            next_planning_step_counter > 0,
            lambda: (next_planning_step_counter - 1) % ENSEMBLE_SIZE,
            lambda: 0,
        )

        # 候选收集逻辑（修复后）
        i_vec = jnp.arange(ENSEMBLE_SIZE)
        num_valid_chunks = jnp.minimum(next_planning_step_counter, ENSEMBLE_SIZE)
        valid_mask_a = i_vec < num_valid_chunks

        action_idx_vec = i_vec * INFERENCE_HORIZON + j
        valid_mask_b = action_idx_vec < TRAJ_ACT_LEN

        valid_mask = valid_mask_a & valid_mask_b

        # 从最新到最老的环形读取
        chunk_read_ptr_vec = (read_ptr - i_vec + ENSEMBLE_SIZE) % ENSEMBLE_SIZE

        # 安全读取 + 掩码
        safe_action_idx = jnp.where(valid_mask, action_idx_vec, 0)
        all_candidates = chunk_history[chunk_read_ptr_vec, safe_action_idx, :]
        candidates = all_candidates * valid_mask[:, None]

        # 权重计算（保持与参考版一致）
        num_valid = jnp.sum(valid_mask)

        def compute_weighted_action():
            # 老预测（i大）权重高：weight_idx = num_valid - 1 - i
            weight_idx = num_valid - 1 - i_vec
            weights_pre = jnp.exp(-DECAY_RATE * weight_idx)
            weights = weights_pre * valid_mask
            weights = weights / jnp.sum(weights)
            return jnp.sum(candidates * weights[:, None], axis=0)

        final_action = lax.cond(
            num_valid > 0, compute_weighted_action, lambda: jnp.zeros((ACTION_DIM,))
        )

        next_carry = {
            "chunk_history": chunk_history,
            "planning_step_counter": next_planning_step_counter,
        }

        return next_carry, final_action

    _, final_actions = lax.scan(step, init_carry, jnp.arange(MAX_TIMESTEPS))
    return final_actions


# --- The Test ---
def test_aggregation_equivalence():
    print("\n--- Testing Temporal Aggregation Implementations ---")

    num_planning_steps = (MAX_TIMESTEPS + INFERENCE_HORIZON - 1) // INFERENCE_HORIZON

    # Create predictions in both formats
    mock_predictions_list = [get_mock_prediction(k) for k in range(num_planning_steps)]
    mock_predictions_array = jnp.stack(mock_predictions_list)

    print(f"Number of planning steps: {num_planning_steps}")
    print(f"Predictions array shape: {mock_predictions_array.shape}")

    print("Running reference implementation...")
    reference_output = reference_aggregation_loop(mock_predictions_list)

    print("Running efficient implementation...")
    efficient_output = efficient_aggregation_loop(mock_predictions_list)

    print("Running vectorized implementation...")
    vectorized_output = vectorized_efficient_aggregation(mock_predictions_array)

    print(f"Reference output shape: {reference_output.shape}")
    print(f"Efficient output shape: {efficient_output.shape}")
    print(f"Vectorized output shape: {vectorized_output.shape}")

    assert (
        reference_output.shape == efficient_output.shape == vectorized_output.shape
    ), "Output shapes do not match!"

    print("\nChecking reference vs efficient...")
    assert jnp.allclose(reference_output, efficient_output, atol=1e-6), (
        "Efficient output does not match reference!"
    )

    print("Checking reference vs vectorized...")
    assert jnp.allclose(reference_output, vectorized_output, atol=1e-6), (
        "Vectorized output does not match reference!"
    )

    print("\n--- All Tests Passed: Implementations are equivalent! ---")


if __name__ == "__main__":
    test_aggregation_equivalence()
