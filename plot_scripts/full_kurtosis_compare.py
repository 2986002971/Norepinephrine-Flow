import csv
import pickle
from typing import Dict, List

import flax
import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags
from ml_collections import config_flags
from tqdm import tqdm, trange

from ne_flow.agents import agents as agent_registry
from ne_flow.datasets import (
    GCChunkDataset,
    GCDataset,
    HGCChunkDataset,
    HGCDataset,
)
from ne_flow.env_utils import make_env_and_datasets
from ne_flow.evaluation import supply_rng

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "pointmaze-medium-navigate-v0", "Environment name.")
flags.DEFINE_integer("seed", 0, "Random seed for evaluation.")
flags.DEFINE_multi_string(
    "model_paths",
    None,
    "List of items in the form temp:/path/to/params.pkl for each model.",
)
flags.DEFINE_float(
    "best_temp",
    3.0,
    "Temperature whose model will interact with the environment.",
)
flags.DEFINE_integer("num_episodes", 1, "Number of rollout episodes.")
flags.DEFINE_integer("max_steps", 1000, "Maximum steps per episode.")
flags.DEFINE_string(
    "output_step_plot",
    "exp/step_kurtosis_compare.png",
    "Path to save per-step kurtosis chart.",
)
flags.DEFINE_string(
    "output_overall_plot",
    "exp/overall_kurtosis_compare.png",
    "Path to save overall kurtosis comparison chart (Violin Plot).",
)
flags.DEFINE_string(
    "output_step_csv",
    "exp/step_kurtosis_data.csv",
    "CSV path to save per-step kurtosis values.",
)
flags.DEFINE_string(
    "output_overall_csv",
    "exp/overall_kurtosis_data.csv",
    "CSV path to save overall kurtosis values.",
)

config_flags.DEFINE_config_file(
    "agent",
    "agents/ne.py",
    "Path to the agent config file.",
    lock_config=False,
)


DATASET_CLASSES = {
    "GCDataset": GCDataset,
    "GCChunkDataset": GCChunkDataset,
    "HGCDataset": HGCDataset,
    "HGCChunkDataset": HGCChunkDataset,
}


def parse_model_paths(model_flags: List[str]) -> Dict[float, str]:
    if not model_flags:
        raise ValueError("Please provide --model_paths flags (temp:/path).")
    mapping = {}
    for item in model_flags:
        if ":" not in item:
            raise ValueError(f"Invalid model spec '{item}', expected temp:/path.")
        temp_str, path = item.split(":", 1)
        temp = float(temp_str)
        mapping[temp] = path
    return mapping


def load_agent_from_path(agent, path: str):
    with open(path, "rb") as f:
        state = pickle.load(f)
    agent = flax.serialization.from_state_dict(agent, state["agent"])
    agent = agent.replace(_state={})
    return agent


def collect_action_distribution(agent, observation, goal, rng_key):
    obs = jnp.expand_dims(jnp.asarray(observation), axis=0)
    goal_arr = jnp.expand_dims(jnp.asarray(goal), axis=0)
    rng_key, high_rng = jax.random.split(rng_key)
    subgoal = agent.sample_high_actions(obs, goal_arr, rng=high_rng)
    rng_key, low_rng = jax.random.split(rng_key)
    horizon = agent.config["horizon_length"] if agent.config["action_chunking"] else 1
    action_dim = agent.config["action_dim"] * horizon
    candidates = agent.sample_flow_actions(
        "low_actor",
        obs,
        subgoal,
        action_dim,
        agent.config["low_num_samples"],
        low_rng,
    )
    candidates = np.array(np.squeeze(candidates, axis=0))
    return candidates, rng_key


def compute_kurtosis_from_array(data: np.ndarray) -> float:
    flat = data.reshape(data.shape[0], -1)
    mean = flat.mean(axis=0, keepdims=True)
    var = flat.var(axis=0, keepdims=True) + 1e-8
    standardized = (flat - mean) / np.sqrt(var)
    kurt = np.mean(standardized**4, axis=0)
    return float(np.mean(kurt))


def set_arial_font():
    plt.rcParams["font.family"] = "Arial"


def get_custom_colormap():
    """
    Creates a custom LinearSegmentedColormap based on the user's specification:
    Yellow (Low Gain) -> Light Green -> Blue (High Gain)
    """
    colors = [
        (228 / 255, 184 / 255, 127 / 255),  # Yellow
        (70 / 255, 170 / 255, 180 / 255),  # Cyan-Blue
        (48 / 255, 104 / 255, 141 / 255),  # Blue
    ]
    return mcolors.LinearSegmentedColormap.from_list("custom_gain", colors, N=100)


def plot_step_kurtosis(step_records: Dict[float, List[float]], output_path: str):
    set_arial_font()
    cmap = get_custom_colormap()

    plt.figure(figsize=(7, 4))

    temps = sorted(step_records.keys())
    num_temps = len(temps)

    for i, temp in enumerate(temps):
        values = step_records[temp]
        if not values:
            continue

        # Calculate color based on position in the sorted list
        if num_temps > 1:
            color = cmap(i / (num_temps - 1))
        else:
            color = cmap(0.5)

        plt.plot(
            range(1, len(values) + 1),
            values,
            label=f"temp={temp}",
            color=color,
            linewidth=2,
        )

    plt.xlabel("Decision step")
    plt.ylabel("Kurtosis")
    plt.title("Per-step Action Distribution Kurtosis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_kurtosis_violin(step_records: Dict[float, List[float]], output_path: str):
    set_arial_font()
    cmap = get_custom_colormap()

    temps = sorted(step_records.keys())
    data = [step_records[t] for t in temps]
    num_temps = len(temps)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Violin plot
    parts = ax.violinplot(data, showmeans=True, showmedians=False, showextrema=True)

    # Customize aesthetics
    for i, pc in enumerate(parts["bodies"]):
        if num_temps > 1:
            color = cmap(i / (num_temps - 1))
        else:
            color = cmap(0.5)

        pc.set_facecolor(color)
        pc.set_edgecolor("black")
        pc.set_alpha(0.9)  # High opacity to show the color clearly

    for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor("#333333")  # Dark grey lines
            vp.set_linewidth(1.0)

    ax.set_title("Kurtosis Distribution by Temperature", fontsize=14)
    ax.set_ylabel("Kurtosis", fontsize=12)
    ax.set_xlabel("Temperature / Gain", fontsize=12)

    ax.set_xticks(np.arange(1, len(temps) + 1))
    ax.set_xticklabels([str(t) for t in temps])

    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    ax.axhline(y=3.0, color="r", linestyle="--", alpha=0.5, label="Normal Dist. (3.0)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def save_step_csv(step_records: Dict[float, List[float]], path: str):
    temps = sorted(step_records.keys())
    max_len = max(len(step_records[t]) for t in temps)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + [f"temp={t}" for t in temps])
        for idx in range(max_len):
            row = [idx + 1]
            for temp in temps:
                values = step_records[temp]
                row.append(values[idx] if idx < len(values) else "")
            writer.writerow(row)


def save_overall_csv(kurtosis_map: Dict[float, float], path: str):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["temp", "kurtosis"])
        for temp in sorted(kurtosis_map.keys()):
            writer.writerow([temp, kurtosis_map[temp]])


def rollout_stepwise(env, best_agent, agents_map, num_episodes, max_steps, seed):
    per_step_kurt = {temp: [] for temp in agents_map}
    all_samples = {temp: [] for temp in agents_map}
    actor_fn = supply_rng(
        best_agent.sample_actions,
        rng=jax.random.PRNGKey(np.random.randint(0, 2**32)),
    )
    agent_rngs = {}
    for idx, temp in enumerate(agents_map.keys()):
        agent_rngs[temp] = jax.random.PRNGKey(seed + 3000 + idx)

    for episode_idx in trange(num_episodes, desc="Rollout Episodes", unit="episode"):
        try:
            observation, info = env.reset(options=dict(task_id=None, render_goal=False))
        except TypeError:
            observation, info = env.reset()
        goal = info.get("goal")
        if goal is None:
            raise ValueError("Environment reset did not provide 'goal' information.")

        done = False
        step = 0
        step_bar = tqdm(
            total=max_steps,
            desc=f"Episode {episode_idx + 1} Steps",
            unit="step",
            leave=False,
        )
        while not done:
            action = actor_fn(
                observations=observation,
                goals=goal,
                temperature=0,
            )
            action = np.array(action)
            if not best_agent.config.get("discrete"):
                action = np.clip(action, -1, 1)

            next_observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            step_bar.update(1)

            for temp, agent in agents_map.items():
                rng_key = agent_rngs[temp]
                candidates, rng_key = collect_action_distribution(
                    agent, observation, goal, rng_key
                )
                agent_rngs[temp] = rng_key
                per_step_kurt[temp].append(compute_kurtosis_from_array(candidates))
                all_samples[temp].append(candidates)

            observation = next_observation
            goal = info.get("goal", goal)

            if step >= max_steps:
                break
        step_bar.close()
    return per_step_kurt, all_samples


def main(_):
    config = FLAGS.agent.copy_and_resolve_references()
    model_paths = parse_model_paths(FLAGS.model_paths)
    if FLAGS.best_temp not in model_paths:
        raise ValueError(
            f"best_temp={FLAGS.best_temp} not found in provided model paths."
        )

    env, train_dataset, _ = make_env_and_datasets(
        FLAGS.env_name, frame_stack=config.get("frame_stack")
    )

    dataset_class = DATASET_CLASSES[config["dataset_class"]]
    wrapped_dataset = dataset_class(train_dataset, config)
    example_batch = wrapped_dataset.sample(1)

    agent_class = agent_registry[config["agent_name"]]
    agents_map = {}
    for temp, path in model_paths.items():
        cfg = config.copy_and_resolve_references()
        cfg["low_awr_temp"] = temp
        agent = agent_class.create(
            FLAGS.seed,
            example_batch["observations"],
            example_batch["actions"],
            cfg,
        )
        agent = load_agent_from_path(agent, path)
        agent = agent.replace(rng=jax.random.PRNGKey(FLAGS.seed))
        agents_map[temp] = agent

    best_agent = agents_map[FLAGS.best_temp]
    per_step_kurt, all_samples = rollout_stepwise(
        env,
        best_agent,
        agents_map,
        FLAGS.num_episodes,
        FLAGS.max_steps,
        FLAGS.seed,
    )

    overall_kurt = {}
    for temp, kurt_values in per_step_kurt.items():
        if not kurt_values:
            overall_kurt[temp] = float("nan")
        else:
            overall_kurt[temp] = np.mean(kurt_values)

    plot_step_kurtosis(per_step_kurt, FLAGS.output_step_plot)
    plot_kurtosis_violin(per_step_kurt, FLAGS.output_overall_plot)
    save_step_csv(per_step_kurt, FLAGS.output_step_csv)
    save_overall_csv(overall_kurt, FLAGS.output_overall_csv)

    for temp in sorted(overall_kurt.keys()):
        print(
            f"low_awr_temp={temp}: steps={len(per_step_kurt[temp])}, "
            f"overall_kurtosis={overall_kurt[temp]:.4f}"
        )


if __name__ == "__main__":
    app.run(main)
