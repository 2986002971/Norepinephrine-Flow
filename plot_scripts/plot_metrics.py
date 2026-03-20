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
from scipy import stats
from tqdm import tqdm

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

# === Configuration ===
flags.DEFINE_string("env_name", "pointmaze-medium-navigate-v0", "Environment name.")
flags.DEFINE_integer("seed", 0, "Random seed for evaluation.")
flags.DEFINE_multi_string("model_paths", None, "List: temp:/path/to/params.pkl")
flags.DEFINE_float("best_temp", 3.0, "Temperature for the interaction policy.")
flags.DEFINE_integer("num_episodes", 1, "Number of rollout episodes.")
flags.DEFINE_integer("max_steps", 200, "Maximum steps per episode.")

# Plot Output Paths
flags.DEFINE_string(
    "output_step_plot", "exp/multi_metric_dynamics.svg", "Step-wise dynamics plot."
)
flags.DEFINE_string(
    "output_violin_plot", "exp/multi_metric_violin.svg", "Overall stats violin plot."
)
flags.DEFINE_string("output_csv", "exp/metrics_data.csv", "Raw data CSV.")

config_flags.DEFINE_config_file(
    "agent",
    "src/ne_flow/agents/ne_without_temporal_ensemble.py",
    "Agent config file.",
    lock_config=False,
)

DATASET_CLASSES = {
    "GCDataset": GCDataset,
    "GCChunkDataset": GCChunkDataset,
    "HGCDataset": HGCDataset,
    "HGCChunkDataset": HGCChunkDataset,
}

# === Metric Definitions ===
# 定义我们要计算的指标，方便扩展
METRICS_CONFIG = {
    "entropy": {
        "label": r"Differential Entropy ($H$)",
        "color_idx": 0,
    },
    "variance": {
        "label": r"Mean Action Variance ($\sigma^2$)",
        "color_idx": 1,
    },
    "kurtosis": {
        "label": "Kurtosis",
        "color_idx": 2,
    },
    "peak_density": {
        "label": r"$\log_{10}$ Peak Density",
        "color_idx": 3,
    },
}


def parse_model_paths(model_flags: List[str]) -> Dict[float, str]:
    if not model_flags:
        raise ValueError("Please provide --model_paths flags (temp:/path).")
    mapping = {}
    for item in model_flags:
        if ":" not in item:
            raise ValueError(f"Invalid model spec '{item}', expected temp:/path.")
        temp_str, path = item.split(":", 1)
        mapping[float(temp_str)] = path
    return mapping


def load_agent_from_path(agent, path: str):
    with open(path, "rb") as f:
        state = pickle.load(f)
    agent = flax.serialization.from_state_dict(agent, state["agent"])
    agent = agent.replace(_state={})
    return agent


def collect_action_distribution(agent, observation, goal, rng_key):
    """Samples N actions from the flow model."""
    obs = jnp.expand_dims(jnp.asarray(observation), axis=0)
    goal_arr = jnp.expand_dims(jnp.asarray(goal), axis=0)
    rng_key, high_rng = jax.random.split(rng_key)
    subgoal = agent.sample_high_actions(obs, goal_arr, rng=high_rng)
    rng_key, low_rng = jax.random.split(rng_key)

    # Generate candidates
    horizon = agent.config["low_chunk_length"] if agent.config["action_chunking"] else 1
    action_dim = agent.config["action_dim"] * horizon
    candidates = agent.sample_flow_actions(
        "low_actor", obs, subgoal, action_dim, agent.config["low_num_samples"], low_rng
    )
    candidates = np.array(
        np.squeeze(candidates, axis=0)
    )  # Shape: (Num_Samples, Action_Dim)
    return candidates, rng_key


def compute_metrics_from_samples(data: np.ndarray) -> Dict[str, float]:
    """
    Computes all statistical metrics for a given batch of action samples.
    Data shape: (N_samples, Action_Dim)
    """
    # 1. Pre-processing: Flatten for simple stats, Keep dims for KDE
    # Normalize checks
    if data.shape[0] < 5:
        return {k: 0.0 for k in METRICS_CONFIG.keys()}

    # --- Simple Statistics ---
    # Variance: Average variance across action dimensions
    variance = np.var(data, axis=0).mean()

    # Kurtosis: Average kurtosis across action dimensions (Fisher's definition, normal=0)
    # Using Pearson's definition (normal=3) aligns better with "sharpness" intuition visually sometimes,
    # but scipy returns Fisher (excess). Let's Add 3 to make it comparable to normal dist=3 if desired,
    # or just keep raw. Here we keep raw Fisher (+0 is Normal) but many papers use Pearson (+3).
    # Let's return Pearson Kurtosis (Normal=3) for easier interpretation of "High > 3".
    kurt = stats.kurtosis(data, axis=0, fisher=False).mean()

    # --- Advanced Statistics via KDE (Kernel Density Estimation) ---
    # Since Flow Matching outputs can be non-Gaussian, we use KDE.
    # We add slight jitter to prevent singular matrix errors if actions are collapsed.
    jitter = np.random.normal(0, 1e-6, data.shape)
    data_safe = data + jitter

    try:
        # data_safe.T shape is (Dim, N_samples) for scipy gaussian_kde
        kde = stats.gaussian_kde(data_safe.T)

        # Evaluate PDF at the sample points to find stats
        # (Monte Carlo estimation of integrals)
        log_pdf_vals = kde.logpdf(data_safe.T)
        pdf_vals = np.exp(log_pdf_vals)

        # Entropy: H = - E[log p(x)] approx -mean(log_pdf_vals)
        entropy = -np.mean(log_pdf_vals)

        # Peak Density (Max Probability Mass proxy):
        # The maximum density value observed among the samples.
        # Since we sample from the distribution, the samples cluster at the mode.
        peak_density = np.max(pdf_vals)

    except np.linalg.LinAlgError:
        # Fallback if distribution is extremely degenerate (point mass)
        entropy = -10.0  # Very sure
        peak_density = 100.0  # Very high density

    return {
        "entropy": float(entropy),
        "variance": float(variance),
        "kurtosis": float(kurt),
        "peak_density": float(np.log10(peak_density + 1e-10)),
    }


# === Plotting Utilities ===


def set_style():
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linestyle"] = "--"


def get_gain_color(idx, total):
    """Returns a color from Yellow (Low Gain) to Blue (High Gain)."""
    # Custom colormap
    colors = [
        (228 / 255, 184 / 255, 127 / 255),  # Yellow
        (70 / 255, 170 / 255, 180 / 255),  # Cyan-Teal
        (48 / 255, 104 / 255, 141 / 255),  # Deep Blue
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_gain", colors, N=100)
    if total > 1:
        return cmap(idx / (total - 1))
    return cmap(0.5)


def plot_step_dynamics(records: Dict[float, Dict[str, List[float]]], output_path: str):
    set_style()
    metrics = list(METRICS_CONFIG.keys())
    temps = sorted(records.keys())

    # figsize 稍微调窄一点，看起来更紧凑
    fig, axes = plt.subplots(nrows=len(metrics), ncols=1, figsize=(6, 9), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for row_idx, metric_key in enumerate(metrics):
        ax = axes[row_idx]
        metric_info = METRICS_CONFIG[metric_key]

        for i, temp in enumerate(temps):
            values = records[temp][metric_key]
            ax.plot(
                range(1, len(values) + 1),
                values,
                label=f"$\\beta={temp}$",  # 使用 LaTeX 格式
                color=get_gain_color(i, len(temps)),
                linewidth=1.5,
                alpha=0.9,
            )

        # 字体加粗稍微轻一点，字号加大
        ax.set_ylabel(metric_info["label"], fontsize=11, fontweight="medium")
        ax.tick_params(axis="both", which="major", labelsize=10)

        # 这里的 Legend 去掉边框，并且只在第一个图显示
        if row_idx == 0:
            ax.legend(
                loc="upper right",
                fontsize="small",
                frameon=False,  # 去掉边框，更现代
                handlelength=1.5,
            )

    # X轴标签更加通用
    axes[-1].set_xlabel("Time Step ($t$)", fontsize=11, fontweight="medium")
    axes[-1].set_xlim(left=0)  # 确保从0开始对齐

    # 移除 suptitle，紧凑布局
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_combined_violin(
    records: Dict[float, Dict[str, List[float]]], output_path: str
):
    set_style()
    metrics = list(METRICS_CONFIG.keys())
    temps = sorted(records.keys())

    fig, axes = plt.subplots(nrows=len(metrics), ncols=1, figsize=(6, 9), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for row_idx, metric_key in enumerate(metrics):
        ax = axes[row_idx]
        metric_info = METRICS_CONFIG[metric_key]

        data_to_plot = [records[t][metric_key] for t in temps]

        # Violin
        parts = ax.violinplot(
            data_to_plot, showmeans=False, showextrema=False
        )  # mean由我们需要自己画线
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(get_gain_color(i, len(temps)))
            pc.set_alpha(0.8)
            pc.set_edgecolor("black")
            pc.set_linewidth(0.5)

        # Mean Trend Line (灰色虚线 + 实心点)
        means = [np.mean(d) for d in data_to_plot]
        ax.plot(
            range(1, len(temps) + 1),
            means,
            color="#777777",
            linestyle="--",
            linewidth=1.2,
            marker="o",
            markersize=4,
            markerfacecolor="white",
            markeredgecolor="#777777",
            markeredgewidth=1.0,
            zorder=5,
            alpha=0.9,
        )

        ax.set_ylabel(metric_info["label"], fontsize=11, fontweight="medium")
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(True, axis="y", alpha=0.2, linestyle="--")

    # X轴标签
    axes[-1].set_xlabel(
        "Neural Gain Coefficient ($\\beta$)", fontsize=11, fontweight="medium"
    )
    axes[-1].set_xticks(range(1, len(temps) + 1))
    axes[-1].set_xticklabels([str(t) for t in temps], fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


# === Main Logic ===


def rollout_and_measure(env, best_agent, agents_map, num_episodes, max_steps, seed):
    # Initialize storage: temp -> metric -> list of values
    history = {t: {m: [] for m in METRICS_CONFIG} for t in agents_map}

    actor_fn = supply_rng(best_agent.sample_actions, rng=jax.random.PRNGKey(seed))

    # Independent RNGs for each agent's sampling
    agent_rngs = {
        t: jax.random.PRNGKey(seed + 1000 + i) for i, t in enumerate(agents_map)
    }

    for episode_idx in range(num_episodes):
        try:
            obs, info = env.reset(options=dict(task_id=None, render_goal=False))
        except TypeError:
            obs, info = env.reset()
        goal = info.get("goal")

        done = False
        step = 0
        pbar = tqdm(total=max_steps, desc=f"Ep {episode_idx + 1}", leave=False)

        while not done and step < max_steps:
            # 1. Interact with environment using the BEST agent (fixed trajectory)
            # This ensures all gain models evaluate the SAME state context.
            act_exec = actor_fn(observations=obs, goals=goal, temperature=0)
            act_exec = np.clip(np.array(act_exec), -1, 1)  # Assuming continuous

            next_obs, _, term, trunc, info = env.step(act_exec)
            done = term or trunc

            # 2. Probe each model at this state
            for temp, agent in agents_map.items():
                # Sample distribution
                rng_key = agent_rngs[temp]
                candidates, rng_key = collect_action_distribution(
                    agent, obs, goal, rng_key
                )
                agent_rngs[temp] = rng_key

                # Compute all metrics
                metrics = compute_metrics_from_samples(candidates)

                # Store
                for m_key, m_val in metrics.items():
                    history[temp][m_key].append(m_val)

            obs = next_obs
            goal = info.get("goal", goal)
            step += 1
            pbar.update(1)

        pbar.close()

    return history


def main(_):
    # Setup
    config = FLAGS.agent.copy_and_resolve_references()
    model_paths = parse_model_paths(FLAGS.model_paths)

    env, train_dataset, _ = make_env_and_datasets(
        FLAGS.env_name, frame_stack=config.get("frame_stack")
    )

    # Load Agents
    dataset_class = DATASET_CLASSES[config["dataset_class"]]
    wrapped_dataset = dataset_class(train_dataset, config)
    example_batch = wrapped_dataset.sample(1)
    agent_class = agent_registry[config["agent_name"]]

    agents_map = {}
    print("Loading models...")
    for temp, path in model_paths.items():
        cfg = config.copy_and_resolve_references()
        cfg["low_awr_temp"] = temp
        agent = agent_class.create(
            FLAGS.seed, example_batch["observations"], example_batch["actions"], cfg
        )
        agents_map[temp] = load_agent_from_path(agent, path)

    # Rollout
    print("Starting rollout analysis...")
    best_agent = agents_map[FLAGS.best_temp]  # The one that drives the car
    history = rollout_and_measure(
        env, best_agent, agents_map, FLAGS.num_episodes, FLAGS.max_steps, FLAGS.seed
    )

    # Plotting
    print("Generating plots...")
    plot_step_dynamics(history, FLAGS.output_step_plot)
    plot_combined_violin(history, FLAGS.output_violin_plot)

    print(
        f"Done! Plots saved to:\n- {FLAGS.output_step_plot}\n- {FLAGS.output_violin_plot}"
    )


if __name__ == "__main__":
    app.run(main)
