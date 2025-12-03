import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from ne_flow.agents import agents
from ne_flow.datasets import (
    Dataset,
    GCChunkDataset,
    GCDataset,
    HGCChunkDataset,
    HGCDataset,
)
from ne_flow.env_utils import make_env_and_datasets
from ne_flow.evaluation import evaluate
from ne_flow.flax_utils import restore_agent, save_agent
from ne_flow.log_utils import (
    CsvLogger,
    get_exp_name,
    get_flag_dict,
    setup_wandb,
)

FLAGS = flags.FLAGS

# Common Flags
flags.DEFINE_string("run_group", "Benchmark", "Run group for wandb.")
flags.DEFINE_string("env_name", "antmaze-large-navigate-v0", "Environment name.")
flags.DEFINE_string("save_dir", "exp/", "Save directory.")
flags.DEFINE_string("base_log_dir", "exp/", "Base directory for logs.")

# Launcher Flags
flags.DEFINE_list("seeds", [0, 1, 2, 3], "List of seeds to run sequentially.")
flags.DEFINE_bool("worker_mode", False, "Internal flag: run as a worker process.")
flags.DEFINE_string("result_file", "", "Internal flag: path to save worker results.")

# Worker Flags (Subset of main.py flags relevant for training)
flags.DEFINE_integer("seed", 0, "Random seed (overridden by launcher).")
flags.DEFINE_string("restore_path", None, "Restore path.")
flags.DEFINE_integer("restore_epoch", None, "Restore epoch.")
flags.DEFINE_integer("train_steps", 1000000, "Number of training steps.")
flags.DEFINE_integer("log_interval", 5000, "Logging interval.")
flags.DEFINE_integer("save_interval", 1000000, "Saving interval.")  # Save at the end
flags.DEFINE_integer("eval_episodes", 50, "Number of episodes for evaluation.")
flags.DEFINE_integer("video_episodes", 0, "No video for benchmark to save time.")
flags.DEFINE_float("eval_temperature", 0, "Actor temperature for evaluation.")
flags.DEFINE_float("eval_gaussian", None, "Action Gaussian noise for evaluation.")
flags.DEFINE_integer("eval_on_cpu", 1, "Whether to evaluate on CPU.")

config_flags.DEFINE_config_file("agent", "agents/gciql.py", lock_config=False)

# Benchmarking checkpoints
CHECKPOINTS = [800000, 900000, 1000000]


def run_worker(_):
    """
    Worker process: Runs a single training job with specific evaluation points.
    """
    # Set up display for headless rendering (if needed for env creation, though video is off)
    if "DISPLAY" not in os.environ:
        from pyvirtualdisplay import Display

        display = Display(visible=0, size=(400, 400))
        display.start()

    # Set up logger (Worker run)
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(
        project="OGBench_Benchmark", group=FLAGS.run_group, name=f"{exp_name}_worker"
    )

    FLAGS.save_dir = os.path.join(
        FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name
    )
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    # Save flags
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, "flags.json"), "w") as f:
        json.dump(flag_dict, f)

    # Set up environment and dataset
    config = FLAGS.agent
    env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=config["frame_stack"]
    )

    dataset_class = {
        "GCDataset": GCDataset,
        "GCChunkDataset": GCChunkDataset,
        "HGCDataset": HGCDataset,
        "HGCChunkDataset": HGCChunkDataset,
    }[config["dataset_class"]]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), config)

    # Initialize agent
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    if config["discrete"]:
        example_batch["actions"] = np.full_like(
            example_batch["actions"], env.action_space.n - 1
        )

    agent_class = agents[config["agent_name"]]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch["observations"],
        example_batch["actions"],
        config,
    )

    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, "train.csv"))

    benchmark_results = {}  # To store results for 800k, 900k, 1M

    first_time = time.time()
    last_time = time.time()

    # Ensure we run at least until the last checkpoint
    max_steps = max(FLAGS.train_steps, max(CHECKPOINTS))

    for i in tqdm.tqdm(
        range(1, max_steps + 1),
        smoothing=0.1,
        dynamic_ncols=True,
        desc=f"Seed {FLAGS.seed}",
    ):
        # Update agent
        batch = train_dataset.sample(config["batch_size"])
        agent, update_info = agent.update(batch)

        # Log training metrics
        if i % FLAGS.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            train_metrics["time/epoch_time"] = (
                time.time() - last_time
            ) / FLAGS.log_interval
            train_metrics["time/total_time"] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Benchmark Evaluation logic
        if i in CHECKPOINTS:
            print(f"[*] Executing Benchmark Evaluation at step {i}...")
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices("cpu")[0])
            else:
                eval_agent = agent

            # Evaluate
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            task_infos = (
                env.unwrapped.task_infos
                if hasattr(env.unwrapped, "task_infos")
                else env.task_infos
            )

            # Iterate over all tasks (standard evaluation doesn't specify tasks usually implies all)
            # We assume evaluating all tasks is correct for the benchmark.
            for task_id in tqdm.trange(1, len(task_infos) + 1, desc="Eval Tasks"):
                task_name = task_infos[task_id - 1]["task_name"]
                eval_info, _, _ = evaluate(
                    agent=eval_agent,
                    env=env,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,  # Fixed to 50
                    num_video_episodes=0,
                    video_frame_skip=1,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                )

                # Collect 'success' metric
                if "success" in eval_info:
                    overall_metrics["success"].append(eval_info["success"])
                    eval_metrics[f"evaluation/{task_name}_success"] = eval_info[
                        "success"
                    ]

            # Aggregated metric for this checkpoint
            mean_success = np.mean(overall_metrics["success"])
            eval_metrics["evaluation/overall_success"] = mean_success

            # Log to WandB
            wandb.log(eval_metrics, step=i)

            # Store for final report
            benchmark_results[i] = float(mean_success)
            print(f"[*] Step {i} Result: {mean_success:.4f}")

        # Save agent at the end or periodic
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()

    # Save benchmark results to file for the launcher to read
    if FLAGS.result_file:
        with open(FLAGS.result_file, "w") as f:
            json.dump(benchmark_results, f)

    wandb.finish()


def run_launcher(_):
    """
    Launcher process: Orchestrates the sequential execution of seeds and aggregates results.
    """
    seeds = [int(s) for s in FLAGS.seeds]
    print(f"Starting Benchmark for Seeds: {seeds}")
    print(f"Checkpoints: {CHECKPOINTS}")

    all_results = {}  # {seed: {ckpt: score}}

    # 1. Sequential Execution
    for seed in seeds:
        print(f"\n{'=' * 40}")
        print(f"LAUNCHING WORKER FOR SEED {seed}")
        print(f"{'=' * 40}")

        result_file = os.path.abspath(
            os.path.join(FLAGS.base_log_dir, f"benchmark_res_seed_{seed}.json")
        )

        # Construct command to call self in worker mode
        # We need to pass all current flags + worker_mode + specific seed
        cmd = [sys.executable, sys.argv[0]]

        # Pass through relevant flags.
        # We need to manually reconstruct the flag string for simplicity and safety.
        # Note: We assume this script is run from the root directory.
        cmd.extend(
            [
                "--worker_mode=True",
                f"--seed={seed}",
                f"--result_file={result_file}",
                f"--env_name={FLAGS.env_name}",
                f"--run_group={FLAGS.run_group}",
                f"--agent={config_flags.get_config_filename(FLAGS['agent'])}",  # Get the path of the config file
                f"--eval_episodes={FLAGS.eval_episodes}",
            ]
        )

        # Capture stdout/stderr to keep the main terminal somewhat clean but visible
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError:
            print(f"!! Error running seed {seed}. Check output above.")
            sys.exit(1)

        # Read result
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                data = json.load(f)
                # Keys in json are strings, convert back to int for logic
                all_results[seed] = {int(k): v for k, v in data.items()}
            os.remove(result_file)  # Cleanup
        else:
            print(f"!! Warning: No result file found for seed {seed}.")
            all_results[seed] = {}

    # 2. Aggregation & Final Reporting
    print("\n" + "=" * 50)
    print("GENERATING FINAL BENCHMARK REPORT")
    print("=" * 50)

    setup_wandb(
        project="OGBench_Benchmark", group=FLAGS.run_group, name="Benchmark_Summary"
    )

    # Prepare data for WandB Table
    # Columns: Seed, 800k, 900k, 1M, Seed_Avg
    table_data = []
    columns = ["Seed"] + [f"{k // 1000}k" for k in CHECKPOINTS] + ["Seed_Avg"]

    seed_averages = []

    for seed in seeds:
        row = [seed]
        scores = []
        for ckpt in CHECKPOINTS:
            val = all_results.get(seed, {}).get(ckpt, 0.0)
            scores.append(val)
            row.append(val)

        avg_score = np.mean(scores)
        seed_averages.append(avg_score)
        row.append(avg_score)
        table_data.append(row)

        # Log individual seed average as a metric? Maybe not needed if table exists.

    final_mean = np.mean(seed_averages)
    final_std = np.std(seed_averages)

    # Create WandB Table
    summary_table = wandb.Table(data=table_data, columns=columns)
    wandb.log({"benchmark_results_table": summary_table})

    # Log Scalar Metrics
    wandb.log(
        {
            "benchmark/final_mean_success": final_mean,
            "benchmark/final_std_success": final_std,
        }
    )

    # Also print to console for immediate verification
    print(f"Final Mean Success: {final_mean:.4f}")
    print(f"Final Std Success:  {final_std:.4f}")
    print("-" * 30)
    for row in table_data:
        print(f"Seed {row[0]}: {row[1:]}")

    wandb.finish()


def main(_):
    if FLAGS.worker_mode:
        run_worker(_)
    else:
        run_launcher(_)


if __name__ == "__main__":
    # JAX memory preallocation handling
    # When acting as a launcher, we don't need GPU memory.
    # When acting as a worker, JAX will take what it needs.
    # To be safe, if we are in launcher mode (no --worker_mode flag in raw args), hide GPU?
    # Actually, simple logic: The launcher process does not import/init JAX hard until needed.
    # But we imported jax at top level.
    # Let's set XLA_PYTHON_CLIENT_PREALLOCATE=false for everyone to be safe,
    # or rely on subprocess isolation which is robust.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    app.run(main)
