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

METRIC_PREFIX = "__METRIC__"
METRIC_SEPARATOR = "||"


def get_json_serializable(obj):
    """Recursively converts JAX/Numpy arrays in an object to Python native types."""
    if isinstance(obj, (jax.Array, np.ndarray)):
        return obj.item() if obj.size == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {k: get_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [get_json_serializable(elem) for elem in obj]
    else:
        return obj


def log_to_parent(metrics, step):
    """Helper to print metrics in a format the launcher can parse."""
    # metrics is a dict
    # format: __METRIC__ <json_dump> || step=<step>
    serializable_metrics = get_json_serializable(metrics)
    msg = f"{METRIC_PREFIX} {json.dumps(serializable_metrics)} {METRIC_SEPARATOR} step={step}"
    print(msg, flush=True)


def run_worker(_):
    """
    Worker process: Runs a single training job with specific evaluation points.
    Does NOT init wandb. Prints metrics to stdout.
    """
    # Set up display for headless rendering (if needed for env creation)
    if "DISPLAY" not in os.environ:
        from pyvirtualdisplay import Display

        display = Display(visible=0, size=(400, 400))
        display.start()

    # Note: We DO NOT setup wandb here.

    exp_name = get_exp_name(FLAGS.seed)
    full_save_dir = os.path.join(
        FLAGS.save_dir, "OGBench_Benchmark", FLAGS.run_group, exp_name
    )
    os.makedirs(full_save_dir, exist_ok=True)

    # Save flags
    flag_dict = get_flag_dict()
    with open(os.path.join(full_save_dir, "flags.json"), "w") as f:
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
    train_logger = CsvLogger(os.path.join(full_save_dir, "train.csv"))

    first_time = time.time()
    last_time = time.time()

    max_steps = max(FLAGS.train_steps, max(CHECKPOINTS))

    for i in tqdm.tqdm(
        range(1, max_steps + 1), smoothing=0.1, desc=f"Seed {FLAGS.seed}"
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

            # Send to parent
            log_to_parent(train_metrics, i)
            train_logger.log(train_metrics, step=i)

        # Benchmark Evaluation logic
        if i in CHECKPOINTS:
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices("cpu")[0])
            else:
                eval_agent = agent

            eval_metrics = {}
            overall_metrics = defaultdict(list)
            task_infos = (
                env.unwrapped.task_infos
                if hasattr(env.unwrapped, "task_infos")
                else env.task_infos
            )

            for task_id in range(1, len(task_infos) + 1):
                task_name = task_infos[task_id - 1]["task_name"]
                eval_info, _, _ = evaluate(
                    agent=eval_agent,
                    env=env,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=0,
                    video_frame_skip=1,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                )

                if "success" in eval_info:
                    overall_metrics["success"].append(eval_info["success"])
                    eval_metrics[f"evaluation/{task_name}_success"] = eval_info[
                        "success"
                    ]

            mean_success = np.mean(overall_metrics["success"])
            eval_metrics["evaluation/overall_success"] = mean_success

            # Send to parent
            log_to_parent(eval_metrics, i)

        # Save agent
        if i % FLAGS.save_interval == 0:
            save_agent(agent, full_save_dir, i)

    train_logger.close()


def run_launcher(_):
    """
    Launcher process: Orchestrates the sequential execution of seeds and aggregates results.
    Initializes ONE wandb run.
    """
    seeds = [int(s) for s in FLAGS.seeds]
    print(f"Starting Benchmark for Seeds: {seeds}")
    print(f"Checkpoints: {CHECKPOINTS}")

    # Initialize WandB ONCE
    exp_name = f"{FLAGS.env_name}_Benchmark"
    setup_wandb(project="OGBench_Benchmark", group=FLAGS.run_group, name=exp_name)

    all_results = {}  # {seed: {ckpt: score}}

    for seed in seeds:
        print(f"\n{'=' * 40}")
        print(f"LAUNCHING WORKER FOR SEED {seed}")
        print(f"{'=' * 40}")

        all_results[seed] = {}

        # Construct command
        cmd = [sys.executable, sys.argv[0]]
        cmd.extend(
            [
                "--worker_mode=True",
                f"--seed={seed}",
                f"--env_name={FLAGS.env_name}",
                f"--run_group={FLAGS.run_group}",
                f"--save_dir={FLAGS.save_dir}",  # Pass base save dir
                f"--agent={config_flags.get_config_filename(FLAGS['agent'])}",
                f"--eval_episodes={FLAGS.eval_episodes}",
            ]
        )

        # Run subprocess and stream output
        # Run subprocess and stream output
        # We DO NOT merge stderr here. stderr (tqdm, errors) goes directly to console.
        # stdout is reserved for our __METRIC__ messages and explicit prints.
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1) as proc:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                
                # Check for metric (robust search)
                if METRIC_PREFIX in line:
                    try:
                        # Parse: ... __METRIC__ <json> || step=<step>
                        # We split by prefix in case there's garbage before it
                        content = line.split(METRIC_PREFIX)[1]
                        parts = content.split(METRIC_SEPARATOR)
                        json_part = parts[0].strip()
                        step_part = parts[1].strip()
                        
                        metrics = json.loads(json_part)
                        step = int(step_part.split("=")[1])
                        
                        # Prefix keys with seed
                        prefixed_metrics = {f"seed_{seed}/{k}": v for k, v in metrics.items()}
                        
                        # Store benchmark results if applicable
                        if "evaluation/overall_success" in metrics and step in CHECKPOINTS:
                            score = metrics["evaluation/overall_success"]
                            all_results[seed][step] = score
                            print(f"-> Captured Benchmark Result: Seed {seed} @ {step} = {score:.4f}")
                        
                        # Log to WandB
                        wandb.log(prefixed_metrics, step=step)
                        
                    except Exception as e:
                        print(f"Error parsing metric line: {line} -> {e}")
                else:
                    # Normal log line from stdout (e.g. explicit prints from worker)
                    print(f"[Seed {seed}] {line}")

            if proc.returncode != 0:
                print(
                    f"Worker for seed {seed} finished with error code {proc.returncode}"
                )

    # 2. Aggregation & Final Reporting
    print("\n" + "=" * 50)
    print("GENERATING FINAL BENCHMARK REPORT")
    print("=" * 50)

    # Prepare data for WandB Table
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
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    app.run(main)
