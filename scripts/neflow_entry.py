import json
import os
import random
import time
from collections import defaultdict

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

# 保留原有导入
from ne_flow.agents import agents
from ne_flow.datasets import Dataset, HGCChunkDataset
from ne_flow.env_utils import make_env_and_datasets
from ne_flow.flax_utils import restore_agent, save_agent
from ne_flow.log_utils import (
    CsvLogger,
    get_exp_name,
    get_flag_dict,
    get_wandb_video,
    setup_wandb,
)

# ============================================
# 1. FLAGS定义（精简版）
# ============================================
FLAGS = flags.FLAGS

flags.DEFINE_string("run_group", "Debug", "Run group.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string(
    "env_name", "antmaze-large-navigate-v0", "Environment (dataset) name."
)
flags.DEFINE_string("save_dir", "exp/", "Save directory.")
flags.DEFINE_string("restore_path", None, "Restore path.")
flags.DEFINE_integer("restore_epoch", None, "Restore epoch.")

# 核心训练参数
flags.DEFINE_integer("offline_steps", 1000000, "Number of offline training steps.")
flags.DEFINE_integer("log_interval", 5000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 100000, "Evaluation interval.")
flags.DEFINE_integer("save_interval", 1000000, "Saving interval.")

# 评估参数（移除temperature和gaussian）
flags.DEFINE_integer("eval_tasks", None, "Number of tasks to evaluate (None for all).")
flags.DEFINE_integer("eval_episodes", 20, "Number of episodes for each task.")
flags.DEFINE_integer("video_episodes", 1, "Number of video episodes for each task.")
flags.DEFINE_integer("video_frame_skip", 3, "Frame skip for videos.")
flags.DEFINE_integer("eval_on_cpu", 1, "Whether to evaluate on CPU.")

# Agent配置（horizon_length从agent config读取）
config_flags.DEFINE_config_file("agent", "agents/hgc_iql.py", lock_config=False)


# ============================================
# 2. 核心评估函数（在main之前定义）
# ============================================
def evaluate_chunked_agent(
    agent,
    env,
    task_id,
    config,
    num_eval_episodes=20,
    num_video_episodes=1,
    video_frame_skip=3,
):
    """评估动作分块agent，支持多任务"""

    # 开始评估
    all_stats = defaultdict(list)
    renders = []
    action_dim = config["action_dim"]

    for episode_idx in tqdm.trange(num_eval_episodes + num_video_episodes):
        should_render = episode_idx >= num_eval_episodes

        # 重置环境
        obs, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
        goal = info.get("goal")
        goal = goal[None, :]  # 添加batch维度
        goal_frame = info.get("goal_rendered")

        episode_stats = defaultdict(list)
        step = 0
        rng = jax.random.PRNGKey(0)
        render = []

        # 主循环
        while True:
            rng, high_key, low_key = jax.random.split(rng, 3)
            obs = obs[None, :]  # 添加batch维度
            # =================== 核心评估逻辑 ===================
            # 采样高层子目标
            subgoal = agent.sample_high_actions(
                observations=obs, goals=goal, rng=high_key
            )

            # 采样底层动作块 [1, action_dim*horizon_length]
            action_chunk = agent.sample_low_actions(
                observations=obs, subgoals=subgoal, rng=low_key
            )
            action_chunk = np.array(action_chunk).reshape(-1, action_dim)

            # 执行完整动作块
            for chunk_action in action_chunk:
                next_obs, reward, terminated, truncated, info = env.step(chunk_action)
                done = terminated or truncated
                step += 1

                # 录制视频
                if should_render and (step % video_frame_skip == 0 or done):
                    frame = env.render().copy()
                    if goal_frame is not None:
                        render.append(np.concatenate([goal_frame, frame], axis=0))
                    else:
                        render.append(frame)

                # 记录统计信息（仅在评估episode，非视频episode）
                if not should_render:
                    for k, v in info.items():
                        if k.startswith("success") or k.startswith("distance"):
                            episode_stats[k].append(v)

                if done:
                    break

                obs = next_obs

            if done:
                break
            else:
                obs = next_obs

        # 收集结果
        if should_render:
            renders.append(np.array(render))
        else:
            for k, v in episode_stats.items():
                all_stats[k].append(np.mean(v))

    # 聚合统计
    final_stats = {k: np.mean(v) for k, v in all_stats.items()}

    return final_stats, renders  # 保持接口简洁


# ============================================
# 3. main函数（精简改造）
# ============================================
def main(_):
    # 保留虚拟显示设置
    if "DISPLAY" not in os.environ:
        from pyvirtualdisplay import Display

        display = Display(visible=0, size=(400, 400))
        display.start()

    # 设置日志
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project="OGBench", group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(
        FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name
    )
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    # 保存配置
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, "flags.json"), "w") as f:
        json.dump(flag_dict, f)

    # 从agent配置读取horizon_length
    config = FLAGS.agent
    config["horizon_length"] = config.get("horizon_length", 8)  # 默认值8

    # ========================================
    # 4. 数据集加载（核心改造）
    # ========================================
    env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=config.get("frame_stack", 1)
    )

    # 处理训练数据集
    base_train = Dataset.create(**train_dataset)
    train_dataset = HGCChunkDataset(base_train, config=config)

    # 处理验证数据集（如果存在）
    if val_dataset is not None:
        base_val = Dataset.create(**val_dataset)
        val_dataset = HGCChunkDataset(base_val, config=config)

    # ========================================
    # 5. Agent创建（适配sample_chunk）
    # ========================================
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # 获取示例批次（从sample_chunk）
    example_batch = train_dataset.sample_chunk(1)

    # 验证批次结构（调试用，可后续删除）
    print(f"[DEBUG] actions shape: {example_batch['actions'].shape}")
    print(f"[DEBUG] observations shape: {example_batch['observations'].shape}")

    agent_class = agents[config["agent_name"]]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch["observations"],
        example_batch["actions"],
        config,
    )

    # 恢复逻辑（保持不变）
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # ========================================
    # 6. 训练循环（核心改造）
    # ========================================
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, "train.csv"))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, "eval.csv"))

    first_time = time.time()
    last_time = time.time()

    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1)):
        # 关键改动：使用sample_chunk
        batch = train_dataset.sample_chunk(batch_size=config["batch_size"])

        # 更新agent（保持不变，假设已适配）
        agent, update_info = agent.update(batch)

        # 日志记录（保持不变）
        if i % FLAGS.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample_chunk(config["batch_size"])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update(
                    {f"validation/{k}": v for k, v in val_info.items()}
                )
            train_metrics["time/epoch_time"] = (
                time.time() - last_time
            ) / FLAGS.log_interval
            train_metrics["time/total_time"] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # ====================================
        # 7. 评估调用（核心改造）
        # ====================================
        if i == 1 or i % FLAGS.eval_interval == 0:
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices("cpu")[0])
            else:
                eval_agent = agent

            # 获取任务信息
            task_infos = (
                env.unwrapped.task_infos
                if hasattr(env.unwrapped, "task_infos")
                else env.task_infos
            )
            num_tasks = (
                FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
            )

            all_eval_metrics = {}
            renders = []

            for task_id in tqdm.trange(1, num_tasks + 1):
                task_name = task_infos[task_id - 1]["task_name"]

                # 调用新的评估函数
                eval_stats, cur_renders = evaluate_chunked_agent(
                    agent=eval_agent,
                    env=env,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes
                    if task_id == 1
                    else 0,  # 仅第一个任务录视频
                    video_frame_skip=FLAGS.video_frame_skip,
                )

                # 收集指标
                for k, v in eval_stats.items():
                    if k in ["success", "distance"]:
                        all_eval_metrics[f"evaluation/{task_name}_{k}"] = v

                # 收集视频
                renders.extend(cur_renders)

            # 计算总体指标
            if "success" in eval_stats:
                all_eval_metrics["evaluation/overall_success"] = np.mean(
                    [v for k, v in all_eval_metrics.items() if "success" in k]
                )

            # 视频记录
            if renders and FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=num_tasks)
                all_eval_metrics["video"] = video

            # 记录日志
            wandb.log(all_eval_metrics, step=i)
            eval_logger.log(all_eval_metrics, step=i)

        # 保存检查点
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    # 清理
    train_logger.close()
    eval_logger.close()

    # 保存wandb链接
    with open(os.path.join(FLAGS.save_dir, "token.tk"), "w") as f:
        f.write(wandb.run.url)


if __name__ == "__main__":
    app.run(main)
