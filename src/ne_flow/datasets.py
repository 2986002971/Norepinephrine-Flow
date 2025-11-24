import dataclasses
from functools import partial
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=("padding",))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(
        img, ((padding, padding), (padding, padding), (0, 0)), mode="edge"
    )
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=("padding",))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class.

    This class supports both regular datasets (i.e., storing both observations and next_observations) and
    compact datasets (i.e., storing only observations). It assumes 'observations' is always present in the keys. If
    'next_observations' is not present, it will be inferred from 'observations' by shifting the indices by 1. In this
    case, set 'valids' appropriately to mask out the last state of each trajectory.
    """

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert "observations" in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if "valids" in self._dict:
            (self.valid_idxs,) = np.nonzero(self["valids"] > 0)

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        if "valids" in self._dict:
            return self.valid_idxs[
                np.random.randint(len(self.valid_idxs), size=num_idxs)
            ]
        else:
            return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        return self.get_subset(idxs)

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if "next_observations" not in result:
            result["next_observations"] = self._dict["observations"][
                np.minimum(idxs + 1, self.size - 1)
            ]
        return result


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0


@dataclasses.dataclass
class GCDataset:
    """Dataset class for goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals (value_goals and actor_goals) from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.
    It also supports frame stacking and random-cropping image augmentation.

    It reads the following keys from the config:
    - discount: Discount factor for geometric sampling.
    - value_p_curgoal: Probability of using the current state as the value goal.
    - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
    - value_p_randomgoal: Probability of using a random state as the value goal.
    - value_geom_sample: Whether to use geometric sampling for future value goals.
    - actor_p_curgoal: Probability of using the current state as the actor goal.
    - actor_p_trajgoal: Probability of using a future state in the same trajectory as the actor goal.
    - actor_p_randomgoal: Probability of using a random state as the actor goal.
    - actor_geom_sample: Whether to use geometric sampling for future actor goals.
    - gc_negative: Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as the reward.
    - p_aug: Probability of applying image augmentation.
    - frame_stack: Number of frames to stack.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
        preprocess_frame_stack: Whether to preprocess frame stacks. If False, frame stacks are computed on-the-fly. This
            saves memory but may slow down training.
    """

    dataset: Dataset
    config: Any
    preprocess_frame_stack: bool = True

    def __post_init__(self):
        self.size = self.dataset.size

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset["terminals"] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        # Assert probabilities sum to 1.
        assert np.isclose(
            self.config["value_p_curgoal"]
            + self.config["value_p_trajgoal"]
            + self.config["value_p_randomgoal"],
            1.0,
        )
        assert np.isclose(
            self.config["actor_p_curgoal"]
            + self.config["actor_p_trajgoal"]
            + self.config["actor_p_randomgoal"],
            1.0,
        )

        if self.config["frame_stack"] is not None:
            # Only support compact (observation-only) datasets.
            assert "next_observations" not in self.dataset
            if self.preprocess_frame_stack:
                stacked_observations = self.get_stacked_observations(
                    np.arange(self.size)
                )
                self.dataset = Dataset(
                    self.dataset.copy(dict(observations=stacked_observations))
                )

    def sample(self, batch_size, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals (value_goals and actor_goals) from the dataset. They are
        stored in the keys 'value_goals' and 'actor_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config["frame_stack"] is not None:
            batch["observations"] = self.get_observations(idxs)
            batch["next_observations"] = self.get_observations(idxs + 1)

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config["value_p_curgoal"],
            self.config["value_p_trajgoal"],
            self.config["value_p_randomgoal"],
            self.config["value_geom_sample"],
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config["actor_p_curgoal"],
            self.config["actor_p_trajgoal"],
            self.config["actor_p_randomgoal"],
            self.config["actor_geom_sample"],
        )

        batch["value_goals"] = self.get_observations(value_goal_idxs)
        batch["actor_goals"] = self.get_observations(actor_goal_idxs)
        successes = (idxs == value_goal_idxs).astype(float)
        batch["masks"] = 1.0 - successes
        batch["rewards"] = successes - (1.0 if self.config["gc_negative"] else 0.0)

        if self.config["p_aug"] is not None and not evaluation:
            if np.random.rand() < self.config["p_aug"]:
                self.augment(
                    batch,
                    ["observations", "next_observations", "value_goals", "actor_goals"],
                )

        return batch

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample):
        """Sample goals for the given indices."""
        batch_size = len(idxs)

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(
                p=1 - self.config["discount"], size=batch_size
            )  # in [1, inf)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            traj_goal_idxs = np.round(
                (
                    np.minimum(idxs + 1, final_state_idxs) * distances
                    + final_state_idxs * (1 - distances)
                )
            ).astype(int)
        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal),
                traj_goal_idxs,
                random_goal_idxs,
            )

            # Goals at the current state.
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs
            )

        return goal_idxs

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate(
            [crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1
        )
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding))
                if len(arr.shape) == 4
                else arr,
                batch[key],
            )

    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        if self.config["frame_stack"] is None or self.preprocess_frame_stack:
            return jax.tree_util.tree_map(
                lambda arr: arr[idxs], self.dataset["observations"]
            )
        else:
            return self.get_stacked_observations(idxs)

    def get_stacked_observations(self, idxs):
        """Return the frame-stacked observations for the given indices."""
        initial_state_idxs = self.initial_locs[
            np.searchsorted(self.initial_locs, idxs, side="right") - 1
        ]
        rets = []
        for i in reversed(range(self.config["frame_stack"])):
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            rets.append(
                jax.tree_util.tree_map(
                    lambda arr: arr[cur_idxs], self.dataset["observations"]
                )
            )
        return jax.tree_util.tree_map(
            lambda *args: np.concatenate(args, axis=-1), *rets
        )


@dataclasses.dataclass
class HGCDataset(GCDataset):
    """Dataset class for hierarchical goal-conditioned RL.

    This class extends GCDataset to support high-level actor goals and prediction targets. It reads the following
    additional key from the config:
    - subgoal_steps: Subgoal steps (i.e., the number of steps to reach the low-level goal).
    """

    def sample(self, batch_size, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals from the dataset. The goals are stored in the keys
        'value_goals', 'low_actor_goals', 'high_actor_goals', and 'high_actor_targets'. It also computes the 'rewards'
        and 'masks' based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config["frame_stack"] is not None:
            batch["observations"] = self.get_observations(idxs)
            batch["next_observations"] = self.get_observations(idxs + 1)

        # Sample value goals.
        value_goal_idxs = self.sample_goals(
            idxs,
            self.config["value_p_curgoal"],
            self.config["value_p_trajgoal"],
            self.config["value_p_randomgoal"],
            self.config["value_geom_sample"],
        )
        batch["value_goals"] = self.get_observations(value_goal_idxs)

        successes = (idxs == value_goal_idxs).astype(float)
        batch["masks"] = 1.0 - successes
        batch["rewards"] = successes - (1.0 if self.config["gc_negative"] else 0.0)

        # Set low-level actor goals.
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        low_goal_idxs = np.minimum(
            idxs + self.config["subgoal_steps"], final_state_idxs
        )
        batch["low_actor_goals"] = self.get_observations(low_goal_idxs)

        # Sample high-level actor goals and set prediction targets.
        # High-level future goals.
        if self.config["actor_geom_sample"]:
            # Geometric sampling.
            offsets = np.random.geometric(
                p=1 - self.config["discount"], size=batch_size
            )  # in [1, inf)
            high_traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            high_traj_goal_idxs = np.round(
                (
                    np.minimum(idxs + 1, final_state_idxs) * distances
                    + final_state_idxs * (1 - distances)
                )
            ).astype(int)
        high_traj_target_idxs = np.minimum(
            idxs + self.config["subgoal_steps"], high_traj_goal_idxs
        )

        # High-level random goals.
        high_random_goal_idxs = self.dataset.get_random_idxs(batch_size)
        high_random_target_idxs = np.minimum(
            idxs + self.config["subgoal_steps"], final_state_idxs
        )

        # Pick between high-level future goals and random goals.
        pick_random = np.random.rand(batch_size) < self.config["actor_p_randomgoal"]
        high_goal_idxs = np.where(
            pick_random, high_random_goal_idxs, high_traj_goal_idxs
        )
        high_target_idxs = np.where(
            pick_random, high_random_target_idxs, high_traj_target_idxs
        )

        batch["high_actor_goals"] = self.get_observations(high_goal_idxs)
        batch["high_actor_targets"] = self.get_observations(high_target_idxs)

        if self.config["p_aug"] is not None and not evaluation:
            if np.random.rand() < self.config["p_aug"]:
                self.augment(
                    batch,
                    [
                        "observations",
                        "next_observations",
                        "value_goals",
                        "low_actor_goals",
                        "high_actor_goals",
                        "high_actor_targets",
                    ],
                )

        return batch


@dataclasses.dataclass
class GCChunkDataset(GCDataset):
    """目标条件动作块数据集

    批次数据结构（sample返回）:
    --------------------------------
    observations: np.ndarray
        起始观测状态，形状 [batch_size, obs_dim]
        对应时间步 s_t

    actions: np.ndarray
        动作序列，形状 [batch_size, horizon_length, action_dim]
        对应时间步 [a_t, a_{t+1}, ..., a_{t+H-1}]

    next_observations: np.ndarray
        H步后的观测状态，形状 [batch_size, obs_dim]
        对应时间步 s_{t+H}（钳位到轨迹边界）

    value_goals: np.ndarray
        价值目标状态，形状 [batch_size, goal_dim]
        用于价值函数训练的最终目标

    actor_goals: np.ndarray
        高层演员目标状态，形状 [batch_size, goal_dim]
        用于策略训练的最终目标

    rewards: np.ndarray
        奖励序列，形状 [batch_size, horizon_length]
        gc_negative=True:  [-1,...,-1, 0,0,...,0]（达成前惩罚，达成后中性）
        gc_negative=False: [0,...,0, 1,0,...,0]（仅在达成瞬间奖励）
        达成后的所有时间步奖励为0，避免干扰学习

    masks: np.ndarray
        价值目标达成掩码，形状 [batch_size]
        取值范围 {0.0, 1.0}
        若在horizon内达成value_goal则masks=0，否则为1
        用于屏蔽已达成目标的样本对价值函数的影响

    valid: np.ndarray
        动作时序有效性掩码，形状 [batch_size, horizon_length]
        取值范围 [0.0, 1.0]
        在actor_goal达成前（含达成时刻）为1，之后为0
        用于屏蔽无效动作对演员函数的影响

    """

    def __post_init__(self):
        super().__post_init__()
        assert "horizon_length" in self.config, "horizon_length未在配置中指定"
        self.horizon_length = self.config["horizon_length"]

    def sample(
        self,
        batch_size: int,
        idxs: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """采样一个动作块批次，包含完整的分层目标和时序有效性掩码"""

        # 1. 采样起始索引（允许任意位置，后续通过钳位和valid保证有效性）
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        # 2. 组装基础批次数据
        batch = self._assemble_base_batch(idxs)

        # 3. 采样所有目标相关数据和索引
        goal_data = self._sample_all_goals(idxs)
        batch.update(goal_data)

        return batch

    def _assemble_base_batch(self, idxs: np.ndarray) -> Dict[str, Any]:
        """组装基础批次数据：observations, actions, next_observations"""
        batch = {}

        # 计算轨迹边界索引（向量化）
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]

        # 起始观测（单帧）
        batch["observations"] = self.get_observations(idxs)

        # 动作序列索引 [batch, horizon_length]
        seq_indices = idxs[:, None] + np.arange(self.horizon_length)[None, :]
        seq_indices_clipped = np.minimum(seq_indices, final_state_idxs[:, None])

        # 安全地获取动作序列 [batch, horizon_length, action_dim]
        batch["actions"] = self.dataset["actions"][seq_indices_clipped]

        # horizon_length步后的下一观测（用于价值函数）
        next_idxs = np.minimum(idxs + self.horizon_length, final_state_idxs)

        batch["next_observations"] = self.get_observations(next_idxs)

        return batch

    def _sample_all_goals(self, idxs: np.ndarray) -> Dict[str, Any]:
        """采样所有目标相关数据和索引"""
        data = {}

        # 1. 价值目标（复用GCD逻辑）
        value_goal_idxs = self.sample_goals(
            idxs,
            self.config["value_p_curgoal"],
            self.config["value_p_trajgoal"],
            self.config["value_p_randomgoal"],
            self.config["value_geom_sample"],
        )
        data["value_goals"] = self.get_observations(value_goal_idxs)

        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]

        # 1. 判定是否是未来时刻
        is_future = value_goal_idxs >= idxs

        # 2. 判定是否在 Horizon 范围内
        horizon_limits = idxs + self.horizon_length
        in_horizon = value_goal_idxs < horizon_limits

        # 3. 判定是否在同一条轨迹 (目标索引不能超过当前轨迹的终点)
        same_traj = value_goal_idxs <= final_state_idxs

        # 综合判定
        value_goal_hit = is_future & in_horizon & same_traj

        # 计算目标相对于起始索引的步数
        # 对于未命中的样本，设goal_steps = horizon_length（永不触发奖励转换）
        goal_steps = np.where(
            value_goal_hit, value_goal_idxs - idxs, self.horizon_length
        )  # 关键：设为horizon_length，永远大于时间步矩阵

        # 生成时间步矩阵 (1, horizon_length)
        time_steps = np.arange(self.horizon_length)[None, :]

        # 计算奖励序列
        if self.config.get("gc_negative", True):
            # 模式A（惩罚-直到达成）：[-1, -1, ..., -1, 0, 0, ..., 0]
            # 达成前每一步惩罚，达成后中性
            data["rewards"] = np.where(time_steps < goal_steps[:, None], -1.0, 0.0)
        else:
            # 模式B（稀疏-仅在达成瞬间）：[0, 0, ..., 0, 1, 0, ..., 0]
            # 只有达成瞬间奖励，其他时间中性（包括达成后）
            data["rewards"] = np.where(time_steps == goal_steps[:, None], 1.0, 0.0)

        # 计算掩码：只要命中，整个序列都mask掉
        data["masks"] = 1.0 - value_goal_hit.astype(float)

        # 3. 演员目标（复用sample_goals）
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config["actor_p_curgoal"],
            self.config["actor_p_trajgoal"],
            self.config["actor_p_randomgoal"],
            self.config["actor_geom_sample"],
        )
        data["actor_goals"] = self.get_observations(actor_goal_idxs)

        # 计算演员目标达成的相对步数
        actor_goal_steps = actor_goal_idxs - idxs

        time_steps = np.arange(self.horizon_length)[None, :]

        # t <= actor_goal_step时有效，否则无效
        data["valid"] = np.where(time_steps <= actor_goal_steps[:, None], 1.0, 0.0)

        return data


@dataclasses.dataclass
class HGCChunkDataset(HGCDataset):  # TODO: 待修正奖励逻辑
    """分层目标条件动作块数据集

    融合HGCD的分层目标采样与动作块序列采样能力。
    仅支持状态空间数据，返回完整的动作序列和分层目标。
    批次数据结构（sample返回）:
    --------------------------------
    observations: np.ndarray
        起始观测状态，形状 [batch_size, obs_dim]
        对应时间步 s_t

    actions: np.ndarray
        动作序列，形状 [batch_size, horizon_length, action_dim]
        对应时间步 [a_t, a_{t+1}, ..., a_{t+H-1}]

    next_observations: np.ndarray
        H步后的观测状态，形状 [batch_size, obs_dim]
        对应时间步 s_{t+H}（钳位到轨迹边界）

    value_goals: np.ndarray
        价值目标状态，形状 [batch_size, goal_dim]
        用于价值函数训练的最终目标

    low_actor_goals: np.ndarray
        低层演员目标状态，形状 [batch_size, goal_dim]
        对应时间步 s_{t+subgoal_steps}

    high_actor_goals: np.ndarray
        高层演员目标状态，形状 [batch_size, goal_dim]
        用于高层策略的远距规划

    high_actor_targets: np.ndarray
        高层预测目标状态，形状 [batch_size, goal_dim]
        用于监督高层策略的预测

    rewards: np.ndarray
        奖励序列，形状 [batch_size, horizon_length]
        gc_negative=True:  [-1,...,-1, 0,0,...,0]（达成前惩罚，达成后中性）
        gc_negative=False: [0,...,0, 1,0,...,0]（仅在达成瞬间奖励）
        达成后的所有时间步奖励为0，避免干扰学习

    masks: np.ndarray
        价值目标达成掩码，形状 [batch_size]
        取值范围 {0.0, 1.0}
        若在horizon内达成value_goal则masks=0，否则为1
        用于屏蔽已达成目标的样本对价值函数的影响

    valid: np.ndarray
        动作时序有效性掩码，形状 [batch_size, horizon_length]
        取值范围 [0.0, 1.0]
        在low_actor_goal达成前（含达成时刻）为1，之后为0
        用于屏蔽无效动作对演员函数的影响

    """

    def __post_init__(self):
        super().__post_init__()
        assert "horizon_length" in self.config, "horizon_length未在配置中指定"
        self.horizon_length = self.config["horizon_length"]

    def sample(
        self,
        batch_size: int,
        idxs: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """采样一个动作块批次，包含完整的分层目标和时序有效性掩码"""

        # 1. 采样起始索引（允许任意位置，后续通过钳位和valid保证有效性）
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        # 2. 组装基础批次数据
        batch = self._assemble_base_batch(idxs)

        # 3. 采样所有目标相关数据和索引
        goal_data = self._sample_all_goals(idxs)
        batch.update(goal_data)

        # 4. 计算奖励序列（基于value_goal达成时间）
        batch["rewards"], batch["masks"] = self._compute_reward_and_mask(
            idxs, batch["value_goal_idxs"]
        )

        # 5. 计算时序有效性掩码（基于low_actor_goal达成时间）
        batch["valid"] = self._compute_valid_sequence(idxs, batch["low_goal_idxs"])

        # 6. 清理内部索引字段
        batch.pop("value_goal_idxs")
        batch.pop("low_goal_idxs")
        batch.pop("high_goal_idxs")
        batch.pop("high_target_idxs")

        return batch

    def _assemble_base_batch(self, idxs: np.ndarray) -> Dict[str, Any]:
        """组装基础批次数据：observations, actions, next_observations"""
        batch = {}

        # 计算轨迹边界索引（向量化）
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]

        # 起始观测（单帧）
        batch["observations"] = self.get_observations(idxs)

        # 动作序列索引 [batch, horizon_length]
        seq_indices = idxs[:, None] + np.arange(self.horizon_length)[None, :]
        seq_indices_clipped = np.minimum(seq_indices, final_state_idxs[:, None])

        # 安全地获取动作序列 [batch, horizon_length, action_dim]
        batch["actions"] = self.dataset["actions"][seq_indices_clipped]

        # horizon_length步后的下一观测（用于价值函数）
        next_idxs = np.minimum(idxs + self.horizon_length, final_state_idxs)

        batch["next_observations"] = self.get_observations(next_idxs)

        return batch

    def _sample_all_goals(self, idxs: np.ndarray) -> Dict[str, Any]:
        """采样所有目标相关数据和索引"""
        data = {}

        # 计算轨迹边界
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]

        # 1. 价值目标（复用HGCD逻辑）
        data["value_goal_idxs"] = self.sample_goals(
            idxs,
            self.config["value_p_curgoal"],
            self.config["value_p_trajgoal"],
            self.config["value_p_randomgoal"],
            self.config["value_geom_sample"],
        )
        data["value_goals"] = self.get_observations(data["value_goal_idxs"])

        # 2. 低层演员目标（固定步长）
        data["low_goal_idxs"] = np.minimum(
            idxs + self.config["subgoal_steps"], final_state_idxs
        )
        data["low_actor_goals"] = self.get_observations(data["low_goal_idxs"])

        # 3. 高层演员目标（复用sample_goals）
        data["high_goal_idxs"] = self.sample_goals(
            idxs,
            self.config["actor_p_curgoal"],
            self.config["actor_p_trajgoal"],
            self.config["actor_p_randomgoal"],
            self.config["actor_geom_sample"],
        )
        data["high_actor_goals"] = self.get_observations(data["high_goal_idxs"])

        # 高层目标预测目标（双重钳位：不超过high_goal，不超出轨迹边界）
        high_target_candidate = np.minimum(
            idxs + self.config["subgoal_steps"], data["high_goal_idxs"]
        )
        data["high_target_idxs"] = np.minimum(high_target_candidate, final_state_idxs)
        data["high_actor_targets"] = self.get_observations(data["high_target_idxs"])

        return data

    def _compute_reward_and_mask(
        self, idxs: np.ndarray, value_goal_idxs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """统一计算奖励序列和掩码"""
        # 核心：检测value_goal是否在horizon范围内命中
        # value_goal_hit[i] = True 当且仅当 value_goal_idxs[i] < idxs[i] + horizon_length
        horizon_limits = idxs + self.horizon_length
        value_goal_hit = value_goal_idxs < horizon_limits

        # 计算目标相对于起始索引的步数
        # 对于未命中的样本，设goal_steps = horizon_length（永不触发奖励转换）
        goal_steps = np.where(
            value_goal_hit, value_goal_idxs - idxs, self.horizon_length
        )  # 关键：设为horizon_length，永远大于时间步矩阵

        # 生成时间步矩阵 (1, horizon_length)
        time_steps = np.arange(self.horizon_length)[None, :]

        # 计算奖励序列
        if self.config.get("gc_negative", True):
            # 模式A（惩罚-直到达成）：[-1, -1, ..., -1, 0, 0, ..., 0]
            # 达成前每一步惩罚，达成后中性
            rewards = np.where(time_steps < goal_steps[:, None], -1.0, 0.0)
        else:
            # 模式B（稀疏-仅在达成瞬间）：[0, 0, ..., 0, 1, 0, ..., 0]
            # 只有达成瞬间奖励，其他时间中性（包括达成后）
            rewards = np.where(time_steps == goal_steps[:, None], 1.0, 0.0)

        # 计算掩码：只要命中，整个序列都mask掉
        masks = 1.0 - value_goal_hit.astype(float)

        return rewards, masks

    def _compute_valid_sequence(
        self, idxs: np.ndarray, low_goal_idxs: np.ndarray
    ) -> np.ndarray:
        """计算时序有效性掩码 [batch, horizon_length]

        逻辑：low_actor_goal达成前及达成时刻动作有效，达成后无效
        """
        # 计算低层目标达成的相对步数
        low_steps = low_goal_idxs - idxs

        # 生成时间步矩阵 [batch, horizon_length]
        time_steps = np.arange(self.horizon_length)[None, :]

        # t <= low_step时有效，否则无效
        valid = np.where(time_steps <= low_steps[:, None], 1.0, 0.0)

        return valid
