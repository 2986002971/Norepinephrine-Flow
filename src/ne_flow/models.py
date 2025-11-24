import functools
from typing import Any, List, Optional, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x


class ResnetStack(nn.Module):
    """ResNet stack module."""

    num_features: int
    num_blocks: int
    max_pooling: bool = True

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding="SAME",
        )(x)

        if self.max_pooling:
            conv_out = nn.max_pool(
                conv_out,
                window_shape=(3, 3),
                padding="SAME",
                strides=(2, 2),
            )

        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding="SAME",
                kernel_init=initializer,
            )(conv_out)

            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding="SAME",
                kernel_init=initializer,
            )(conv_out)
            conv_out += block_input

        return conv_out


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = False

    def setup(self):
        stack_sizes = self.stack_sizes
        self.stack_blocks = [
            ResnetStack(
                num_features=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
            )
            for i in range(len(stack_sizes))
        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        x = x.astype(jnp.float32) / 255.0

        conv_out = x

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)

        conv_out = nn.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm()(conv_out)
        out = conv_out.reshape((*x.shape[:-3], -1))

        out = MLP(
            self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm
        )(out)

        return out


class GCEncoder(nn.Module):
    """Helper module to handle inputs to goal-conditioned networks.

    It takes in observations (s) and goals (g) and returns the concatenation of `state_encoder(s)`, `goal_encoder(g)`,
    and `concat_encoder([s, g])`. It ignores the encoders that are not provided. This way, the module can handle both
    early and late fusion (or their variants) of state and goal information.
    """

    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None
    concat_encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations, goals=None, goal_encoded=False):
        """Returns the representations of observations and goals.

        If `goal_encoded` is True, `goals` is assumed to be already encoded representations. In this case, either
        `goal_encoder` or `concat_encoder` must be None.
        """
        reps = []
        if self.state_encoder is not None:
            reps.append(self.state_encoder(observations))
        if goals is not None:
            if goal_encoded:
                # Can't have both goal_encoder and concat_encoder in this case.
                assert self.goal_encoder is None or self.concat_encoder is None
                reps.append(goals)
            else:
                if self.goal_encoder is not None:
                    reps.append(self.goal_encoder(goals))
                if self.concat_encoder is not None:
                    reps.append(
                        self.concat_encoder(
                            jnp.concatenate([observations, goals], axis=-1)
                        )
                    )
        reps = jnp.concatenate(reps, axis=-1)
        return reps


encoder_modules = {
    "impala": ImpalaEncoder,
    "impala_debug": functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    "impala_small": functools.partial(ImpalaEncoder, num_blocks=1),
    "impala_large": functools.partial(
        ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)
    ),
}


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class FourierFeatures(nn.Module):
    # used for timestep embedding
    output_size: int = 64
    learnable: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param(
                "kernel",
                nn.initializers.normal(0.2),
                (self.output_size // 2, x.shape[-1]),
                jnp.float32,
            )
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class LengthNormalize(nn.Module):
    """Length normalization layer.

    It normalizes the input along the last dimension to have a length of sqrt(dim).
    """

    @nn.compact
    def __call__(self, x):
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True) * jnp.sqrt(x.shape[-1])


class Param(nn.Module):
    """Scalar parameter module."""

    init_value: float = 0.0

    @nn.compact
    def __call__(self):
        return self.param("value", init_fn=lambda key: jnp.full((), self.init_value))


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param(
            "log_value", init_fn=lambda key: jnp.full((), jnp.log(self.init_value))
        )
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class RunningMeanStd(flax.struct.PyTreeNode):
    """Running mean and standard deviation.

    Attributes:
        eps: Epsilon value to avoid division by zero.
        mean: Running mean.
        var: Running variance.
        clip_max: Clip value after normalization.
        count: Number of samples.
    """

    eps: Any = 1e-6
    mean: Any = 1.0
    var: Any = 1.0
    clip_max: Any = 10.0
    count: int = 0

    def normalize(self, batch):
        batch = (batch - self.mean) / jnp.sqrt(self.var + self.eps)
        batch = jnp.clip(batch, -self.clip_max, self.clip_max)
        return batch

    def unnormalize(self, batch):
        return batch * jnp.sqrt(self.var + self.eps) + self.mean

    def update(self, batch):
        batch_mean, batch_var = jnp.mean(batch, axis=0), jnp.var(batch, axis=0)
        batch_count = len(batch)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        return self.replace(mean=new_mean, var=new_var, count=total_count)


class GCActor(nn.Module):
    """Goal-conditioned actor.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True)
        self.mean_net = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(
                self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
            )
        else:
            if not self.const_std:
                self.log_stds = self.param(
                    "log_stds", nn.initializers.zeros, (self.action_dim,)
                )

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Scaling factor for the standard deviation.
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        if self.tanh_squash:
            distribution = TransformedWithMode(
                distribution, distrax.Block(distrax.Tanh(), ndims=1)
            )

        return distribution


class GCDiscreteActor(nn.Module):
    """Goal-conditioned actor for discrete actions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True)
        self.logit_net = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Inverse scaling factor for the logits (set to 0 to get the argmax).
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        logits = self.logit_net(outputs)

        distribution = distrax.Categorical(
            logits=logits / jnp.maximum(1e-6, temperature)
        )

        return distribution


class GCValue(nn.Module):
    """Goal-conditioned value/critic function.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    ensemble: bool = True
    gc_encoder: nn.Module = None

    def setup(self):
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)
        value_net = mlp_module(
            (*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm
        )

        self.value_net = value_net

    def __call__(self, observations, goals=None, actions=None):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals (optional).
            actions: Actions (optional).
        """
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v


class SymetricStateEncoder(nn.Module):
    """
    专门处理同构状态空间 (s, g) 的编码器。
    利用了 Siamese 结构和几何差分特征。
    """

    hidden_dim: int = 256
    activations: Any = nn.gelu
    layer_norm: bool = True

    @nn.compact
    def __call__(self, obs, goal):
        # 1. 显式几何特征 (Explicit Physics)
        # 在原始空间计算差分，这对低层控制非常重要
        raw_diff = goal - obs

        # 2. 孪生投影 (Siamese Projection)
        # 定义一个共享的 Encoder
        encoder = nn.Dense(
            self.hidden_dim, kernel_init=default_init(), name="shared_encoder"
        )

        # 分别编码
        h_s = encoder(obs)
        h_g = encoder(goal)

        # 3. 特征交互 (Feature Interaction)
        # 在潜在空间计算特征差
        h_diff = h_g - h_s
        # 也可以加上点积来捕捉相似度
        h_prod = h_s * h_g

        # 4. 融合 (Fusion)
        # 我们希望网络同时知道："我在哪" (h_s) 和 "我要去哪" (raw_diff, h_diff)
        # 注意：通常不需要把 h_g 再次拼进去，因为 h_s + h_diff = h_g，信息是冗余的
        # 但为了保留绝对目标信息，拼上也无妨

        features = jnp.concatenate(
            [
                h_s,  # 当前状态的潜在特征
                h_diff,  # 潜在空间的相对距离
                h_prod,  # 潜在空间的匹配度
                raw_diff,  # 物理空间的真实距离 (Shortcut)
            ],
            axis=-1,
        )

        # 投影回标准维度
        out = nn.Dense(self.hidden_dim, kernel_init=default_init())(features)
        out = self.activations(out)

        if self.layer_norm:
            out = nn.LayerNorm()(out)

        return out


class GeometricValueNetwork(nn.Module):
    """
    核心 V 网络逻辑：
    (s, g) -> SymetricEncoder -> MLP Head -> V
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    activations: Any = nn.gelu

    @nn.compact
    def __call__(self, observations, goals, actions=None):
        # 注意：V 网络忽略 actions，即使不小心传进来也无所谓

        # 1. 几何特征提取
        # 使用 hidden_dims[0] 作为 embedding 维度
        embedding = SymetricStateEncoder(
            hidden_dim=self.hidden_dims[0],
            layer_norm=self.layer_norm,
            activations=self.activations,
        )(observations, goals)

        x = embedding

        # 2. MLP Head
        # 从第二个 hidden_dim 开始构建后续层
        # 如果 hidden_dims 只有一个元素，这就变成直接输出
        v = MLP(
            (*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm
        )(x)

        # 3. 输出 V 值
        return v.squeeze(-1)


class GCGeometricValue(nn.Module):
    """
    对外接口类：几何感知的 V 函数。
    替代 GCValue。
    """

    hidden_dims: Sequence[int] = (256, 256)
    layer_norm: bool = True
    ensemble: bool = False  # V 通常不 ensemble，但保持接口灵活
    gc_encoder: nn.Module = None  # 兼容性保留

    def setup(self):
        config = {
            "hidden_dims": self.hidden_dims,
            "layer_norm": self.layer_norm,
        }

        if self.ensemble:
            GVnetwork = ensemblize(GeometricValueNetwork, 2, **config)
            self.net = GVnetwork(**config)
        else:
            self.net = GeometricValueNetwork(**config)

    def __call__(self, observations, goals, actions=None):
        return self.net(observations, goals)


class FiLM1DBlock(nn.Module):
    """
    基础组件：受Context调制的1D卷积块
    Conv1D -> FiLM -> Activation -> LayerNorm
    """

    features: int
    kernel_size: int = 3
    activations: Any = nn.gelu
    layer_norm: bool = True  # 默认开启，对 IQL 稳定性很重要

    @nn.compact
    def __call__(self, x, context):
        # x: [Batch, Time, Channels] (Action stream)
        # context: [Batch, Embed_Dim] (State stream)

        # 1. 提取时序特征
        x = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            padding="SAME",
            kernel_init=default_init(),
        )(x)

        # 2. 生成 FiLM 参数 (Gamma, Beta)
        # 将 context 映射到 2 * features
        stats = nn.Dense(self.features * 2, kernel_init=default_init())(context)
        gamma, beta = jnp.split(stats, 2, axis=-1)

        # 扩展维度以匹配时间轴: [B, F] -> [B, 1, F]
        gamma = jnp.expand_dims(gamma, axis=1)
        beta = jnp.expand_dims(beta, axis=1)

        # 3. 执行调制 (Affine)
        x = (1.0 + gamma) * x + beta

        # 4. 激活与归一化
        x = self.activations(x)
        if self.layer_norm:
            x = nn.LayerNorm()(x)

        return x


class ChunkCriticNetwork(nn.Module):
    """
    核心网络逻辑：
    Context(s, g) --> MLP --+--> FiLM调制
                            |
    Actions(a_chunk) -----> ConvStack --> Flatten --> Concat --> MLP Head -> Q
    """

    hidden_dims: Sequence[int]  # MLP Head 的层宽
    conv_dims: Sequence[int]  # 卷积层的通道数配置
    layer_norm: bool = True
    activations: Any = nn.gelu

    @nn.compact
    def __call__(self, observations, goals, actions):
        # 1. 构建 Context (Obs + Goal)
        context = SymetricStateEncoder(
            hidden_dim=self.hidden_dims[0],
            layer_norm=self.layer_norm,
            activations=self.activations,
        )(observations, goals)

        # 2. 处理动作块 (Action Chunk Stream)
        # 确保 actions 是 3D 的 [Batch, Horizon, Act_Dim]
        x = actions

        # 堆叠 FiLM 卷积块
        for features in self.conv_dims:
            x = FiLM1DBlock(
                features=features,
                layer_norm=self.layer_norm,
                activations=self.activations,
            )(x, context)

        # 3. 融合
        # 展平时间维: [B, H, C] -> [B, H*C]
        x = x.reshape((x.shape[0], -1))

        # 再次拼接 Context (Skip Connection)，增强梯度流动
        x = jnp.concatenate([x, context], axis=-1)

        # 4. MLP Head 输出 Q 值
        x = MLP(
            self.hidden_dims,
            activate_final=True,
            layer_norm=self.layer_norm,
            activations=self.activations,
        )(x)

        q = nn.Dense(1, kernel_init=default_init())(x)
        return q.squeeze(-1)


class GCChunkCritic(nn.Module):
    """
    对外接口类：专门用于 Action Chunking 的 Critic。
    替代 GCValue 用于 Critic 部分。
    """

    hidden_dims: Sequence[int] = (256, 256)
    conv_dims: Sequence[int] = (64, 128, 256)  # 推荐配置
    layer_norm: bool = True
    ensemble: bool = True
    # gc_encoder 保留但不使用，为了维持 config 结构的兼容性
    gc_encoder: nn.Module = None

    def setup(self):
        # 配置内部网络
        config = {
            "hidden_dims": self.hidden_dims,
            "conv_dims": self.conv_dims,
            "layer_norm": self.layer_norm,
        }

        if self.ensemble:
            ChunkCriticClass = ensemblize(ChunkCriticNetwork, 2)
            self.net = ChunkCriticClass(**config)
        else:
            self.net = ChunkCriticNetwork(**config)

    def __call__(self, observations, goals, actions):
        """
        显式要求传入 actions
        参数:
            observations: [B, Obs_Dim]
            goals: [B, Goal_Dim]
            actions: [B, Horizon, Act_Dim] (必须是分块后的形状)
        """
        return self.net(observations, goals, actions)


class GCDiscreteCritic(GCValue):
    """Goal-conditioned critic for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, goals=None, actions=None):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions)


class GCBilinearValue(nn.Module):
    """Goal-conditioned bilinear value/critic function.

    This module computes the value function as V(s, g) = phi(s)^T psi(g) / sqrt(d) or the critic function as
    Q(s, a, g) = phi(s, a)^T psi(g) / sqrt(d), where phi and psi output d-dimensional vectors.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        value_exp: Whether to exponentiate the value. Useful for contrastive learning.
        state_encoder: Optional state encoder.
        goal_encoder: Optional goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None

    def setup(self):
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)

        self.phi = mlp_module(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )
        self.psi = mlp_module(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    def __call__(self, observations, goals, actions=None, info=False):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals.
            actions: Actions (optional).
            info: Whether to additionally return the representations phi and psi.
        """
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)

        if actions is None:
            phi_inputs = observations
        else:
            phi_inputs = jnp.concatenate([observations, actions], axis=-1)

        phi = self.phi(phi_inputs)
        psi = self.psi(goals)

        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)

        if self.value_exp:
            v = jnp.exp(v)

        if info:
            return v, phi, psi
        else:
            return v


class GCDiscreteBilinearCritic(GCBilinearValue):
    """Goal-conditioned bilinear critic for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, goals=None, actions=None, info=False):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions, info)


class GCMRNValue(nn.Module):
    """Metric residual network (MRN) value function.

    This module computes the value function as the sum of a symmetric Euclidean distance and an asymmetric
    L^infinity-based quasimetric.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional state/goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    encoder: nn.Module = None

    def setup(self):
        self.phi = MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    def __call__(self, observations, goals, is_phi=False, info=False):
        """Return the MRN value function.

        Args:
            observations: Observations.
            goals: Goals.
            is_phi: Whether the inputs are already encoded by phi.
            info: Whether to additionally return the representations phi_s and phi_g.
        """
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        sym_s = phi_s[..., : self.latent_dim // 2]
        sym_g = phi_g[..., : self.latent_dim // 2]
        asym_s = phi_s[..., self.latent_dim // 2 :]
        asym_g = phi_g[..., self.latent_dim // 2 :]
        squared_dist = ((sym_s - sym_g) ** 2).sum(axis=-1)
        quasi = jax.nn.relu((asym_s - asym_g).max(axis=-1))
        v = jnp.sqrt(jnp.maximum(squared_dist, 1e-12)) + quasi

        if info:
            return v, phi_s, phi_g
        else:
            return v


class GCIQEValue(nn.Module):
    """Interval quasimetric embedding (IQE) value function.

    This module computes the value function as an IQE-based quasimetric.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        dim_per_component: Dimension of each component in IQE (i.e., number of intervals in each group).
        layer_norm: Whether to apply layer normalization.
        encoder: Optional state/goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    dim_per_component: int
    layer_norm: bool = True
    encoder: nn.Module = None

    def setup(self):
        self.phi = MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )
        self.alpha = Param()

    def __call__(self, observations, goals, is_phi=False, info=False):
        """Return the IQE value function.

        Args:
            observations: Observations.
            goals: Goals.
            is_phi: Whether the inputs are already encoded by phi.
            info: Whether to additionally return the representations phi_s and phi_g.
        """
        alpha = jax.nn.sigmoid(self.alpha())
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        x = jnp.reshape(phi_s, (*phi_s.shape[:-1], -1, self.dim_per_component))
        y = jnp.reshape(phi_g, (*phi_g.shape[:-1], -1, self.dim_per_component))
        valid = x < y
        xy = jnp.concatenate(jnp.broadcast_arrays(x, y), axis=-1)
        ixy = xy.argsort(axis=-1)
        sxy = jnp.take_along_axis(xy, ixy, axis=-1)
        neg_inc_copies = jnp.take_along_axis(
            valid, ixy % self.dim_per_component, axis=-1
        ) * jnp.where(ixy < self.dim_per_component, -1, 1)
        neg_inp_copies = jnp.cumsum(neg_inc_copies, axis=-1)
        neg_f = -1.0 * (neg_inp_copies < 0)
        neg_incf = jnp.concatenate(
            [neg_f[..., :1], neg_f[..., 1:] - neg_f[..., :-1]], axis=-1
        )
        components = (sxy * neg_incf).sum(axis=-1)
        v = alpha * components.mean(axis=-1) + (1 - alpha) * components.max(axis=-1)

        if info:
            return v, phi_s, phi_g
        else:
            return v


class GCFlowActor(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        gc_encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    gc_encoder: nn.Module = None
    use_fourier_features: bool = False
    fourier_feature_dim: int = 64

    def setup(self) -> None:
        self.mlp = MLP(
            (*self.hidden_dims, self.action_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )
        if self.use_fourier_features:
            self.ff = FourierFeatures(self.fourier_feature_dim)

    @nn.compact
    def __call__(self, observations, goals, actions, times, is_encoded=False):
        """Return the vectors at the given states, goals, actions, and times (optional).

        Args:
            observations: Observations.
            goals: Goals.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the goal are already encoded.
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=is_encoded)
        else:
            inputs = [observations, goals]
        inputs.append(actions)

        if times is None:
            inputs = jnp.concatenate(inputs, axis=-1)
        else:
            if self.use_fourier_features:
                times = self.ff(times)
            inputs.append(times)
            inputs = jnp.concatenate(inputs, axis=-1)

        v = self.mlp(inputs)

        return v


class ContinuousPositionEmbedding(nn.Module):
    """Sinus positional embedding adapted for continuous signal with given range"""

    size_emb: int
    period_min: float
    period_max: float

    def setup(self):
        size_half = self.size_emb // 2
        tensor_period_ratio = jnp.linspace(0.0, 1.0, size_half)
        self.periods = self.period_min * jnp.power(
            self.period_max / self.period_min, tensor_period_ratio
        )

    def __call__(self, tensor_time: jnp.ndarray) -> jnp.ndarray:
        """Forward.
        Args:
            tensor_time: Input time (size_batch, 1).
        Returns:
            Generated embedding (size_batch, size_emb).
        """
        size_batch = tensor_time.shape[0]
        tensor_phase = tensor_time / self.periods

        tensor_value = jnp.stack(
            [
                jnp.sin(2.0 * jnp.pi * tensor_phase),
                jnp.cos(2.0 * jnp.pi * tensor_phase),
            ],
            axis=-1,
        )

        tensor_value = tensor_value.reshape(size_batch, self.size_emb)
        return tensor_value


class BlockConv1d(nn.Module):
    """1d convolution keeping same temporal length with non-linearity and normalization"""

    size_channel_in: int
    size_channel_out: int
    size_kernel: int
    size_group_norm: int

    @nn.compact
    def __call__(self, tensor_in: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(
            features=self.size_channel_out,
            kernel_size=(self.size_kernel,),
            strides=(1,),
            padding="SAME",
        )(tensor_in)
        x = nn.GroupNorm(num_groups=self.size_group_norm)(x)
        x = nn.silu(x)
        return x


class BlockDownsample(nn.Module):
    """Downscale the sequence by 2"""

    size_channel: int

    @nn.compact
    def __call__(self, tensor_in: jnp.ndarray) -> jnp.ndarray:
        return nn.Conv(
            features=self.size_channel,
            kernel_size=(2,),
            strides=(2,),
            padding="VALID",
        )(tensor_in)


class BlockUpsample(nn.Module):
    """Upscale the sequence by 2"""

    size_channel: int

    @nn.compact
    def __call__(self, tensor_in: jnp.ndarray) -> jnp.ndarray:
        return nn.ConvTranspose(
            features=self.size_channel,
            kernel_size=(2,),
            strides=(2,),
            padding="VALID",
        )(tensor_in)


class BlockConv1dResidualConditional(nn.Module):
    """Convolutional block with residual connection and conditioning using FiLM"""

    size_channel_in: int
    size_channel_out: int
    size_cond: int
    size_kernel: int
    size_group_norm: int

    @nn.compact
    def __call__(self, tensor_in: jnp.ndarray, tensor_cond: jnp.ndarray) -> jnp.ndarray:
        cond_film = nn.Dense(features=2 * self.size_channel_out)(tensor_cond)
        cond_scale = cond_film[..., : self.size_channel_out]
        cond_bias = cond_film[..., self.size_channel_out :]

        # Expend dim for broadcasting
        cond_scale = jnp.expand_dims(cond_scale, axis=-2)
        cond_bias = jnp.expand_dims(cond_bias, axis=-2)

        x = BlockConv1d(
            size_channel_in=self.size_channel_in,
            size_channel_out=self.size_channel_out,
            size_kernel=self.size_kernel,
            size_group_norm=self.size_group_norm,
        )(tensor_in)

        x = cond_scale * x + cond_bias

        x = BlockConv1d(
            size_channel_in=self.size_channel_out,
            size_channel_out=self.size_channel_out,
            size_kernel=self.size_kernel,
            size_group_norm=self.size_group_norm,
        )(x)

        residual = nn.Conv(features=self.size_channel_out, kernel_size=(1,))(tensor_in)
        return x + residual


class GCUnet(nn.Module):
    """Implement a 1d convolutional Unet architecture with
    residual block, group normalization and conditional vector.
    Uses position sinusoidal embedding.
    """

    size_channel: int
    size_emb_transport: int
    size_channel_hidden: List[int]
    period_min: float
    period_max: float
    size_kernel: int
    size_group_norm: int

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,  # shape: (batch_size, obs_dim)
        goals: jnp.ndarray,  # shape: (batch_size, goal_dim)
        actions: jnp.ndarray,  # shape: (batch_size, seq_len, action_dim)
        times: jnp.ndarray,  # shape: (batch_size, 1)
    ) -> jnp.ndarray:
        """Forward pass of GCUnet.

        Args:
            observations: 当前观测，形状 (batch_size, obs_dim)
            goals: 目标状态，形状 (batch_size, goal_dim)
            actions: 动作序列/轨迹，形状 (batch_size, seq_len, action_dim)
            times: 时间步，形状 (batch_size, 1)，用于扩散/流模型

        Returns:
            输出张量，形状 (batch_size, seq_len, size_channel)
        """
        # 将 observations 和 goals 拼接成条件向量
        tensor_cond = jnp.concatenate([observations, goals], axis=-1)

        # 时间嵌入
        transport_embedded = ContinuousPositionEmbedding(
            size_emb=self.size_emb_transport,
            period_min=self.period_min,
            period_max=self.period_max,
        )(times)

        transport_embedded = nn.Dense(features=self.size_emb_transport * 4)(
            transport_embedded
        )
        transport_embedded = nn.silu(transport_embedded)
        transport_embedded = nn.Dense(features=self.size_emb_transport)(
            transport_embedded
        )
        transport_embedded = nn.silu(transport_embedded)

        # 合并条件
        cond_all = jnp.concatenate([transport_embedded, tensor_cond], axis=-1)

        # 初始输入是 actions
        x = actions

        list_residuals = []

        # Downsample path
        list_size_channel = [self.size_channel] + list(self.size_channel_hidden)
        for i in range(len(list_size_channel) - 1):
            size_in = list_size_channel[i]
            size_out = list_size_channel[i + 1]

            x = BlockConv1dResidualConditional(
                size_channel_in=size_in,
                size_channel_out=size_out,
                size_cond=cond_all.shape[-1],
                size_kernel=self.size_kernel,
                size_group_norm=self.size_group_norm,
            )(x, cond_all)

            list_residuals.append(x)
            x = BlockDownsample(size_channel=size_out)(x)

        # Middle path
        size_last = list_size_channel[-1]
        x = BlockConv1dResidualConditional(
            size_channel_in=size_last,
            size_channel_out=size_last,
            size_cond=cond_all.shape[-1],
            size_kernel=self.size_kernel,
            size_group_norm=self.size_group_norm,
        )(x, cond_all)

        # Upsample path
        for i in range(len(list_size_channel) - 1, 0, -1):
            size_in = list_size_channel[i]
            if i == 1:
                size_out = list(self.size_channel_hidden)[0]
            else:
                size_out = list_size_channel[i - 1]

            x = BlockUpsample(size_channel=size_in)(x)
            x = jnp.concatenate([x, list_residuals.pop()], axis=-1)

            x = BlockConv1dResidualConditional(
                size_channel_in=x.shape[-1],  # Adjusted for concatenation
                size_channel_out=size_out,
                size_cond=cond_all.shape[-1],
                size_kernel=self.size_kernel,
                size_group_norm=self.size_group_norm,
            )(x, cond_all)

        # Final convolution
        return nn.Conv(features=self.size_channel, kernel_size=(1,))(x)
