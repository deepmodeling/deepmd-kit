# SPDX-License-Identifier: LGPL-3.0-or-later
"""
多层感知机(MLP)网络模块

本模块实现了DeePMD-kit中使用的各种神经网络层和网络集合：
1. MLPLayer: 单个MLP层，支持激活函数、残差连接、时间步等
2. MLP: 多层MLP网络
3. EmbeddingNet: 嵌入网络，用于原子类型嵌入
4. FittingNet: 拟合网络，用于将描述符映射为原子性质
5. NetworkCollection: 网络集合，管理多个网络实例

这些网络是DPA3模型中RepFlow层和Fitting层的基础组件。
"""
from typing import (
    ClassVar,
    Optional,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepmd.pt.utils import (
    env,
)

device = env.DEVICE

from deepmd.dpmodel.utils import (
    NativeLayer,
)
from deepmd.dpmodel.utils import NetworkCollection as DPNetworkCollection
from deepmd.dpmodel.utils import (
    make_embedding_network,
    make_fitting_network,
    make_multilayer_network,
)
from deepmd.pt.model.network.init import (
    kaiming_normal_,
    normal_,
    trunc_normal_,
    xavier_uniform_,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
)
from deepmd.pt.utils.utils import (
    ActivationFn,
    get_generator,
    to_numpy_array,
    to_torch_tensor,
)


def empty_t(shape, precision):
    """创建指定形状和精度的空张量"""
    return torch.empty(shape, dtype=precision, device=device)


class Identity(nn.Module):
    """恒等映射层
    
    这是一个简单的恒等映射层，输入什么就输出什么。
    主要用于网络结构中的占位符或跳过连接。
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        xx: torch.Tensor,
    ) -> torch.Tensor:
        """恒等映射操作：直接返回输入"""
        return xx

    def serialize(self) -> dict:
        """序列化层参数"""
        return {
            "@class": "Identity",
            "@version": 1,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "Identity":
        """从序列化数据反序列化层"""
        return Identity()


class MLPLayer(nn.Module):
    """单个多层感知机(MLP)层
    
    这是DeePMD-kit中最基础的神经网络层，实现了：
    1. 线性变换: y = Wx + b
    2. 激活函数: y = activation(y)
    3. 残差连接: y = y + x (当resnet=True时)
    4. 时间步缩放: y = y * idt (当use_timestep=True时)
    
    在DPA3模型中，这些层被用于：
    - RepFlow层中的消息传递网络
    - Fitting层中的能量/力预测网络
    - 类型嵌入网络
    """
    def __init__(
        self,
        num_in,                    # 输入维度
        num_out,                   # 输出维度
        bias: bool = True,         # 是否使用偏置
        use_timestep: bool = False, # 是否使用时间步缩放
        activation_function: Optional[str] = None, # 激活函数名称
        resnet: bool = False,      # 是否使用残差连接
        bavg: float = 0.0,         # 偏置初始化均值
        stddev: float = 1.0,       # 权重初始化标准差
        precision: str = DEFAULT_PRECISION, # 数值精度
        init: str = "default",     # 初始化方法
        seed: Optional[Union[int, list[int]]] = None, # 随机种子
    ) -> None:
        super().__init__()
        
        # 时间步缩放：只有在残差连接建立时才使用
        # 要求输出维度等于输入维度或输入维度的2倍
        self.use_timestep = use_timestep and (
            num_out == num_in or num_out == num_in * 2
        )
        
        # 基本参数
        self.num_in = num_in
        self.num_out = num_out
        self.activate_name = activation_function
        self.activate = ActivationFn(self.activate_name)  # 激活函数对象
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
        
        # 权重矩阵: [num_in, num_out]
        self.matrix = nn.Parameter(data=empty_t((num_in, num_out), self.prec))
        
        # 随机数生成器
        random_generator = get_generator(seed)
        
        # 偏置参数: [num_out]
        if bias:
            self.bias = nn.Parameter(
                data=empty_t([num_out], self.prec),
            )
        else:
            self.bias = None
            
        # 时间步参数: [num_out] (用于ResNet中的时间步缩放)
        if self.use_timestep:
            self.idt = nn.Parameter(data=empty_t([num_out], self.prec)) # 定义一个可学习的参数
        else:
            self.idt = None
            
        self.resnet = resnet
        
        # =============================================================================
        # 参数初始化
        # =============================================================================
        if init == "default":
            # 默认正态分布初始化
            self._default_normal_init(
                bavg=bavg, stddev=stddev, generator=random_generator
            )
        elif init == "trunc_normal":
            # 截断正态分布初始化
            self._trunc_normal_init(1.0, generator=random_generator)
        elif init == "relu":
            # ReLU激活函数的截断正态分布初始化
            self._trunc_normal_init(2.0, generator=random_generator)
        elif init == "glorot":
            # Glorot均匀分布初始化
            self._glorot_uniform_init(generator=random_generator)
        elif init == "gating":
            # 门控机制的零初始化
            self._zero_init(self.use_bias)
        elif init == "kaiming_normal":
            # Kaiming正态分布初始化
            self._normal_init(generator=random_generator)
        elif init == "final":
            # 最终层的零初始化
            self._zero_init(False)
        else:
            raise ValueError(f"Unknown initialization method: {init}")

    def check_type_consistency(self) -> None:
        """检查所有参数的数据类型一致性"""
        precision = self.precision

        def check_var(var) -> None:
            if var is not None:
                # 检查参数的数据类型是否与指定的精度一致
                # 注意：断言 "float64" == "double" 会失败，所以使用PRECISION_DICT
                assert PRECISION_DICT[var.dtype.name] is PRECISION_DICT[precision]

        check_var(self.matrix)
        check_var(self.bias)
        check_var(self.idt)

    def dim_in(self) -> int:
        """返回输入维度"""
        return self.matrix.shape[0]

    def dim_out(self) -> int:
        """返回输出维度"""
        return self.matrix.shape[1]

    def _default_normal_init(
        self,
        bavg: float = 0.0,
        stddev: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """默认正态分布初始化
        
        权重矩阵使用Xavier初始化：std = stddev / sqrt(fan_in + fan_out)
        偏置使用正态分布：mean=bavg, std=stddev
        时间步参数使用小方差正态分布：mean=0.1, std=0.001
        """
        normal_(
            self.matrix.data,
            std=stddev / np.sqrt(self.num_out + self.num_in),
            generator=generator,
        )
        if self.bias is not None:
            normal_(self.bias.data, mean=bavg, std=stddev, generator=generator)
        if self.idt is not None:
            normal_(self.idt.data, mean=0.1, std=0.001, generator=generator)

    def _trunc_normal_init(
        self, scale=1.0, generator: Optional[torch.Generator] = None
    ) -> None:
        """截断正态分布初始化
        
        使用截断正态分布初始化权重矩阵，有助于避免梯度爆炸问题。
        常数来自scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        """
        # 截断正态分布的标准差因子（来自scipy.stats.truncnorm）
        TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
        _, fan_in = self.matrix.shape
        scale = scale / max(1, fan_in)
        std = (scale**0.5) / TRUNCATED_NORMAL_STDDEV_FACTOR
        trunc_normal_(self.matrix, mean=0.0, std=std, generator=generator)

    def _glorot_uniform_init(self, generator: Optional[torch.Generator] = None) -> None:
        """Glorot均匀分布初始化（Xavier均匀分布）"""
        xavier_uniform_(self.matrix, gain=1, generator=generator)

    def _zero_init(self, use_bias=True) -> None:
        """零初始化
        
        权重矩阵初始化为0，偏置初始化为1（如果use_bias=True）
        用于门控机制或最终层
        """
        with torch.no_grad():
            self.matrix.fill_(0.0)
            if use_bias and self.bias is not None:
                with torch.no_grad():
                    self.bias.fill_(1.0)

    def _normal_init(self, generator: Optional[torch.Generator] = None) -> None:
        """Kaiming正态分布初始化
        
        适用于线性激活函数，有助于保持前向传播时的方差
        """
        kaiming_normal_(self.matrix, nonlinearity="linear", generator=generator)

    def forward(
        self,
        xx: torch.Tensor,
    ) -> torch.Tensor:
        """MLP层的前向传播
        
        实现以下计算流程：
        1. 线性变换: y = W^T * x + b
        2. 激活函数: y = activation(y)
        3. 时间步缩放: y = y * idt (如果启用)
        4. 残差连接: y = y + x (如果启用)

        Parameters
        ----------
        xx : torch.Tensor
            输入张量，形状为 [..., num_in]

        Returns
        -------
        yy: torch.Tensor
            输出张量，形状为 [..., num_out]
        """
        # 保存原始精度，用于后续恢复
        ori_prec = xx.dtype
        
        # 精度转换（如果允许）
        if not env.DP_DTYPE_PROMOTION_STRICT:
            xx = xx.to(self.prec)
            
        # 1. 线性变换: y = W^T * x + b
        # 注意：这里使用matrix.t()进行转置，因为PyTorch的linear函数期望权重为[out_features, in_features]
        yy = F.linear(xx, self.matrix.t(), self.bias)
        
        # 2. 激活函数: y = activation(y)
        yy = self.activate(yy)
        
        # 3. 时间步缩放: y = y * idt (用于ResNet中的时间步控制)
        yy = yy * self.idt if self.idt is not None else yy
        
        # 4. 残差连接: y = y + x (ResNet跳过连接)
        if self.resnet:
            if xx.shape[-1] == yy.shape[-1]:
                # 维度匹配：直接相加
                yy = yy + xx
            elif 2 * xx.shape[-1] == yy.shape[-1]:
                # 输出维度是输入维度的2倍：将输入重复后相加
                yy = yy + torch.concat([xx, xx], dim=-1)
            else:
                # 维度不匹配：不进行残差连接
                yy = yy
                
        # 恢复原始精度
        if not env.DP_DTYPE_PROMOTION_STRICT:
            yy = yy.to(ori_prec)
        return yy

    def serialize(self) -> dict:
        """序列化层参数到字典
        
        将MLP层的所有参数（权重、偏置、时间步参数）序列化为字典格式，
        用于模型保存和加载。

        Returns
        -------
        dict
            序列化后的层参数字典
        """
        # 创建NativeLayer对象，包含层的基本信息
        nl = NativeLayer(
            self.matrix.shape[0],      # 输入维度
            self.matrix.shape[1],      # 输出维度
            bias=self.bias is not None, # 是否有偏置
            use_timestep=self.idt is not None, # 是否使用时间步
            activation_function=self.activate_name, # 激活函数名称
            resnet=self.resnet,        # 是否使用残差连接
            precision=self.precision,  # 数值精度
        )
        
        # 将PyTorch张量转换为numpy数组并赋值
        nl.w, nl.b, nl.idt = (
            to_numpy_array(self.matrix),  # 权重矩阵
            to_numpy_array(self.bias),    # 偏置向量
            to_numpy_array(self.idt),     # 时间步参数
        )
        return nl.serialize()

    @classmethod
    def deserialize(cls, data: dict) -> "MLPLayer":
        """从字典反序列化层参数
        
        从序列化的字典中恢复MLP层的所有参数。

        Parameters
        ----------
        data : dict
            包含序列化层参数的字典

        Returns
        -------
        MLPLayer
            恢复的MLP层实例
        """
        # 从字典反序列化NativeLayer
        nl = NativeLayer.deserialize(data)
        
        # 创建MLPLayer实例
        obj = cls(
            nl["matrix"].shape[0],      # 输入维度
            nl["matrix"].shape[1],      # 输出维度
            bias=nl["bias"] is not None, # 是否有偏置
            use_timestep=nl["idt"] is not None, # 是否使用时间步
            activation_function=nl["activation_function"], # 激活函数
            resnet=nl["resnet"],        # 是否使用残差连接
            precision=nl["precision"],  # 数值精度
        )
        
        # 获取精度类型
        prec = PRECISION_DICT[obj.precision]

        def check_load_param(ss):
            """检查并加载参数"""
            return (
                nn.Parameter(data=to_torch_tensor(nl[ss]))
                if nl[ss] is not None
                else None
            )

        # 加载所有参数
        obj.matrix = check_load_param("matrix")  # 权重矩阵
        obj.bias = check_load_param("bias")      # 偏置向量
        obj.idt = check_load_param("idt")        # 时间步参数
        return obj


# =============================================================================
# 多层网络定义
# =============================================================================

# 使用make_multilayer_network创建多层MLP网络
MLP_ = make_multilayer_network(MLPLayer, nn.Module)


class MLP(MLP_):
    """多层感知机(MLP)网络
    
    由多个MLPLayer组成的深度神经网络，用于复杂的非线性映射。
    在DPA3模型中用于：
    - RepFlow层中的消息传递网络
    - Fitting层中的能量/力预测网络
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 将层列表转换为PyTorch的ModuleList，确保参数注册
        self.layers = torch.nn.ModuleList(self.layers)

    forward = MLP_.call  # 使用父类的前向传播方法


# =============================================================================
# 专用网络类型定义
# =============================================================================

# 嵌入网络：用于原子类型嵌入，将原子类型ID映射为高维向量
EmbeddingNet = make_embedding_network(MLP, MLPLayer)

# 拟合网络：用于将描述符映射为原子性质（能量、力等）
FittingNet = make_fitting_network(EmbeddingNet, MLP, MLPLayer)


class NetworkCollection(DPNetworkCollection, nn.Module):
    """网络集合类
    
    PyTorch实现的网络集合，用于管理多个网络实例。
    在DPA3模型中用于：
    - 管理不同原子类型的拟合网络
    - 管理RepFlow层中的多个消息传递网络
    - 支持混合类型和分离类型的网络配置
    """

    # 网络类型映射字典
    NETWORK_TYPE_MAP: ClassVar[dict[str, type]] = {
        "network": MLP,                    # 通用MLP网络
        "embedding_network": EmbeddingNet, # 嵌入网络
        "fitting_network": FittingNet,     # 拟合网络
    }

    def __init__(self, *args, **kwargs) -> None:
        # 初始化两个基类
        DPNetworkCollection.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)
        # 将网络列表转换为PyTorch的ModuleList，确保参数注册
        self.networks = self._networks = torch.nn.ModuleList(self._networks)
