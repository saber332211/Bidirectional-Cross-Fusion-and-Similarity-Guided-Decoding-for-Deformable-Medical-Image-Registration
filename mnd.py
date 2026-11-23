import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientModalAgnosticNeighborhoodDescriptor(nn.Module):
    """
    高效版本的模态无关邻域描述符
    使用卷积思想一次性计算所有邻域差值
    核在初始化时创建，适合深度模型训练
    """

    def __init__(self, channels=1):
        """
        Args:
            channels: 输入数据的通道数
        """
        super(EfficientModalAgnosticNeighborhoodDescriptor, self).__init__()
        self.channels = channels

        # 注册为buffer，这样会随着模型一起移动设备，但不参与梯度计算
        self.register_buffer('kernel', self._create_kernels())

    def _create_kernels(self):
        """创建6个方向的差分核"""
        kernels = []

        # 6个方向的核: [上, 下, 左, 右, 前, 后]
        # 每个核的形状: [1, 1, 3, 3, 3] 对应于 [channels_out, channels_in, depth, height, width]

        # 上 (中心在底部): [0, -1, 0]
        kernel_up = torch.zeros((1, 1, 3, 3, 3))
        kernel_up[0, 0, 1, 2, 1] = 1  # 邻居
        kernel_up[0, 0, 1, 1, 1] = -1  # 中心
        kernels.append(kernel_up.repeat(self.channels, 1, 1, 1, 1))

        # 下 (中心在顶部): [0, 1, 0]
        kernel_down = torch.zeros((1, 1, 3, 3, 3))
        kernel_down[0, 0, 1, 0, 1] = 1  # 邻居
        kernel_down[0, 0, 1, 1, 1] = -1  # 中心
        kernels.append(kernel_down.repeat(self.channels, 1, 1, 1, 1))

        # 左 (中心在右侧): [0, 0, -1]
        kernel_left = torch.zeros((1, 1, 3, 3, 3))
        kernel_left[0, 0, 1, 1, 2] = 1  # 邻居
        kernel_left[0, 0, 1, 1, 1] = -1  # 中心
        kernels.append(kernel_left.repeat(self.channels, 1, 1, 1, 1))

        # 右 (中心在左侧): [0, 0, 1]
        kernel_right = torch.zeros((1, 1, 3, 3, 3))
        kernel_right[0, 0, 1, 1, 0] = 1  # 邻居
        kernel_right[0, 0, 1, 1, 1] = -1  # 中心
        kernels.append(kernel_right.repeat(self.channels, 1, 1, 1, 1))

        # 前 (中心在后部): [-1, 0, 0]
        kernel_front = torch.zeros((1, 1, 3, 3, 3))
        kernel_front[0, 0, 2, 1, 1] = 1  # 邻居
        kernel_front[0, 0, 1, 1, 1] = -1  # 中心
        kernels.append(kernel_front.repeat(self.channels, 1, 1, 1, 1))

        # 后 (中心在前部): [1, 0, 0]
        kernel_back = torch.zeros((1, 1, 3, 3, 3))
        kernel_back[0, 0, 0, 1, 1] = 1  # 邻居
        kernel_back[0, 0, 1, 1, 1] = -1  # 中心
        kernels.append(kernel_back.repeat(self.channels, 1, 1, 1, 1))

        # 组合所有核 [6*channels, channels, 3, 3, 3]
        weight = torch.cat(kernels, dim=0)
        return weight

    def forward(self, x):
        """
        Args:
            x: 5D tensor of shape [batch, channels, depth, height, width]

        Returns:
            5D tensor of shape [batch, 6 * channels, depth, height, width]
        """
        # 检查输入通道数是否匹配
        if x.size(1) != self.channels:
            raise ValueError(f"输入通道数 {x.size(1)} 与初始化通道数 {self.channels} 不匹配")

        # 使用3D卷积计算所有差值
        output = F.conv3d(F.pad(x, [1, 1, 1, 1, 1, 1], mode='replicate'),
                          self.kernel,
                          groups=self.channels,
                          padding=0)

        return output

