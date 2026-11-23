import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from .mnd import EfficientModalAgnosticNeighborhoodDescriptor


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(*vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps=7):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, kernal_size, stride, padding)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(UpConvBlock, self).__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)

        self.actout = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(alpha)
        )

    def forward(self, x):
        x = self.upconv(x)
        return self.actout(x)


class ResBlock(nn.Module):
    """
    VoxRes module
    """

    def __init__(self, channel, alpha=0.1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
            nn.Conv3d(channel, channel, kernel_size=3, padding=1)
        )
        self.actout = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
        )

    def forward(self, x):
        out = self.block(x) + x
        return self.actout(out)


class UNetworkFuse(nn.Module):
    def __init__(self, dim, enc_nf=(16, 32, 64, 128), bn=None, full_size=True,
                 input_shape=(128, 96, 160), kernel_sizes=None):
        super(UNetworkFuse, self).__init__()
        self.dim = dim  # 决定是2D还是3D
        self.enc_nf = enc_nf
        self.full_size = full_size
        # The last entry is used at full input resolution (96x128x160).
        # A 5^3 descriptor there materializes 125 channels, which is about
        # 938 MiB for one float32 volume before the fusion convolution.  Keep
        # the larger neighborhoods at coarser scales, but use 3^3 at full
        # resolution so a training step fits on a 24-GiB GPU.
        # Older checkpoints used a 5^3 descriptor at the final (full
        # resolution) decoder level.  New training can pass the smaller 3^3
        # final window to reduce peak memory; keeping this configurable lets
        # inference load either checkpoint family exactly.
        self.kernel_sizes = list(kernel_sizes or [3, 3, 5, 3, 3])
        if len(self.kernel_sizes) != 5 or any(size < 1 or size % 2 == 0 for size in self.kernel_sizes):
            raise ValueError("kernel_sizes must contain five positive odd values")
        in_channel = 1
        c = 16
        # in_shape = (160, 192, 160)
        # in_shape= (192,160,192)  #MR_CT
        # in_shape = (160, 192, 224)
        # in_shape = (96, 128, 128) # WHS
        in_shape = (96,128, 160)  # fmost数据集

        if len(input_shape) != 3 or any(size % 8 for size in input_shape):
            raise ValueError("input_shape must be a 3D shape with every dimension divisible by 8")
        in_shape = tuple(input_shape)

        self.encoder_moving = nn.ModuleList()
        self.encoder_moving.append(ConvInsBlock(in_channel, c, 3, 1))
        self.encoder_moving.append(nn.Sequential(
            nn.Conv3d(c, 2 * c, kernel_size=3, stride=2, padding=1),  # 80
            ResBlock(2 * c)
        ))
        self.encoder_moving.append(nn.Sequential(
            nn.Conv3d(2 * c, 4 * c, kernel_size=3, stride=2, padding=1),  # 40
            ResBlock(4 * c)))
        self.encoder_moving.append(nn.Sequential(
            nn.Conv3d(4 * c, 8 * c, kernel_size=3, stride=2, padding=1),  # 20
            ResBlock(8 * c)))

        self.encoder_fixed = nn.ModuleList()
        self.encoder_fixed.append(ConvInsBlock(in_channel, c, 3, 1))
        self.encoder_fixed.append(nn.Sequential(
            nn.Conv3d(c, 2 * c, kernel_size=3, stride=2, padding=1),  # 80
            ResBlock(2 * c)
        ))
        self.encoder_fixed.append(nn.Sequential(
            nn.Conv3d(2 * c, 4 * c, kernel_size=3, stride=2, padding=1),  # 40
            ResBlock(4 * c)))
        self.encoder_fixed.append(nn.Sequential(
            nn.Conv3d(4 * c, 8 * c, kernel_size=3, stride=2, padding=1),  # 20
            ResBlock(8 * c)))

        # 多层级特征交互模块
        self.interaction_blocks = nn.ModuleList()
        self.interaction_blocks.append(FeatureInteractionModule(enc_nf[0]))
        self.interaction_blocks.append(FeatureInteractionModule(enc_nf[1]))
        self.interaction_blocks.append(FeatureInteractionModule(enc_nf[2]))
        self.interaction_blocks.append(FeatureInteractionModule(enc_nf[3]))

        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(ConvInsBlock(enc_nf[-1] * 2, enc_nf[-1], 3))  # 1
        self.dec.append(ConvInsBlock(enc_nf[-2] * 2 + enc_nf[-1], enc_nf[-2], 3))  # 2
        self.dec.append(ConvInsBlock(enc_nf[-3] * 2 + enc_nf[-2], enc_nf[-3], 3))  # 3
        self.dec.append(ConvInsBlock(enc_nf[-4] * 2 + enc_nf[-3], enc_nf[-4], 3))  # 4
        self.dec.append(ConvInsBlock(enc_nf[-4], enc_nf[-4], 3))  # 4-2

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')

        self.flows = nn.ModuleList()
        self.flow4 = nn.Conv3d(enc_nf[3], dim, kernel_size=3, padding=1)
        self.flow4.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow4.weight.shape))
        self.flow4.bias = nn.Parameter(torch.zeros(self.flow4.bias.shape))
        self.flows.append(self.flow4)

        self.flow3 = nn.Conv3d(enc_nf[2], dim, kernel_size=3, padding=1)
        self.flow3.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow3.weight.shape))
        self.flow3.bias = nn.Parameter(torch.zeros(self.flow3.bias.shape))
        self.flows.append(self.flow3)

        self.flow2 = nn.Conv3d(enc_nf[1], dim, kernel_size=3, padding=1)
        self.flow2.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow2.weight.shape))
        self.flow2.bias = nn.Parameter(torch.zeros(self.flow2.bias.shape))
        self.flows.append(self.flow2)

        self.flow1 = nn.Conv3d(enc_nf[0], dim, kernel_size=3, padding=1)
        self.flow1.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow1.weight.shape))
        self.flow1.bias = nn.Parameter(torch.zeros(self.flow1.bias.shape))
        self.flows.append(self.flow1)

        # 解码时特征融合
        self.fusion_blocks = nn.ModuleList()
        self.fusion_blocks.append(ConvInsBlock(enc_nf[-1] * 2 + self.kernel_sizes[0]**3, enc_nf[-1] * 2, 3))
        self.fusion_blocks.append(ConvInsBlock(enc_nf[-2] * 2 + self.kernel_sizes[1]**3, enc_nf[-2] * 2, 3))
        self.fusion_blocks.append(ConvInsBlock(enc_nf[-3] * 2 + self.kernel_sizes[2]**3, enc_nf[-3] * 2, 3))
        self.fusion_blocks.append(ConvInsBlock(enc_nf[-4] * 2 + self.kernel_sizes[4]**3, enc_nf[-4] * 2, 3))

        self.scaling_squared = nn.ModuleList()
        self.warps = nn.ModuleList()

        for i in range(4):
            self.scaling_squared.append(VecInt([s // 2 ** i for s in in_shape]))
            self.warps.append(SpatialTransformer([s // 2 ** i for s in in_shape]))


    def efficient_similarity_conv_3d_channel_memory_efficient(self, F, M, kernel_size=5, padding=2):
        """
        内存优化版本 - 直接先填充，然后安全切片
        """
        B, C, D, H, W = F.shape
        k = kernel_size
        # 直接先填充！
        F_padded = torch.nn.functional.pad(F, (padding, padding, padding, padding, padding, padding))
        output = torch.zeros(B, k * k * k, D, H, W, device=F.device, dtype=F.dtype)

        # 对每个邻域位置单独计算
        idx = 0
        for dz in range(k):
            for dy in range(k):
                for dx in range(k):
                    # 提取对应的邻域切片
                    F_shifted = F_padded[:, :,
                                dz:dz + D,
                                dy:dy + H,
                                dx:dx + W]

                    # 计算该位置的相似度
                    similarity = (F_shifted * M).sum(dim=1)  # (B, D, H, W)
                    output[:, idx] = similarity
                    idx += 1

        # 对每个位置的k³个值进行softmax归一化
        output = output.permute(0, 2, 3, 4, 1)  # (B, D, H, W, k³)
        output = torch.softmax(output, dim=-1)
        output = output.permute(0, 4, 1, 2, 3).contiguous()  # (B, k³, D, H, W)

        return output

    def forward(self, m, f):
        # Get encoder activations
        m_enc = [m]
        f_enc = [f]

        for i, (l_f, l_m) in enumerate(zip(self.encoder_fixed, self.encoder_moving)):
            m_x = l_m(m_enc[-1])
            f_x = l_f(f_enc[-1])
            # 特征交互，生成权重并影响特征
            m_enhanced, f_enhanced = self.interaction_blocks[i](m_x, f_x)
            m_enc.append(m_enhanced)
            f_enc.append(f_enhanced)

        # 解码
        # 计算相似性特征
        sim = self.efficient_similarity_conv_3d_channel_memory_efficient(m_enc[-1], f_enc[-1], self.kernel_sizes[0],
                                                        self.kernel_sizes[0] // 2)

        m_f_dec = torch.cat((m_enc[-1], f_enc[-1], sim), dim=1)
        m_f_dec = self.fusion_blocks[0](m_f_dec)

        m_f_dec = self.dec[0](m_f_dec)
        flow = self.flows[0](m_f_dec)  # 速度场
        flow = self.scaling_squared[-1](flow)  # 速度场积分

        for i in range(1, 4):
            flow = 2 * self.upsample(flow)
            warp_m_dec = self.warps[-(i + 1)](m_enc[-(i + 1)], flow)

            # 计算相似性特征
            sim = self.efficient_similarity_conv_3d_channel_memory_efficient(warp_m_dec, f_enc[-(i + 1)], self.kernel_sizes[i],
                                                            self.kernel_sizes[i] // 2)

            m_f = torch.cat((warp_m_dec, f_enc[-(i + 1)], sim), dim=1)
            m_f = self.fusion_blocks[i](m_f)
            m_f_dec = self.upsample(m_f_dec)
            m_f_dec = torch.cat([m_f_dec, m_f], dim=1)

            m_f_dec = self.dec[i](m_f_dec)

            flow_ = self.flows[i](m_f_dec)

            flow_ = self.scaling_squared[-(i + 1)](flow_)  # 速度场积分
            flow = self.warps[-(i + 1)](flow, flow_) + flow_

        return flow


class FeatureInteractionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.ModalAgnosticNeighborhoodDescriptor = EfficientModalAgnosticNeighborhoodDescriptor(channels)
        # 为两个方向使用不同的处理网络，但都包含归一化
        self.mf_diff_conv = nn.Sequential(
            # 7C → 4C
            nn.Conv3d(channels * 7, channels * 4, 3, padding=1),
            nn.InstanceNorm3d(channels * 4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            # 4C → 2C
            nn.Conv3d(channels * 4, channels * 2, 3, padding=1),
            nn.InstanceNorm3d(channels * 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            # 2C → C
            nn.Conv3d(channels * 2, channels, 3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.fm_diff_conv = nn.Sequential(
            # 7C → 4C
            nn.Conv3d(channels * 7, channels * 4, 3, padding=1),
            nn.InstanceNorm3d(channels * 4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            # 4C → 2C
            nn.Conv3d(channels * 4, channels * 2, 3, padding=1),
            nn.InstanceNorm3d(channels * 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            # 2C → C
            nn.Conv3d(channels * 2, channels, 3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # 为两个特征分别生成不同的权重
        self.moving_weight_net = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.Sigmoid()
        )

        self.fixed_weight_net = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.Sigmoid()
        )

        # 分别的特征适配器
        self.moving_adaptor = ConvInsBlock(channels, channels, 3)
        self.fixed_adaptor = ConvInsBlock(channels, channels, 3)

    def forward(self, moving_feat, fixed_feat):
        m_mnd = self.ModalAgnosticNeighborhoodDescriptor(moving_feat)
        f_mnd = self.ModalAgnosticNeighborhoodDescriptor(fixed_feat)

        # moving -> fixed 方向
        mf_feat_diff = torch.cat((moving_feat, m_mnd), dim=1) - torch.cat((fixed_feat, f_mnd), dim=1)
        mf_diff_processed = self.mf_diff_conv(mf_feat_diff)  # 这里已经包含归一化了

        # fixed -> moving 方向
        fm_feat_diff = torch.cat((fixed_feat, f_mnd), dim=1) - torch.cat((moving_feat, m_mnd), dim=1)
        fm_diff_processed = self.fm_diff_conv(fm_feat_diff)  # 这里已经包含归一化了

        # 分别拼接
        moving_feat_concat = torch.cat([moving_feat, mf_diff_processed], dim=1)
        moving_weights = self.moving_weight_net(moving_feat_concat)

        fixed_feat_concat = torch.cat([fixed_feat, fm_diff_processed], dim=1)
        fixed_weights = self.fixed_weight_net(fixed_feat_concat)

        # 用不同的权重调整特征
        moving_enhanced = moving_feat * (1 + moving_weights)
        fixed_enhanced = fixed_feat * (1 + fixed_weights)

        # 分别进行特征适配
        moving_enhanced = self.moving_adaptor(moving_enhanced)
        fixed_enhanced = self.fixed_adaptor(fixed_enhanced)

        return moving_enhanced, fixed_enhanced
