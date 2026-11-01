import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.distributions.normal import Normal

class res_encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.bias = bias
        
        super(res_encoder, self).__init__()
        
        self.conv = self.encoder(self.in_channels, self.out_channels, kernel_size = 3, stride=1, padding=1, bias=self.bias)
        
    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size =1, stride=1, padding=0, bias=bias),
            nn.PReLU(num_parameters=out_channels//2),
            nn.Conv2d(out_channels//2, out_channels//2, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.PReLU(num_parameters=out_channels//2),
            nn.Conv2d(out_channels//2, out_channels, kernel_size = 1, stride=1, padding=0, bias=bias),
            nn.PReLU(num_parameters=out_channels))
        return layer
    def forward(self, inputs):
        outputs = self.conv(inputs)
        return outputs + inputs

class skip_encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.bias = bias
        
        super(skip_encoder, self).__init__()
        
        self.conv = self.encoder(self.in_channels, self.out_channels, kernel_size = 3, stride=1, padding=1, bias=self.bias)
        
    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size =1, stride=1, padding=0, bias=bias),
            nn.PReLU(num_parameters=out_channels//2),
            nn.Conv2d(out_channels//2, out_channels//2, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.PReLU(num_parameters=out_channels//2),
            nn.Conv2d(out_channels//2, out_channels, kernel_size = 1, stride=1, padding=0, bias=bias),
            nn.PReLU(num_parameters=out_channels))
        return layer
    def forward(self, inputs, inputs2):
        outputs = self.conv(inputs)
        return outputs + inputs2

class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Sobelfilter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

        Gx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        Gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.Sobelfilter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.Sobelfilter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x

class AttentionFusion(nn.Module): 
    def __init__(self, in_channels=64, squeeze_ratio=16):
        super(AttentionFusion, self).__init__()
        hidden_dim = int(in_channels / squeeze_ratio)
        self.conv = self.fuse_conv_encoder(in_channels= in_channels, out_channels = hidden_dim)
        # self.conv2 = self.conv_encoder(in_channels= in_channels, out_channels = in_channels //2)
    
    def fuse_conv_encoder(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride=1, padding=1, bias=True, groups = in_channels)
            )
        return layer
    # def conv_encoder(self, in_channels, out_channels):
        # layer = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.PReLU()
            # )
        return layer
    def forward(self, sil_feat, map_feat): 
        '''
            sil_feat: [n, c, h, w]
            map_feat: [n, c, h, w]
        '''
        c = sil_feat.size(1)
        feats = torch.cat([sil_feat, map_feat], dim=1)
        score = self.conv(feats) # [n, 2 * c, h, w]
        score = F.sigmoid(score)
        # score = rearrange(score, 'n (d c) h w -> n d c h w', d=2)
        # retun = torch.cat([sil_feat * score[:, 0] , map_feat * score[:, 1]],dim=1)
        # retun = self.conv2(retun)
        return feats * score


class Gating_Selection(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.kernel_size = kernel_size
        # self.padding = padding
        # self.stride = stride
        # self.bias = bias
        # self.hidden_dim = int(in_channels / squeeze_ratio)
        
        super(Gating_Selection, self).__init__()
        
        self.conv1 = self.fuse_conv_encoder(self.in_channels, self.out_channels)
        # self.conv2 = self.fuse_conv_encoder(self.out_channels, self.out_channels, kernel_size = 3, stride=1, padding=1, bias=self.bias, hidden_dim = self.hidden_dim)
    def fuse_conv_encoder(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride=1, padding=1, bias=True, groups = in_channels)
            )
        return layer
    def forward(self, feature):
        lowd_feature = self.conv1(feature)
        score = F.sigmoid(lowd_feature)
        # lowd_feature = self.conv2(lowd_feature)
        # highd_image = self.conv2(image)
        # gate = lowd_feature * highd_image
        # score = F.softmax(gate, dim=1)
        # fuse = score * lowd_feature
        return score

class Gating_Intensity_Structure(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.kernel_size = kernel_size
        # self.padding = padding
        # self.stride = stride
        # self.bias = False
        # self.hidden_dim = int(in_channels / squeeze_ratio)
        
        super(Gating_Intensity_Structure, self).__init__()
        
        self.conv1 = self.fuse_conv_encoder(self.in_channels + 1, self.out_channels)
        # self.conv2 = self.fuse_conv_encoder(self.out_channels, self.out_channels, kernel_size = 3, stride=1, padding=1, bias=self.bias, hidden_dim = self.hidden_dim)
    def fuse_conv_encoder(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //2, kernel_size = 1, stride=1, padding=0, bias=True),#, groups = in_channels
            nn.PReLU(num_parameters=out_channels //2),
            nn.Conv2d(out_channels //2, out_channels //2, kernel_size = 3, stride=1, padding=1, bias=True),
            nn.PReLU(num_parameters=out_channels //2),
            nn.Conv2d(out_channels //2, out_channels, kernel_size = 1, stride=1, padding=0, bias=True),
            nn.PReLU(num_parameters=out_channels))
        return layer
    def forward(self, feature, image):
        lowd_feature = self.conv1(torch.cat([feature, image], dim=1))
        # lowd_feature = self.conv2(lowd_feature)
        # highd_image = self.conv2(image)
        # gate = lowd_feature * highd_image
        # score = F.softmax(gate, dim=1)
        # fuse = score * lowd_feature
        return lowd_feature
'''
class SelectionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.bias = bias
        
        super(SelectionBlock, self).__init__()
        
        # self.group_conv = self.group_conv_encoder(self.in_channels, self.in_channels, kernel_size = self.kernel_size, stride=self.stride, padding=self.padding , bias=self.bias)
        self.fuse_conv1 = self.fuse_conv_encoder(self.in_channels, self.out_channels, kernel_size = self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias)
        self.fuse_conv2 = self.fuse_conv_encoder(self.out_channels, self.out_channels, kernel_size = self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias)
        self.fuse_conv3 = self.fuse_conv_encoder(self.out_channels, self.out_channels, kernel_size = self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias)
        
    # def group_conv_encoder(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        # layer = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups = in_channels),
            # nn.PReLU(num_parameters=out_channels))
        # return layer
    def fuse_conv_encoder(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.PReLU())
        return layer
    def forward(self, inputs):
        # group = self.group_conv(inputs)
        fuse = self.fuse_conv1(inputs)
        fuse = self.fuse_conv2(fuse)
        fuse = self.fuse_conv3(fuse)
        return fuse
'''

class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()
    def forward(self, mov_image, flow, mod = 'bilinear'):
        # d2, h2, w2 = mov_image.shape[-3:]
        h2, w2 = mov_image.shape[-2:]
        # grid_d, grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.to(flow.device).float()
        # grid_d = grid_d.to(flow.device).float()
        grid_w = grid_w.to(flow.device).float()
        # grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow_h = flow[:,:,:,0]
        flow_w = flow[:,:,:,1]
        # flow_w = flow[:,:,:,2]
        #Softsign
        #disp_d = (grid_d + (flow_d * 2 / d2)).squeeze(1)
        #disp_h = (grid_h + (flow_h * 2 / h2)).squeeze(1)
        #disp_w = (grid_w + (flow_w * 2 / w2)).squeeze(1)
        
        # Remove Channel Dimension
        # disp_d = (grid_d + (flow_d)).squeeze(1)
        disp_h = (grid_h + (flow_h)).squeeze(1)
        disp_w = (grid_w + (flow_w)).squeeze(1)
        sample_grid = torch.stack((disp_w, disp_h), 3)  # shape (N, D, H, W, 3)
        warped = torch.nn.functional.grid_sample(mov_image, sample_grid, mode = mod, align_corners = True, padding_mode="border")
        
        return sample_grid, warped

class UNet(nn.Module):
    def __init__(self, start_channel, input_dim=1):
        super(UNet, self).__init__()

        self.start_channel = start_channel
        self.input_dim = input_dim
        bias_opt = True

        # -------------------------------------------------------------
        # Level 4
        # -------------------------------------------------------------
        self.de_level_4_1 = res_encoder(
            self.start_channel * 4,
            self.start_channel * 4,
            kernel_size=3,
            stride=1,
            bias=bias_opt,
        )
        self.de_level_4_2 = res_encoder(
            self.start_channel * 4,
            self.start_channel * 4,
            kernel_size=3,
            stride=1,
            bias=bias_opt,
        )
        self.de_level_4_3 = res_encoder(
            self.start_channel * 4,
            self.start_channel * 4,
            kernel_size=3,
            stride=1,
            bias=bias_opt,
        )

        # Feature propagation remains separate from flow propagation.
        self.up_level_4to3 = self.decoder(
            self.start_channel * 4,
            self.start_channel * 2,
            kernel_size=(2, 2),
            stride=(2, 2),
        )

        # -------------------------------------------------------------
        # Level 3
        # -------------------------------------------------------------
        self.de_level_3_1 = skip_encoder(
            self.start_channel * 2 + 4,
            self.start_channel * 2,
            kernel_size=3,
            stride=1,
            bias=bias_opt,
        )
        self.de_level_3_2 = res_encoder(
            self.start_channel * 2,
            self.start_channel * 2,
            kernel_size=3,
            stride=1,
            bias=bias_opt,
        )
        self.de_level_3_3 = res_encoder(
            self.start_channel * 2,
            self.start_channel * 2,
            kernel_size=3,
            stride=1,
            bias=bias_opt,
        )

        self.up_level_3to2 = self.decoder(
            self.start_channel * 2,
            self.start_channel,
        )

        # -------------------------------------------------------------
        # Level 2
        # -------------------------------------------------------------
        self.de_level_2_1 = skip_encoder(
            self.start_channel + 4,
            self.start_channel,
            kernel_size=3,
            stride=1,
            bias=bias_opt,
        )
        self.de_level_2_2 = res_encoder(
            self.start_channel,
            self.start_channel,
            kernel_size=3,
            stride=1,
            bias=bias_opt,
        )
        self.de_level_2_3 = res_encoder(
            self.start_channel,
            self.start_channel,
            kernel_size=3,
            stride=1,
            bias=bias_opt,
        )

        # -------------------------------------------------------------
        # One residual-flow head at each registration scale.
        #
        # Each head initially produces:
        #     (N, 3, D, H, W)
        #
        # Channel ordering is:
        #     channel 0: displacement in D
        #     channel 1: displacement in H
        #     channel 2: displacement in W
        # -------------------------------------------------------------
        self.flow_small_4_3 = self.outputs(
            self.start_channel * 4,
            2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.flow_small_3_3 = self.outputs(
            self.start_channel * 2,
            2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.flow_small_2_3 = self.outputs(
            self.start_channel,
            2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # -------------------------------------------------------------
        # Gating
        # -------------------------------------------------------------
        self.Gating_Selection_Level_4 = Gating_Selection(
            self.input_dim,
            self.input_dim,
        )

        self.Gating_Intensity_Level_4 = Gating_Intensity_Structure(
            self.input_dim,
            self.start_channel * 4 // 4,
        )
        self.Gating_Structure_Level_4 = Gating_Intensity_Structure(
            self.input_dim,
            self.start_channel * 4 // 4,
        )

        self.Gating_Selection_Level_3 = Gating_Selection(
            self.start_channel * 2 + 4,
            self.start_channel * 2 + 4,
        )
        self.Gating_Selection_Level_2 = Gating_Selection(
            self.start_channel + 4,
            self.start_channel + 4,
        )

        # -------------------------------------------------------------
        # Attention
        # -------------------------------------------------------------
        self.AttentionFusion_Level_4_Ins = AttentionFusion(
            in_channels=self.start_channel * 4 // 2,
            squeeze_ratio=self.start_channel // 2,
        )
        self.AttentionFusion_Level_4_Str = AttentionFusion(
            in_channels=self.start_channel * 4 // 2,
            squeeze_ratio=self.start_channel // 2,
        )
        self.AttentionFusion_Level_4 = AttentionFusion(
            in_channels=self.start_channel * 4,
            squeeze_ratio=self.start_channel,
        )

        self.torch_sobel = Sobel()

        # Uses the SpatialTransform supplied in the question.
        self.spatial_transform = SpatialTransform_WithoutGrid()

        # Image pyramid sizes.
        self.avgp_level_2 = nn.AdaptiveAvgPool2d((128, 128))
        self.avgp_level_3 = nn.AdaptiveAvgPool2d((64, 64))
        self.avgp_level_4 = nn.AdaptiveAvgPool2d((32, 32))

        # Start each registration level from zero displacement.
        self._zero_initialize_flow_head(self.flow_small_4_3)
        self._zero_initialize_flow_head(self.flow_small_3_3)
        self._zero_initialize_flow_head(self.flow_small_2_3)

    def decoder(
        self,
        in_channels,
        out_channels,
        kernel_size=2,
        stride=2,
        padding=0,
        output_padding=0,
        bias=True,
    ):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=bias,
            ),
            nn.PReLU(num_parameters=out_channels),
        )

    def outputs(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=False,
        batchnorm=False,
    ):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        )

    @staticmethod
    def _zero_initialize_flow_head(flow_head):
        """
        Make the initial deformation approximately identity.
        """
        conv = flow_head[-1]

        nn.init.zeros_(conv.weight)

        if conv.bias is not None:
            nn.init.zeros_(conv.bias)

    @staticmethod
    def _flow_to_grid_layout(flow_cf):
        """
        Convert a channel-first flow:

            (N, 2, H, W)

        into the layout required by SpatialTransform:

            (N, 2, W, 3)

        The last dimension remains ordered as (h, w).
        """
        if flow_cf.ndim != 4 or flow_cf.shape[1] != 2:
            raise ValueError(
                "Expected flow with shape (N, 2, H, W), "
                f"but received {tuple(flow_cf.shape)}."
            )

        return flow_cf.permute(0, 2, 3, 1).contiguous()

    @classmethod
    def fourier_resize_flow(cls, flow_cf, target_size):
        """
        Enlarge a normalized 2-D displacement using Fourier zero padding.

        Parameters
        ----------
        flow_cf:
            Flow with shape (N, 2, H, W).

        target_size:
            Target spatial size as (target_H, target_W).

        Returns
        -------
        Tensor
            Resized flow with shape
            (N, 2, target_H, target_W).

        Notes
        -----
        SpatialTransform adds the flow directly to a grid in [-1, 1].
        Therefore the flow is in normalized grid coordinates.

        The vector components must not be multiplied by spatial scale
        ratios when moving between pyramid levels. The FFT amplitude
        correction below is still necessary.
        """
        if flow_cf.ndim != 4 or flow_cf.shape[1] != 2:
            raise ValueError(
                "Expected flow with shape (N, 2, H, W), "
                f"but received {tuple(flow_cf.shape)}."
            )

        source_size = tuple(int(v) for v in flow_cf.shape[-2:])
        target_size = tuple(int(v) for v in target_size)

        if len(target_size) != 2:
            raise ValueError(
                "target_size must contain exactly three values: H, W."
            )

        if any(target < source for source, target in zip(source_size, target_size)):
            raise ValueError(
                "fourier_resize_flow only supports enlargement. "
                f"Source size: {source_size}; target size: {target_size}."
            )

        if source_size == target_size:
            return flow_cf

        source_h, source_w = source_size
        target_h, target_w = target_size

        difference_h = target_h - source_h
        difference_w = target_w - source_w

        pad_h_left = difference_h // 2
        pad_h_right = difference_h - pad_h_left

        pad_w_left = difference_w // 2
        pad_w_right = difference_w - pad_w_left

        spatial_dimensions = (-2, -1)

        # FFT may require float32 when mixed precision is active.
        original_dtype = flow_cf.dtype

        if flow_cf.dtype in (torch.float16, torch.bfloat16):
            fft_input = flow_cf.float()
        else:
            fft_input = flow_cf

        spectrum = torch.fft.fftn(
            fft_input,
            dim=spatial_dimensions,
        )

        spectrum = torch.fft.fftshift(
            spectrum,
            dim=spatial_dimensions,
        )

        # F.pad lists dimensions in reverse order:
        #
        # (W-left, W-right,
        #  H-left, H-right)
        spectrum = F.pad(
            spectrum,
            (
                pad_w_left,
                pad_w_right,
                pad_h_left,
                pad_h_right,
            ),
            mode="constant",
            value=0,
        )

        spectrum = torch.fft.ifftshift(
            spectrum,
            dim=spatial_dimensions,
        )

        resized_flow = torch.fft.ifftn(
            spectrum,
            dim=spatial_dimensions,
        ).real

        # Without this correction, a constant displacement becomes smaller
        # after Fourier zero padding because ifftn divides by the target
        # number of samples.
        source_volume = source_h * source_w
        target_volume = target_h * target_w

        resized_flow = resized_flow * (
            float(target_volume) / float(source_volume)
        )

        return resized_flow.to(dtype=original_dtype)

    def warp_with_flow(self, tensor, flow_cf, mode="bilinear"):
        """
        Warp an image, feature map, or another flow field.

        tensor:
            (N, C, H, W)

        flow_cf:
            (N, 3, H, W)
        """
        if tensor.shape[-2:] != flow_cf.shape[-2:]:
            raise ValueError(
                "Tensor and flow must have equal spatial dimensions. "
                f"Tensor: {tuple(tensor.shape[-2:])}; "
                f"flow: {tuple(flow_cf.shape[-2:])}."
            )

        flow_grid = self._flow_to_grid_layout(flow_cf)

        return self.spatial_transform(
            tensor,
            flow_grid,
            mod=mode,
        )

    def compose_flows(self, coarse_flow_cf, residual_flow_cf):
        """
        Spatially compose an existing flow with a new residual flow.

        Your SpatialTransform samples the moving image at:

            p + flow(p)

        Suppose an image is first warped with coarse_flow and then warped
        with residual_flow:

            warp(warp(image, coarse), residual)

        The corresponding composed displacement is:

            total(p)
                = residual(p)
                + coarse(p + residual(p))

        This is not generally equivalent to:

            coarse + residual
        """
        if coarse_flow_cf.shape != residual_flow_cf.shape:
            raise ValueError(
                "Flows must have the same shape before composition. "
                f"Coarse: {tuple(coarse_flow_cf.shape)}; "
                f"residual: {tuple(residual_flow_cf.shape)}."
            )

        coarse_at_residual_locations = self.warp_with_flow(
            coarse_flow_cf,
            residual_flow_cf,
            mode="bilinear",
        )

        composed_flow = (
            residual_flow_cf
            + coarse_at_residual_locations
        )

        return composed_flow

    def forward(self, x, y):
        """
        x:
            Moving image.

        y:
            Fixed image.

        Returns
        -------
        Tensor
            Final normalized displacement with shape:

                (N, H, W, 2)

            It can be passed directly to the supplied SpatialTransform.
        """
        if x.shape != y.shape:
            raise ValueError(
                "Moving and fixed images must have identical shapes. "
                f"Received x={tuple(x.shape)} and y={tuple(y.shape)}."
            )

        # =============================================================
        # Construct image pyramids
        # =============================================================
        x_ins_level_2 = self.avgp_level_2(x)
        x_ins_level_3 = self.avgp_level_3(x_ins_level_2)
        x_ins_level_4 = self.avgp_level_4(x_ins_level_3)

        y_ins_level_2 = self.avgp_level_2(y)
        y_ins_level_3 = self.avgp_level_3(y_ins_level_2)
        y_ins_level_4 = self.avgp_level_4(y_ins_level_3)

        x_str_level_2 = self.torch_sobel(x_ins_level_2)
        x_str_level_3 = self.torch_sobel(x_ins_level_3)
        x_str_level_4 = self.torch_sobel(x_ins_level_4)

        y_str_level_2 = self.torch_sobel(y_ins_level_2)
        y_str_level_3 = self.torch_sobel(y_ins_level_3)
        y_str_level_4 = self.torch_sobel(y_ins_level_4)

        # if x_feature.shape[-3:] != x_ins_level_4.shape[-3:]:
            # raise ValueError(
                # "x_feature and the Level-4 moving image must have the "
                # "same spatial size. "
                # f"x_feature={tuple(x_feature.shape[-3:])}, "
                # f"x_level_4={tuple(x_ins_level_4.shape[-3:])}."
            # )

        # if y_feature.shape[-3:] != y_ins_level_4.shape[-3:]:
            # raise ValueError(
                # "y_feature and the Level-4 fixed image must have the "
                # "same spatial size. "
                # f"y_feature={tuple(y_feature.shape[-3:])}, "
                # f"y_level_4={tuple(y_ins_level_4.shape[-3:])}."
            # )

        # =============================================================
        # Level 4: coarse displacement
        # =============================================================
        x_feature, y_feature = x_ins_level_4, y_ins_level_4
        diff_level_4 = torch.abs(x_feature - y_feature)

        score_level_4 = self.Gating_Selection_Level_4(
            diff_level_4
        )

        x_gatingfeature_level_4 = (
            x_feature * score_level_4
        )
        y_gatingfeature_level_4 = (
            y_feature * score_level_4
        )

        x_insfeat_level_4 = self.Gating_Intensity_Level_4(
            x_gatingfeature_level_4,
            x_ins_level_4,
        )
        x_strfeat_level_4 = self.Gating_Structure_Level_4(
            x_gatingfeature_level_4,
            x_str_level_4,
        )

        y_insfeat_level_4 = self.Gating_Intensity_Level_4(
            y_gatingfeature_level_4,
            y_ins_level_4,
        )
        y_strfeat_level_4 = self.Gating_Structure_Level_4(
            y_gatingfeature_level_4,
            y_str_level_4,
        )

        xy_insfeat_level_4 = self.AttentionFusion_Level_4_Ins(
            x_insfeat_level_4,
            y_insfeat_level_4,
        )
        xy_strfeat_level_4 = self.AttentionFusion_Level_4_Str(
            x_strfeat_level_4,
            y_strfeat_level_4,
        )

        xy_ins_str_feat_level_4 = self.AttentionFusion_Level_4(
            xy_insfeat_level_4,
            xy_strfeat_level_4,
        )

        de_level_4_1 = self.de_level_4_1(
            xy_ins_str_feat_level_4
        )
        de_level_4_2 = self.de_level_4_2(
            de_level_4_1
        )
        de_level_4_3 = self.de_level_4_3(
            de_level_4_2
        )

        # Level-4 residual flow:
        #
        # expected spatial size: (64, 32, 32)
        residual_flow_level_4 = self.flow_small_4_3(
            de_level_4_3
        )

        # Preserve decoder features for the Level-3 network.
        dup_level_4to3 = self.up_level_4to3(
            de_level_4_3
        )

        # Fourier enlargement:
        #
        # (64, 32, 32) -> (64, 64, 64)
        flow_level_4_to_3 = self.fourier_resize_flow(
            residual_flow_level_4,
            target_size=x_ins_level_3.shape[-2:],
        )

        # Warp only the moving Level-3 image and structure.
        x_ins_level_3_warped = self.warp_with_flow(
            x_ins_level_3,
            flow_level_4_to_3,
        )
        x_str_level_3_warped = self.warp_with_flow(
            x_str_level_3,
            flow_level_4_to_3,
        )

        # =============================================================
        # Level 3: residual displacement
        # =============================================================
        dup_level_4to3_cue = torch.cat(
            (
                dup_level_4to3,
                x_ins_level_3_warped,
                y_ins_level_3,
                x_str_level_3_warped,
                y_str_level_3,
            ),
            dim=1,
        )

        score_dup_level_4to3_cue = (
            self.Gating_Selection_Level_3(
                dup_level_4to3_cue
            )
        )

        de_level_3_1 = self.de_level_3_1(
            score_dup_level_4to3_cue
            * dup_level_4to3_cue,
            dup_level_4to3,
        )
        de_level_3_2 = self.de_level_3_2(
            de_level_3_1
        )
        de_level_3_3 = self.de_level_3_3(
            de_level_3_2
        )

        residual_flow_level_3 = self.flow_small_3_3(
            de_level_3_3
        )

        # Compose rather than simply adding the flows.
        composed_flow_level_3 = self.compose_flows(
            coarse_flow_cf=flow_level_4_to_3,
            residual_flow_cf=residual_flow_level_3,
        )

        # Preserve decoder features for the Level-2 network.
        dup_level_3to2 = self.up_level_3to2(
            de_level_3_3
        )

        # Fourier enlargement:
        #
        # (64, 64, 64) -> (128, 128, 128)
        flow_level_3_to_2 = self.fourier_resize_flow(
            composed_flow_level_3,
            target_size=x_ins_level_2.shape[-2:],
        )

        # Warp only the moving Level-2 image and structure.
        x_ins_level_2_warped = self.warp_with_flow(
            x_ins_level_2,
            flow_level_3_to_2,
        )
        x_str_level_2_warped = self.warp_with_flow(
            x_str_level_2,
            flow_level_3_to_2,
        )

        # =============================================================
        # Level 2: residual displacement
        # =============================================================
        dup_level_3to2_cue = torch.cat(
            (
                dup_level_3to2,
                x_ins_level_2_warped,
                y_ins_level_2,
                x_str_level_2_warped,
                y_str_level_2,
            ),
            dim=1,
        )

        score_dup_level_3to2_cue = (
            self.Gating_Selection_Level_2(
                dup_level_3to2_cue
            )
        )

        de_level_2_1 = self.de_level_2_1(
            score_dup_level_3to2_cue
            * dup_level_3to2_cue,
            dup_level_3to2,
        )
        de_level_2_2 = self.de_level_2_2(
            de_level_2_1
        )
        de_level_2_3 = self.de_level_2_3(
            de_level_2_2
        )

        residual_flow_level_2 = self.flow_small_2_3(
            de_level_2_3
        )

        composed_flow_level_2 = self.compose_flows(
            coarse_flow_cf=flow_level_3_to_2,
            residual_flow_cf=residual_flow_level_2,
        )

        # =============================================================
        # Full-resolution displacement
        # =============================================================
        # For the original input this should dynamically perform:
        #
        # (128, 128) -> (192, 224)
        flow_final_cf = self.fourier_resize_flow(
            composed_flow_level_2,
            target_size=x.shape[-2:],
        )

        # Make the result directly compatible with SpatialTransform:
        #
        # (N, 2, H, W) -> (N, H, W, 2)
        flow_final = self._flow_to_grid_layout(
            flow_final_cf
        )

        return flow_final.permute(0, 3, 1, 2).contiguous()

class SpatialTransform_WithoutGrid(nn.Module):
    def __init__(self):
        super(SpatialTransform_WithoutGrid, self).__init__()
    def forward(self, mov_image, flow, mod = 'bilinear'):
        # d2, h2, w2 = mov_image.shape[-3:]
        h2, w2 = mov_image.shape[-2:]
        # grid_d, grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.to(flow.device).float()
        # grid_d = grid_d.to(flow.device).float()
        grid_w = grid_w.to(flow.device).float()
        # grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow_h = flow[:,:,:,0]
        flow_w = flow[:,:,:,1]
        # flow_w = flow[:,:,:,2]
        #Softsign
        #disp_d = (grid_d + (flow_d * 2 / d2)).squeeze(1)
        #disp_h = (grid_h + (flow_h * 2 / h2)).squeeze(1)
        #disp_w = (grid_w + (flow_w * 2 / w2)).squeeze(1)
        
        # Remove Channel Dimension
        # disp_d = (grid_d + (flow_d)).squeeze(1)
        disp_h = (grid_h + (flow_h)).squeeze(1)
        disp_w = (grid_w + (flow_w)).squeeze(1)
        sample_grid = torch.stack((disp_w, disp_h), 3)  # shape (N, D, H, W, 3)
        warped = torch.nn.functional.grid_sample(mov_image, sample_grid, mode = mod, align_corners = True)
        
        return warped

class DiffeomorphicTransform(nn.Module):
    def __init__(self, time_step=7):
        super(DiffeomorphicTransform, self).__init__()
        self.time_step = time_step

    def forward(self, flow):
    
        # print(flow.shape)
        h2, w2 = flow.shape[-2:]
        grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.to(flow.device).float()
        grid_w = grid_w.to(flow.device).float()
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow = flow / (2 ** self.time_step)

        
        
        for i in range(self.time_step):
            flow_h = flow[:,0,:,:]
            flow_w = flow[:,1,:,:]
            disp_h = (grid_h + flow_h).squeeze(1)
            disp_w = (grid_w + flow_w).squeeze(1)
            
            deformation = torch.stack((disp_w,disp_h), dim=3)
    
            # print(deformation.shape)
            flow = flow + torch.nn.functional.grid_sample(flow, deformation, mode='bilinear',padding_mode="border", align_corners = True)
            #Softsign
            #disp_d = (grid_d + (flow_d * 2 / d2)).squeeze(1)
            #disp_h = (grid_h + (flow_h * 2 / h2)).squeeze(1)
            #disp_w = (grid_w + (flow_w * 2 / w2)).squeeze(1)
            
            # Remove Channel Dimension
            #disp_h = (grid_h + (flow_h)).squeeze(1)
            #disp_w = (grid_w + (flow_w)).squeeze(1)

            #sample_grid = torch.stack((disp_w, disp_h), 3)  # shape (N, H, W, 2)
            #flow = torch.nn.functional.grid_sample(mov_image, sample_grid, mode = mod)
        
        return flow


def smoothloss(y_pred):
    #print('smoothloss y_pred.shape    ',y_pred.shape)
    #[N,3,D,H,W]
    h2, w2 = y_pred.shape[-2:]
    # dy = torch.abs(y_pred[:,:,1:, :, :] - y_pred[:,:, :-1, :, :]) / 2 * d2
    dx = torch.abs(y_pred[:,:, 1:, :] - y_pred[:, :, :-1, :]) / 2 * h2
    dz = torch.abs(y_pred[:,:, :, 1:] - y_pred[:, :, :, :-1]) / 2 * w2
    return (torch.mean(dx * dx) + torch.mean(dz*dz))/2.0


"""
Normalized local cross-correlation function in Pytorch. Modified from https://github.com/voxelmorph/voxelmorph.
"""
class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=5, eps=1e-5):
        super(NCC, self).__init__()
        self.win_raw = win
        self.eps = eps
        self.win = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.win_raw
        self.win = [self.win_raw] * ndims

        weight_win_size = self.win_raw
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv2d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)
class SAD:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))
