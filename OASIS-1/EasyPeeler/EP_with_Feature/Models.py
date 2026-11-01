import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange
#https://github.com/ShiqiYu/OpenGait/blob/master/opengait/modeling/models/skeletongait%2B%2B.py

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
class UNet(nn.Module):
    def __init__(self, start_channel, feat_channel=1):
        self.start_channel = start_channel
        # self.scale_factor = scale_factor
        self.feat_channel = feat_channel
        bias_opt = True

        super(UNet, self).__init__()
        
        self.de_level_4_1 = res_encoder(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.de_level_4_2 = res_encoder(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.de_level_4_3 = res_encoder(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.up_level_4to3 = self.decoder(self.start_channel * 4, self.start_channel * 2)
        self.de_level_3_1 = skip_encoder(self.start_channel * 2 + 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.de_level_3_2 = res_encoder(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.de_level_3_3 = res_encoder(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.up_level_3to2 = self.decoder(self.start_channel * 2, self.start_channel * 1)
        self.de_level_2_1 = skip_encoder(self.start_channel * 1 + 4, self.start_channel * 1, kernel_size=3, stride=1, bias=bias_opt)
        self.de_level_2_2 = res_encoder(self.start_channel * 1, self.start_channel * 1, kernel_size=3, stride=1, bias=bias_opt)
        self.de_level_2_3 = res_encoder(self.start_channel * 1, self.start_channel * 1, kernel_size=3, stride=1, bias=bias_opt)
        # self.up_level_2to1 = self.decoder(self.start_channel * 1, self.start_channel * 1)
        # self.de_level_1_1 = skip_encoder(self.start_channel * 1 + 4, self.start_channel * 1, kernel_size=3, stride=1, bias=bias_opt)
        # self.de_level_1_2 = res_encoder(self.start_channel * 1, self.start_channel * 1, kernel_size=3, stride=1, bias=bias_opt)
        # self.de_level_1_3 = res_encoder(self.start_channel * 1, self.start_channel * 1, kernel_size=3, stride=1, bias=bias_opt)
        
        
        
        self.flow_small = self.outputs(self.start_channel * 1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        
        
        self.Gating_Selection_Level_4 = Gating_Selection(self.feat_channel, self.feat_channel)
        self.Gating_Intensity_Level_4 = Gating_Intensity_Structure(self.feat_channel, self.start_channel * 4//4)
        self.Gating_Structure_Level_4 = Gating_Intensity_Structure(self.feat_channel, self.start_channel * 4//4)
        self.Gating_Selection_Level_3 = Gating_Selection(self.start_channel * 2 + 4, self.start_channel * 2 + 4)
        self.Gating_Selection_Level_2 = Gating_Selection(self.start_channel * 1 + 4, self.start_channel * 1 + 4)
        # self.Gating_Selection_Level_1 = Gating_Selection(self.start_channel * 1 + 4, self.start_channel * 1 + 4)
        
        self.AttentionFusion_Level_4_Ins = AttentionFusion(in_channels=self.start_channel * 4 // 2, squeeze_ratio = self.start_channel // 2)
        self.AttentionFusion_Level_4_Str = AttentionFusion(in_channels=self.start_channel * 4 // 2, squeeze_ratio = self.start_channel // 2)
        self.AttentionFusion_Level_4 = AttentionFusion(in_channels=self.start_channel * 4, squeeze_ratio = self.start_channel)
        # self.AttentionFusion80 = AttentionFusion(in_channels=self.start_channel * 4 * 2, squeeze_ratio=self.start_channel)
        
        self.torch_sobel = Sobel()
        
        # self.avgp_level_1 = nn.AdaptiveAvgPool2d((128, 128))
        self.avgp_level_2 = nn.AdaptiveAvgPool2d((128, 128))
        self.avgp_level_3 = nn.AdaptiveAvgPool2d((64, 64))
        self.avgp_level_4 = nn.AdaptiveAvgPool2d((32, 32))
        
        self.softsign = nn.Softsign()
        

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias),
            nn.PReLU(num_parameters=out_channels))
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, batchnorm=False):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer
    

    def forward(self, x_feature, y_feature, x, y):#, x_80, y_80):
        
        
        assert 32 ==x_feature.shape[-1], x_feature.shape
        assert 192 ==x.shape[-1]
        # x_40 = torch.nn.functional.interpolate(x_40, size=[50, 60], mode='bilinear', align_corners=True)
        # y_40 = torch.nn.functional.interpolate(y_40, size=[50, 60], mode='bilinear', align_corners=True)
        
        # x_ins_level_1 = self.avgp_level_1(x)
        x_ins_level_2 = self.avgp_level_2(x)
        x_ins_level_3 = self.avgp_level_3(x_ins_level_2)
        x_ins_level_4 = self.avgp_level_4(x_ins_level_3)
        
        
        # y_ins_level_1 = self.avgp_level_1(y)
        y_ins_level_2 = self.avgp_level_2(y)
        y_ins_level_3 = self.avgp_level_3(y_ins_level_2)
        y_ins_level_4 = self.avgp_level_4(y_ins_level_3)
        
        # x_str_level_1 = self.torch_sobel(x_ins_level_1)
        x_str_level_2 = self.torch_sobel(x_ins_level_2)
        x_str_level_3 = self.torch_sobel(x_ins_level_3)
        x_str_level_4 = self.torch_sobel(x_ins_level_4)
        
        # y_str_level_1 = self.torch_sobel(y_ins_level_1)
        y_str_level_2 = self.torch_sobel(y_ins_level_2)
        y_str_level_3 = self.torch_sobel(y_ins_level_3)
        y_str_level_4 = self.torch_sobel(y_ins_level_4)
        
        
        
        diff_level_4 = torch.abs(x_feature - y_feature)
        score = self.Gating_Selection_Level_4(diff_level_4)
        
        x_gatingfeature_level_4 = x_feature * score #self.Gating_Selection_MF_40(diff_40)
        y_gatingfeature_level_4 = y_feature * score #self.Gating_Selection_MF_40(diff_40)
        
        x_insfeat_level_4 = self.Gating_Intensity_Level_4(x_gatingfeature_level_4, x_ins_level_4)
        x_strfeat_level_4 = self.Gating_Structure_Level_4(x_gatingfeature_level_4, x_str_level_4)
        
        y_insfeat_level_4 = self.Gating_Intensity_Level_4(y_gatingfeature_level_4, y_ins_level_4)
        y_strfeat_level_4 = self.Gating_Structure_Level_4(y_gatingfeature_level_4, y_str_level_4)
        
        xy_insfeat_level_4 = self.AttentionFusion_Level_4_Ins(x_insfeat_level_4, y_insfeat_level_4)
        xy_strfeat_level_4 = self.AttentionFusion_Level_4_Str(x_strfeat_level_4, y_strfeat_level_4)
        
        xy_ins_str_feat_level_4 = self.AttentionFusion_Level_4(xy_insfeat_level_4, xy_strfeat_level_4)
        
        de_level_4_1 = self.de_level_4_1(xy_ins_str_feat_level_4)
        de_level_4_2 = self.de_level_4_2(de_level_4_1)
        de_level_4_3 = self.de_level_4_3(de_level_4_2)
        
        dup_level_4to3 = self.up_level_4to3(de_level_4_3)
        
        
        dup_level_4to3_cue = torch.cat((dup_level_4to3, x_ins_level_3, y_ins_level_3, x_str_level_3, y_str_level_3), dim = 1)
        score_dup_level_4to3_cue = self.Gating_Selection_Level_3(dup_level_4to3_cue)
        
        de_level_3_1 = self.de_level_3_1(score_dup_level_4to3_cue * dup_level_4to3_cue, dup_level_4to3)
        de_level_3_2 = self.de_level_3_2(de_level_3_1)
        de_level_3_3 = self.de_level_3_3(de_level_3_2)
        
        
        dup_level_3to2 = self.up_level_3to2(de_level_3_3)
        
        
        dup_level_3to2_cue = torch.cat((dup_level_3to2, x_ins_level_2, y_ins_level_2, x_str_level_2, y_str_level_2), dim = 1)
        score_dup_level_3to2_cue = self.Gating_Selection_Level_2(dup_level_3to2_cue)
        
        de_level_2_1 = self.de_level_2_1(score_dup_level_3to2_cue * dup_level_3to2_cue, dup_level_3to2)
        de_level_2_2 = self.de_level_2_2(de_level_2_1)
        de_level_2_3 = self.de_level_2_3(de_level_2_2)
        
        # dup_level_2to1 = self.up_level_2to1(de_level_2_3)
        
        
        # dup_level_2to1_cue = torch.cat((dup_level_2to1, x_ins_level_1, y_ins_level_1, x_str_level_1, y_str_level_1), dim = 1)
        # score_dup_level_2to1_cue = self.Gating_Selection_Level_1(dup_level_2to1_cue)
        
        # de_level_1_1 = self.de_level_1_1(score_dup_level_2to1_cue * dup_level_2to1_cue, dup_level_2to1)
        # de_level_1_2 = self.de_level_1_2(de_level_1_1)
        # de_level_1_3 = self.de_level_1_3(de_level_1_2)
        
        
        
        flow_small = self.flow_small(de_level_2_3)
        '''
        out_1, out_2 = f_xy[:,0:1,:,:], f_xy[:,1:2,:,:]
        out_1 = out_1.squeeze().squeeze()
        out_2 = out_2.squeeze().squeeze()
        out_ifft1 = torch.fft.fftshift(torch.fft.fft2(out_1))
        out_ifft2 = torch.fft.fftshift(torch.fft.fft2(out_2))
        # p3d = (48, 48, 40, 40)
        # p3d = (24, 24, 20, 20)
        # p3d = (36, 36, 30, 30)
        p3d = (int((192 - self.scale_factor * 48)//2), int((192 - self.scale_factor * 48)//2), int((160 - self.scale_factor * 40)//2), int((160 - self.scale_factor * 40)//2))
        out_ifft1 = F.pad(out_ifft1, p3d, "constant", 0)
        out_ifft2 = F.pad(out_ifft2, p3d, "constant", 0)
        
        disp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft1)))
        disp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft2)))
        
        fxy_1 = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)
        '''
        
        out_ifft = torch.fft.fftshift(torch.fft.fftn(flow_small,dim=(2,3)))
        p3d = (int((192 - 4 * 32)//2), int((192 - 4 * 32)//2), int((160 - 4 * 32)//2), int((160 - 4 * 32)//2))
        out_ifft = F.pad(out_ifft, p3d, "constant", 0)
        flow = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft), dim =(2,3)))
        
        return flow
        
        # f_xy = self.softsign(fxy_1)
        
        
        # return f_xy



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
