import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Sobelfilter = nn.Conv3d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=0, bias=False)

        Gx = torch.tensor([[[1.0, 0.0, -1.0], 
                            [2.0, 0.0, -2.0], 
                            [1.0, 0.0, -1.0]],

                           [[2.0, 0.0, -2.0], 
                            [4.0, 0.0, -4.0], 
                            [2.0, 0.0, -2.0]],

                           [[1.0, 0.0, -1.0], 
                            [2.0, 0.0, -2.0], 
                            [1.0, 0.0, -1.0]]])

        Gy = torch.tensor([[[1.0, 2.0, 1.0], 
                            [0.0, 0.0, 0.0], 
                            [-1.0, -2.0, -1.0]],

                           [[2.0, 4.0, 2.0], 
                            [0.0, 0.0, 0.0], 
                            [-2.0, -4.0, -2.0]],

                           [[1.0, 2.0, 1.0], 
                            [0.0, 0.0, 0.0], 
                            [-1.0, -2.0, -1.0]]])

        Gz = torch.tensor([[[1.0, 2.0, 1.0], 
                            [2.0, 4.0, 2.0], 
                            [1.0, 2.0, 1.0]],

                           [[0.0, 0.0, 0.0], 
                            [0.0, 0.0, 0.0], 
                            [0.0, 0.0, 0.0]],

                           [[-1.0, -2.0, -1.0], 
                            [-2.0, -4.0, -2.0], 
                            [-1.0, -2.0, -1.0]]])

        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0), Gz.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.Sobelfilter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.Sobelfilter(img)
        pad = (1, 1, 1, 1, 1, 1)
        x = F.pad(x, pad, "constant", 0)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x


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
            nn.Conv3d(in_channels, out_channels//2, kernel_size =1, stride=1, padding=0, bias=bias),
            nn.PReLU(num_parameters=out_channels//2),
                                                                                                                             
                     
                                  
            nn.Conv3d(out_channels//2, out_channels//2, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.PReLU(num_parameters=out_channels//2),
             
                                  
            nn.Conv3d(out_channels//2, out_channels, kernel_size = 1, stride=1, padding=0, bias=bias),
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
            nn.Conv3d(in_channels, out_channels//2, kernel_size =1, stride=1, padding=0, bias=bias),
            nn.PReLU(num_parameters=out_channels//2),
            nn.Conv3d(out_channels//2, out_channels//2, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.PReLU(num_parameters=out_channels//2),
                                  
            nn.Conv3d(out_channels//2, out_channels, kernel_size = 1, stride=1, padding=0, bias=bias),
            nn.PReLU(num_parameters=out_channels))
        return layer
    def forward(self, inputs, inputs2):
        outputs = self.conv(inputs)
        return outputs + inputs2

class AttentionFusion(nn.Module): 
    def __init__(self, in_channels=64, squeeze_ratio=16):
        super(AttentionFusion, self).__init__()
        hidden_dim = int(in_channels / squeeze_ratio)
        self.conv = self.fuse_conv_encoder(in_channels= in_channels, out_channels = hidden_dim)
        # self.conv2 = self.conv_encoder(in_channels= in_channels, out_channels = in_channels //2)
    
    def fuse_conv_encoder(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size = 3, stride=1, padding=1, bias=True, groups = in_channels)
            )
                       
        return layer

    # def conv_encoder(self, in_channels, out_channels):
                                             
                     
        # layer = nn.Sequential(
            # nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
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
            nn.Conv3d(in_channels, in_channels, kernel_size = 3, stride=1, padding=1, bias=True, groups = in_channels)
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
            nn.Conv3d(in_channels, out_channels //2, kernel_size = 1, stride=1, padding=0, bias=True),#, groups = in_channels
            nn.PReLU(num_parameters=out_channels //2),
            nn.Conv3d(out_channels //2, out_channels //2, kernel_size = 3, stride=1, padding=1, bias=True),
            nn.PReLU(num_parameters=out_channels //2),
            nn.Conv3d(out_channels //2, out_channels, kernel_size = 1, stride=1, padding=0, bias=True),
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

class UNet(nn.Module):
    def __init__(self, start_channel, feat_channel=1):
        self.start_channel = start_channel
        self.input_dim = feat_channel
        self.scale_factor = 2.0
        bias_opt = True

        super(UNet, self).__init__()
        
        self.de_level_4_1 = res_encoder(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.de_level_4_2 = res_encoder(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.de_level_4_3 = res_encoder(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.up_level_4to3 = self.decoder(self.start_channel * 4, self.start_channel * 2, kernel_size=(1,2,2), stride=(1,2,2))
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
        
        
        
        self.flow_small = self.outputs(self.start_channel * 1, 3, kernel_size=3, stride=1, padding=1, bias=False)
        
        
        self.Gating_Selection_Level_4 = Gating_Selection(self.input_dim, self.input_dim)
        self.Gating_Intensity_Level_4 = Gating_Intensity_Structure(self.input_dim, self.start_channel * 4//4)
        self.Gating_Structure_Level_4 = Gating_Intensity_Structure(self.input_dim, self.start_channel * 4//4)
        self.Gating_Selection_Level_3 = Gating_Selection(self.start_channel * 2 + 4, self.start_channel * 2 + 4)
        self.Gating_Selection_Level_2 = Gating_Selection(self.start_channel * 1 + 4, self.start_channel * 1 + 4)
        # self.Gating_Selection_Level_1 = Gating_Selection(self.start_channel * 1 + 4, self.start_channel * 1 + 4)
        
        self.AttentionFusion_Level_4_Ins = AttentionFusion(in_channels=self.start_channel * 4 // 2, squeeze_ratio = self.start_channel // 2)
        self.AttentionFusion_Level_4_Str = AttentionFusion(in_channels=self.start_channel * 4 // 2, squeeze_ratio = self.start_channel // 2)
        self.AttentionFusion_Level_4 = AttentionFusion(in_channels=self.start_channel * 4, squeeze_ratio = self.start_channel)
        # self.AttentionFusion80 = AttentionFusion(in_channels=self.start_channel * 4 * 2, squeeze_ratio=self.start_channel)
        
        self.torch_sobel = Sobel()
        
        # self.avgp_level_1 = nn.AdaptiveAvgPool3d((128, 128))
        self.avgp_level_2 = nn.AdaptiveAvgPool3d((128, 128, 128))
        self.avgp_level_3 = nn.AdaptiveAvgPool3d((64, 64, 64))
        self.avgp_level_4 = nn.AdaptiveAvgPool3d((64, 32, 32))
        
        self.softsign = nn.Softsign()
        

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias),
            nn.PReLU(num_parameters=out_channels))
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, batchnorm=False):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer
    

    def forward(self, x_feature, y_feature, x, y):
        
        

        fdd, fww, fhh = x_feature.shape[-3:] #64, 32,32
        dd, ww,hh = x.shape[-3:]
        
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
        
        out_ifft = torch.fft.fftshift(torch.fft.fftn(flow_small,dim =(2,3,4)))
        p3d = (int((hh - 4 * fhh)//2), int((hh - 4 * fhh)//2), int((ww - 4 * fww)//2), int((ww - 4 * fww)//2), int((dd - 2 * fdd)//2), int((dd - 2 * fdd)//2))
        out_ifft = F.pad(out_ifft, p3d, "constant", 0)
        flow = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft), dim =(2,3,4)))
        
        return flow
        
class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()
    def forward(self, mov_image, flow, mod = 'bilinear'):
        d2, h2, w2 = mov_image.shape[-3:]
        grid_d, grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.cuda().float()
        grid_d = grid_d.cuda().float()
        grid_w = grid_w.cuda().float()
        grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow_d = flow[:,:,:,:,0]
        flow_h = flow[:,:,:,:,1]
        flow_w = flow[:,:,:,:,2]
        #Softsign
        #disp_d = (grid_d + (flow_d * 2 / d2)).squeeze(1)
        #disp_h = (grid_h + (flow_h * 2 / h2)).squeeze(1)
        #disp_w = (grid_w + (flow_w * 2 / w2)).squeeze(1)
        
        # Remove Channel Dimension
        disp_d = (grid_d + (flow_d)).squeeze(1)
        disp_h = (grid_h + (flow_h)).squeeze(1)
        disp_w = (grid_w + (flow_w)).squeeze(1)

        sample_grid = torch.stack((disp_w, disp_h, disp_d), 4)  # shape (N, D, H, W, 3)
        warped = torch.nn.functional.grid_sample(mov_image, sample_grid, mode = mod, align_corners = True)
        
        return warped#, sample_grid
class SpatialTransform_1(nn.Module):
    def __init__(self):
        super(SpatialTransform_1, self).__init__()
    def forward(self, mov_image, flow, mod = 'bilinear'):
        d2, h2, w2 = mov_image.shape[-3:]
        grid_d, grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.cuda().float()
        grid_d = grid_d.cuda().float()
        grid_w = grid_w.cuda().float()
        grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow_d = flow[:,:,:,:,0]
        flow_h = flow[:,:,:,:,1]
        flow_w = flow[:,:,:,:,2]
        #Softsign
        #disp_d = (grid_d + (flow_d * 2 / d2)).squeeze(1)
        #disp_h = (grid_h + (flow_h * 2 / h2)).squeeze(1)
        #disp_w = (grid_w + (flow_w * 2 / w2)).squeeze(1)
        
        # Remove Channel Dimension
        disp_d = (grid_d + (flow_d)).squeeze(1)
        disp_h = (grid_h + (flow_h)).squeeze(1)
        disp_w = (grid_w + (flow_w)).squeeze(1)

        sample_grid = torch.stack((disp_w, disp_h, disp_d), 4)  # shape (N, D, H, W, 3)
        warped = torch.nn.functional.grid_sample(mov_image, sample_grid, mode = mod, align_corners = True)
        
        return warped, sample_grid

class DiffeomorphicTransform(nn.Module):
    def __init__(self, time_step=7):
        super(DiffeomorphicTransform, self).__init__()
        self.time_step = time_step

    def forward(self, flow):
    
        # print(flow.shape)
        d2, h2, w2 = flow.shape[-3:]
        grid_d, grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.cuda().float()
        grid_d = grid_d.cuda().float()
        grid_w = grid_w.cuda().float()
        grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow = flow / (2 ** self.time_step)
        
        
        for i in range(self.time_step):
            flow_d = flow[:,0,:,:,:]
            flow_h = flow[:,1,:,:,:]
            flow_w = flow[:,2,:,:,:]
            disp_d = (grid_d + flow_d).squeeze(1)
            disp_h = (grid_h + flow_h).squeeze(1)
            disp_w = (grid_w + flow_w).squeeze(1)
            
            deformation = torch.stack((disp_w, disp_h, disp_d), 4)  # shape (N, D, H, W, 3)
    
            # print(deformation.shape)
            flow = flow + torch.nn.functional.grid_sample(flow, deformation, mode='bilinear', padding_mode="border", align_corners = True)
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



if __name__ == "__main__":
    moving_img = torch.randn(1, 1, 160, 192, 224)
    fixed_img = torch.randn(1, 1, 160, 192, 224)
    moving_feat = torch.randn(1, 768, 64, 32, 32)
    fixed_feat = torch.randn(1, 768, 64, 32, 32)
    model = UNet(start_channel=32, feat_channel=768)
    displacement = model(moving_feat, fixed_feat, moving_img, fixed_img)
    print(displacement.shape)
    from torchinfo import summary
    summary(model, [(1, 768, 64, 32, 32),(1, 768, 64, 32, 32),(1, 1, 160, 192, 224),(1, 1, 160, 192, 224)])

  
