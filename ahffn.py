import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16
from collections import OrderedDict
from ..builder import NECKS

class DPA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2
        )
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.spatial_gating_unit = DPA(dim)
        self.activation = nn.ReLU()
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x_ = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + x_
        return x


class DPCA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DPCA, self).__init__()
        self.global_avg_pool_h = nn.AdaptiveAvgPool2d((1, None))
        self.global_avg_pool_v = nn.AdaptiveAvgPool2d((None, 1))
        self.conv_h = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv_v = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv_f = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.out_channels = out_channels

    def forward(self, x):
        h = self.global_avg_pool_h(x)
        v = self.global_avg_pool_v(x)
        h = self.conv_h(h)
        v = self.conv_v(v)
        f = self.conv_f(x)
        h = self.sigmoid(h)
        v = self.sigmoid(v)
        f = self.sigmoid(f)
        
        return h * v * f


class FFEM(nn.Module):

    def __init__(self, in_channels, out_channels, fft_norm='ortho'):
        # bn_layer not used
        super(FFEM, self).__init__()
        self.dpca = DPCA(in_channels * 2, out_channels * 2)
        self.bn = torch.nn.BatchNorm2d(in_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]
        # r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        ffted = self.dpca(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        return output 

def BasicConv(filter_in, filter_out, kernel_size, stride=1, pad=None):
    if not pad:
        pad = (kernel_size - 1) // 2 if kernel_size else 0
    else:
        pad = pad
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU(inplace=True)),
    ]))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, filter_in, filter_out):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(filter_in, filter_out, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filter_out, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filter_out, filter_out, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filter_out, momentum=0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        )
    def forward(self, x):
        x = self.upsample(x)

        return x


class Downsample_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x2, self).__init__()

        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 2, 2, 0)
        )

    def forward(self, x, ):
        x = self.downsample(x)

        return x


class Downsample_x4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x4, self).__init__()

        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 4, 4, 0)
        )

    def forward(self, x, ):
        x = self.downsample(x)

        return x


class Downsample_x8(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x8, self).__init__()

        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 8, 8, 0)
        )

    def forward(self, x, ):
        x = self.downsample(x)

        return x


class AMFM_2(nn.Module):
    def __init__(self, inter_dim=512):
        super(AMFM_2, self).__init__()
        self.inter_dim = inter_dim
        self.attention = Attention(self.inter_dim)
        self.fc2 = nn.Conv2d(self.inter_dim, 2, 1, 1, 0)
        self.conv = nn.Conv2d(self.inter_dim, self.inter_dim // 2, 1, 1, 0)
        self.spatial_wise = nn.Sequential(
            #Nx1xHxW
            nn.Conv2d(1, 1, 3, bias=False, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input1, input2):
        x1 = self.conv(input1)
        x2 = self.conv(input2)
        x = torch.cat((x1, x2), dim=1)
        global_x = self.attention(x)
        x_1 = torch.mean(global_x, dim=1, keepdim=True)
        global_x = self.spatial_wise(x_1) + global_x
        global_x = self.fc2(global_x)
        score = F.softmax(global_x, 1)
        out = input1 * score[:, 0:1, :, :] + \
              input2 * score[:, 1:2, :, :]
        return out


class AMFM_3(nn.Module):
    def __init__(self, inter_dim=512):
        super(AMFM_3, self).__init__()

        self.inter_dim = inter_dim
        self.weight_level_1 = Attention(self.inter_dim)
        self.weight_level_2 = Attention(self.inter_dim)
        self.weight_level_3 = Attention(self.inter_dim)

    def forward(self, input1, input2, input3):
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)
        out = level_1_weight_v + level_2_weight_v + level_3_weight_v

        return out

    
class AMFM_4(nn.Module):
    def __init__(self, inter_dim=512):
        super(AMFM_4, self).__init__()
        self.inter_dim = inter_dim
        self.attention = Attention(self.inter_dim)
        self.fc2 = nn.Conv2d(self.inter_dim, 4, 1, 1, 0)
        self.conv = nn.Conv2d(self.inter_dim, self.inter_dim // 4, 1, 1, 0)
        self.spatial_wise = nn.Sequential(
            #Nx1xHxW
            nn.Conv2d(1, 1, 3, bias=False, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input1, input2, input3, input4):
        x1 = self.conv(input1)
        x2 = self.conv(input2)
        x3 = self.conv(input3)
        x4 = self.conv(input4)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        global_x = self.attention(x)
        x_1 = torch.mean(global_x, dim=1, keepdim=True)
        global_x = self.spatial_wise(x_1) + global_x
        global_x = self.fc2(global_x)
        scores = F.softmax(global_x, 1)
        out = input1 * scores[:, 0:1, :, :] + \
              input2 * scores[:, 1:2, :, :] + \
              input3 * scores[:, 2:3, :, :] + \
              input4 * scores[:, 3:4, :, :]
        return out


class BlockBody(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512]):
        super(BlockBody, self).__init__()

        self.FU1 = FFEM(channels[0], channels[0])
        self.FU2 = FFEM(channels[1], channels[1])
        self.FU3 = FFEM(channels[2], channels[2])
        self.FU4 = FFEM(channels[3], channels[3])

        self.downsample_scalezero1_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scaleone1_2 = Downsample_x2(channels[1], channels[2])
        self.downsample_scaletwo1_2 = Downsample_x2(channels[2], channels[3])
        self.upsample_scaleone1_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.upsample_scaletwo1_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.upsample_scalethree1_2 = Upsample(channels[3], channels[2], scale_factor=2)

        self.asff_scalezero1 = AMFM_2(inter_dim=channels[0])
        self.asff_scaleone1 = AMFM_2(inter_dim=channels[1])
        self.asff_scaletwo1 = AMFM_2(inter_dim=channels[2])
        self.asff_scalethree1 = AMFM_2(inter_dim=channels[3])

        self.asff_scalezero2 = AMFM_3(inter_dim=channels[0])
        self.asff_scaleone2 = AMFM_3(inter_dim=channels[1])
        self.asff_scaletwo2 = AMFM_3(inter_dim=channels[2])
        self.asff_scalethree2 = AMFM_3(inter_dim=channels[3])


        self.downsample_scalezero3_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero3_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scalezero3_8 = Downsample_x8(channels[0], channels[3])
        self.upsample_scaleone3_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.downsample_scaleone3_2 = Downsample_x2(channels[1], channels[2])
        self.downsample_scaleone3_4 = Downsample_x4(channels[1], channels[3])
        self.upsample_scaletwo3_4 = Upsample(channels[2], channels[0], scale_factor=4)
        self.upsample_scaletwo3_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.downsample_scaletwo3_2 = Downsample_x2(channels[2], channels[3])
        self.upsample_scalethree3_8 = Upsample(channels[3], channels[0], scale_factor=8)
        self.upsample_scalethree3_4 = Upsample(channels[3], channels[1], scale_factor=4)
        self.upsample_scalethree3_2 = Upsample(channels[3], channels[2], scale_factor=2)

        self.asff_scalezero3 = AMFM_4(inter_dim=channels[0])
        self.asff_scaleone3 = AMFM_4(inter_dim=channels[1])
        self.asff_scaletwo3 = AMFM_4(inter_dim=channels[2])
        self.asff_scalethree3 = AMFM_4(inter_dim=channels[3])


        self.blocks_scalezero4 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0])
        )
        self.blocks_scaleone4 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1])
        )
        self.blocks_scaletwo4 = nn.Sequential(
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2])
        )
        self.blocks_scalethree4 = nn.Sequential(
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3])
        )


    def forward(self, x):
        x0, x1, x2, x3 = x

        x0 = self.FU1(x0) + x0
        x1 = self.FU2(x1) + x1 
        x2 = self.FU3(x2) + x2
        x3 = self.FU4(x3) + x3

        scalezero = self.asff_scalezero1(x0, self.upsample_scaleone1_2(x1))
        scaleone = self.asff_scaleone1(self.downsample_scalezero1_2(x0), x1)

        one1 = self.asff_scaleone1(x1, self.upsample_scaletwo1_2(x2))
        two = self.asff_scaletwo1(self.downsample_scaleone1_2(x1), x2)

        two1 = self.asff_scaletwo1(x2, self.upsample_scalethree1_2(x3))
        three = self.asff_scalethree1(self.downsample_scaletwo1_2(x2), x3)

        scalezero = self.asff_scalezero2(scalezero, self.upsample_scaleone1_2(scaleone), self.upsample_scaletwo3_4(two))
        scaleone = self.asff_scaleone2(scaleone, one1, self.upsample_scaletwo1_2(two))
        scaletwo = self.asff_scaletwo2(self.downsample_scaleone1_2(one1), two, two1)
        scalethree = self.asff_scalethree2(self.downsample_scaleone3_4(one1), self.downsample_scaletwo1_2(two1), three)
        
        scalezero = self.asff_scalezero3(scalezero, self.upsample_scaleone3_2(scaleone), self.upsample_scaletwo3_4(scaletwo), self.upsample_scalethree3_8(scalethree))
        scaleone = self.asff_scaleone3(self.downsample_scalezero3_2(scalezero), scaleone, self.upsample_scaletwo3_2(scaletwo), self.upsample_scalethree3_4(scalethree))
        scaletwo = self.asff_scaletwo3(self.downsample_scalezero3_4(scalezero), self.downsample_scaleone3_2(scaleone), scaletwo, self.upsample_scalethree3_2(scalethree))
        scalethree = self.asff_scalethree3(self.downsample_scalezero3_8(scalezero), self.downsample_scaleone3_4(scaleone), self.downsample_scaletwo3_2(scaletwo), scalethree)


        scalezero = self.blocks_scalezero4(scalezero)
        scaleone = self.blocks_scaleone4(scaleone)
        scaletwo = self.blocks_scaletwo4(scaletwo)
        scalethree = self.blocks_scalethree4(scalethree)


        return scalezero, scaleone, scaletwo, scalethree



@NECKS.register_module()
class AHFFN(nn.Module):
    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 out_channels=256):
        super(AHFFN, self).__init__()

        self.fp16_enabled = False

        self.conv0 = BasicConv(in_channels[0], in_channels[0] // 8, 1)
        self.conv1 = BasicConv(in_channels[1], in_channels[1] // 8, 1)
        self.conv2 = BasicConv(in_channels[2], in_channels[2] // 8, 1)
        self.conv3 = BasicConv(in_channels[3], in_channels[3] // 8, 1)

        self.body = nn.Sequential(
            BlockBody([in_channels[0] // 8, in_channels[1] // 8, in_channels[2] // 8, in_channels[3] // 8])
        )

        self.conv00 = BasicConv(in_channels[0] // 8, out_channels, 1)
        self.conv11 = BasicConv(in_channels[1] // 8, out_channels, 1)
        self.conv22 = BasicConv(in_channels[2] // 8, out_channels, 1)
        self.conv33 = BasicConv(in_channels[3] // 8, out_channels, 1)
        self.conv44 = nn.MaxPool2d(kernel_size=1, stride=2)

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)
    @auto_fp16()
    def forward(self, x):
        x0, x1, x2, x3 = x

        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        out0, out1, out2, out3 = self.body([x0, x1, x2, x3])

        out0 = self.conv00(out0)
        out1 = self.conv11(out1)
        out2 = self.conv22(out2)
        out3 = self.conv33(out3)
        out4 = self.conv44(out3)

        return out0, out1, out2, out3, out4


if __name__ == "__main__":
    print()

