import torch
import torch.nn as nn
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt

def _get_orthogonal_init_weights(weights):
    fan_out = weights.size(0)
    fan_in = weights.size(1) * weights.size(2) * weights.size(3)

    u, _, v = svd(normal(0.0, 1.0, (fan_out, fan_in)), full_matrices=False)

    if u.shape == (fan_out, fan_in):
        return torch.Tensor(u.reshape(weights.size()))
    else:
        return torch.Tensor(v.reshape(weights.size()))
def pixel_reshuffle(input, upscale_factor):
    r"""Rearranges elements in a tensor of shape ``[*, C, H, W]`` to a
    tensor of shape ``[C*r^2, H/r, W/r]``.

    See :class:`~torch.nn.PixelShuffle` for details.

    Args:
        input (Variable): Input
        upscale_factor (int): factor to increase spatial resolution by

    Examples:
        >>> input = autograd.Variable(torch.Tensor(1, 3, 12, 12))
        >>> output = pixel_reshuffle(input,2)
        >>> print(output.size())
        torch.Size([1, 12, 6, 6])
    """
    batch_size, channels, in_height, in_width = input.size()

    # // division is to keep data type unchanged. In this way, the out_height is still int type
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor
    input_view = input.contiguous().view(batch_size, channels, out_height, upscale_factor, out_width, upscale_factor)
    channels = channels * upscale_factor * upscale_factor

    shuffle_out = input_view.permute(0,1,3,5,2,4).contiguous()
    return shuffle_out.view(batch_size, channels, out_height, out_width)
class FourDilateConvResBlockIN(nn.Module):
    def __init__(self, in_channels, out_channels, dilation2, dilation4):
        super(FourDilateConvResBlockIN, self).__init__()
        self.norm1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.norm2 = nn.InstanceNorm2d(in_channels, affine=True)
        self.norm4 = nn.InstanceNorm2d(in_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels,  out_channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.conv2 = nn.Conv2d(in_channels,  out_channels, (3, 3), (1, 1), (dilation2, dilation2), (dilation2, dilation2), bias=False)
        self.conv4 = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (dilation4, dilation4), (dilation4, dilation4), bias=False)
    def forward(self, x):
        out1 = self.norm1(x)
        out1 = self.relu(out1)
        out1 = self.conv1(out1)

        out2 = self.norm2(x)
        out2 = self.relu(out2)
        out2 = self.conv2(out2)
        out2 = self.norm4(out2)
        out2 = self.relu(out2)
        out2 = self.conv4(out2)
        out  = x + out1 + out2
        return out
    def _initialize_weights(self):
        self.conv1.weight.data.copy_(_get_orthogonal_init_weights(self.conv1.weight))
        self.conv2.weight.data.copy_(_get_orthogonal_init_weights(self.conv2.weight))
        self.conv4.weight.data.copy_(_get_orthogonal_init_weights(self.conv4.weight))

class FourConvResBlockIN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FourConvResBlockIN, self).__init__()
        self.norm1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.norm2 = nn.InstanceNorm2d(in_channels, affine=True)
        self.norm4 = nn.InstanceNorm2d(in_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
    def forward(self, x):
        out1 = self.norm1(x)
        out1 = self.relu(out1)
        out1 = self.conv1(out1)

        out2 = self.norm2(x)
        out2 = self.relu(out2)
        out2 = self.conv2(out2)
        out2 = self.norm4(out2)
        out2 = self.relu(out2)
        out2 = self.conv4(out2)
        out  = x + out1 + out2
        return out
    def _initialize_weights(self):
        self.conv1.weight.data.copy_(_get_orthogonal_init_weights(self.conv1.weight))
        self.conv2.weight.data.copy_(_get_orthogonal_init_weights(self.conv2.weight))
        self.conv4.weight.data.copy_(_get_orthogonal_init_weights(self.conv4.weight))

class TwoConvBlockIN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TwoConvBlockIN, self).__init__()
        self.norm1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.norm2 = nn.InstanceNorm2d(in_channels, affine=True)

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)

    def forward(self, x):
        out1 = self.norm1(x)
        out1 = self.relu(out1)
        out1 = self.conv1(out1)


        out2 = self.norm2(x)
        out2 = self.relu(out2)
        out2 = self.conv2(out2)

        out  = out1 + out2
        return out
    def _initialize_weights(self):
        self.conv1.weight.data.copy_(_get_orthogonal_init_weights(self.conv1.weight))
        self.conv2.weight.data.copy_(_get_orthogonal_init_weights(self.conv2.weight))

class centerEsti(nn.Module):
    def __init__(self):
        super(centerEsti, self).__init__()
        self.conv1_B  = nn.Conv2d( 16,  144, (5, 5), (1, 1), (2, 2), bias=False)
        self.conv2  = nn.Conv2d( 3,  64, (3, 3), (1, 1), (1, 1), bias=False)
        
        self.norm3 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv3  = nn.Conv2d( 64,  3, (3, 3), (1, 1), (1, 1), bias=False)

        self.pixel_shuffle = nn.PixelShuffle(4)
        
        self.LocalGrad1 = self.make_layer(FourConvResBlockIN, 144, 144, 3)
        self.LocalGrad2 = self.make_layer2(144, 144)
        self.LocalGrad3 = self.make_layer3(144, 144)
        self.LocalGrad4 = self.make_layer(FourConvResBlockIN, 144, 144, 3)

        self.fuse1 = self.make_layer(TwoConvBlockIN, 144, 16, 1)
        self.fuse2 = self.make_layer(TwoConvBlockIN, 144, 16, 1)
        self.fuse3 = self.make_layer(TwoConvBlockIN, 144, 16, 1)
        self.fuse4 = self.make_layer(TwoConvBlockIN, 144, 16, 1)

        self.GlobalGrad = self.make_layer(FourConvResBlockIN, 64, 64, 1)
    def forward(self, Blurry):
        Blurry_r = Blurry[:,0,:,:].unsqueeze(1)
        Blurry_g = Blurry[:,1,:,:].unsqueeze(1)
        Blurry_b = Blurry[:,2,:,:].unsqueeze(1)
        Blurry_r_0 = pixel_reshuffle(Blurry_r, 4)
        Blurry_g_0 = pixel_reshuffle(Blurry_g, 4)
        Blurry_b_0 = pixel_reshuffle(Blurry_b, 4)

        x_r_1 = self.conv1_B(Blurry_r_0)
        x_r_1 = self.LocalGrad1(x_r_1)
        x_r_2 = self.LocalGrad2(x_r_1)
        x_r_3 = self.LocalGrad3(x_r_2)
        x_r_4 = self.LocalGrad4(x_r_3)

        x_g_1 = self.conv1_B(Blurry_g_0)
        x_g_1 = self.LocalGrad1(x_g_1)
        x_g_2 = self.LocalGrad2(x_g_1)
        x_g_3 = self.LocalGrad3(x_g_2)
        x_g_4 = self.LocalGrad4(x_g_3)

        x_b_1 = self.conv1_B(Blurry_b_0)
        x_b_1 = self.LocalGrad1(x_b_1)
        x_b_2 = self.LocalGrad2(x_b_1)
        x_b_3 = self.LocalGrad3(x_b_2)
        x_b_4 = self.LocalGrad4(x_b_3)

        x_r_1 = self.fuse1(x_r_1)
        x_r_2 = self.fuse2(x_r_2)
        x_r_3 = self.fuse3(x_r_3)
        x_r_4 = self.fuse4(x_r_4)

        x_g_1 = self.fuse1(x_g_1)
        x_g_2 = self.fuse2(x_g_2)
        x_g_3 = self.fuse3(x_g_3)
        x_g_4 = self.fuse4(x_g_4)

        x_b_1 = self.fuse1(x_b_1)
        x_b_2 = self.fuse2(x_b_2)
        x_b_3 = self.fuse3(x_b_3)
        x_b_4 = self.fuse4(x_b_4)

        # we only estimate the residual respect to sharp reference image.
        x_r_5 = self.pixel_shuffle(x_r_1 + x_r_2 + x_r_3 + x_r_4 + Blurry_r_0) 
        x_g_5 = self.pixel_shuffle(x_g_1 + x_g_2 + x_g_3 + x_g_4 + Blurry_g_0) 
        x_b_5 = self.pixel_shuffle(x_b_1 + x_b_2 + x_b_3 + x_b_4 + Blurry_b_0) 

        out = torch.cat( (x_r_5, x_g_5, x_b_5), 1)
        out = self.conv2(out)
        out = self.GlobalGrad(out)
        out = self.norm3(out)
        out = self.relu(out)
        out = self.conv3(out) + Blurry
        return out
    def make_layer(self, block, in_channels, out_channels, blocks):
        layers = []
        for i in range(1, blocks + 1):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)
    def make_layer2(self, in_channels, out_channels):
        layers = []
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 1, 2))
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 2, 4))
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 4, 8))
        return nn.Sequential(*layers)
    def make_layer3(self, in_channels, out_channels):
        layers = []
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 8, 4))
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 4, 2))
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 2, 1))
        return nn.Sequential(*layers)
class N9_IN(nn.Module):
    def __init__(self):
        super(N9_IN, self).__init__()
        self.conv1_B  = nn.Conv2d(  25,  64, (5, 5), (1, 1), (2, 2), bias=False)
        self.conv1_S1  = nn.Conv2d( 25,  32, (5, 5), (1, 1), (2, 2), bias=False)
        self.conv1_S2  = nn.Conv2d( 25,  32, (5, 5), (1, 1), (2, 2), bias=False)
        self.conv2  = nn.Conv2d( 3,  32, (3, 3), (1, 1), (1, 1), bias=False)
        
        self.norm3 = nn.InstanceNorm2d(32, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv3  = nn.Conv2d( 32,  3, (3, 3), (1, 1), (1, 1), bias=False)

        self.pixel_shuffle = nn.PixelShuffle(5)
        
        self.LocalGrad1 = self.make_layer(FourConvResBlockIN, 128, 128, 2)
        self.LocalGrad2 = self.make_layer2(128, 128)
        self.LocalGrad3 = self.make_layer(FourConvResBlockIN, 128, 128, 2)

        self.fuse1 = self.make_layer(TwoConvBlockIN, 128, 25, 1)
        self.fuse2 = self.make_layer(TwoConvBlockIN, 128, 25, 1)
        self.fuse3 = self.make_layer(TwoConvBlockIN, 128, 25, 1)

        self.GlobalGrad = self.make_layer(TwoConvBlockIN, 32, 32, 1)
    def forward(self, x1_r_0, x1_g_0, x1_b_0, x2_r_0, x2_g_0, x2_b_0, x3_r_0, x3_g_0, x3_b_0):

        x1_r_1 = self.conv1_B(x1_r_0)
        x2_r_1 = self.conv1_S1(x2_r_0)
        x3_r_1 = self.conv1_S2(x3_r_0)
        x_r_1 = torch.cat( (x1_r_1, x2_r_1, x3_r_1), 1 )

        x_r_1 = self.LocalGrad1(x_r_1)
        x_r_2 = self.LocalGrad2(x_r_1)
        x_r_3 = self.LocalGrad3(x_r_2)

        x1_g_1 = self.conv1_B(x1_g_0)
        x2_g_1 = self.conv1_S1(x2_g_0)
        x3_g_1 = self.conv1_S2(x3_g_0)
        x_g_1 = torch.cat( (x1_g_1, x2_g_1, x3_g_1), 1 )

        x_g_1 = self.LocalGrad1(x_g_1)
        x_g_2 = self.LocalGrad2(x_g_1)
        x_g_3 = self.LocalGrad3(x_g_2)

        x1_b_1 = self.conv1_B(x1_b_0)
        x2_b_1 = self.conv1_S1(x2_b_0)
        x3_b_1 = self.conv1_S2(x3_b_0)
        x_b_1 = torch.cat( (x1_b_1, x2_b_1, x3_b_1), 1 )

        x_b_1 = self.LocalGrad1(x_b_1)
        x_b_2 = self.LocalGrad2(x_b_1)
        x_b_3 = self.LocalGrad3(x_b_2)

        x_r_1 = self.fuse1(x_r_1)
        x_r_2 = self.fuse2(x_r_2)
        x_r_3 = self.fuse3(x_r_3)

        x_g_1 = self.fuse1(x_g_1)
        x_g_2 = self.fuse2(x_g_2)
        x_g_3 = self.fuse3(x_g_3)

        x_b_1 = self.fuse1(x_b_1)
        x_b_2 = self.fuse2(x_b_2)
        x_b_3 = self.fuse3(x_b_3)

        # we only estimate the residual respect to sharp reference image.
        x_r_5 = self.pixel_shuffle(x_r_1 + x_r_2 + x_r_3  + x3_r_0) 
        x_g_5 = self.pixel_shuffle(x_g_1 + x_g_2 + x_g_3  + x3_g_0) 
        x_b_5 = self.pixel_shuffle(x_b_1 + x_b_2 + x_b_3  + x3_b_0) 

        out = torch.cat( (x_r_5, x_g_5, x_b_5), 1)

        out = self.conv2(out)
        out = self.GlobalGrad(out)
        out = self.norm3(out)
        out = self.relu(out)
        out = self.conv3(out)
        return out
    def make_layer(self, block, in_channels, out_channels, blocks):
        layers = []
        for i in range(1, blocks + 1):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)
    def make_layer2(self, in_channels, out_channels):
        layers = []
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 1, 2))
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 4, 8))
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 8, 4))
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 2, 1))
        return nn.Sequential(*layers)
class F26_N9(nn.Module):
    def __init__(self):
        super(F26_N9, self).__init__()
        self.generateFrame2_6 =  N9_IN()
    def forward(self, Blurry, ref3, ref4, ref5):

        Blurry_r = Blurry[:,0,:,:].unsqueeze(1)
        Blurry_g = Blurry[:,1,:,:].unsqueeze(1)
        Blurry_b = Blurry[:,2,:,:].unsqueeze(1)
        Blurry_r_0 = pixel_reshuffle(Blurry_r, 5)
        Blurry_g_0 = pixel_reshuffle(Blurry_g, 5)
        Blurry_b_0 = pixel_reshuffle(Blurry_b, 5)

        ref3_r = ref3[:,0,:,:].unsqueeze(1)
        ref3_g = ref3[:,1,:,:].unsqueeze(1)
        ref3_b = ref3[:,2,:,:].unsqueeze(1)
        ref3_r_0 = pixel_reshuffle(ref3_r, 5)
        ref3_g_0 = pixel_reshuffle(ref3_g, 5)
        ref3_b_0 = pixel_reshuffle(ref3_b, 5)

        ref4_r = ref4[:,0,:,:].unsqueeze(1)
        ref4_g = ref4[:,1,:,:].unsqueeze(1)
        ref4_b = ref4[:,2,:,:].unsqueeze(1)
        ref4_r_0 = pixel_reshuffle(ref4_r, 5)
        ref4_g_0 = pixel_reshuffle(ref4_g, 5)
        ref4_b_0 = pixel_reshuffle(ref4_b, 5)

        ref5_r = ref5[:,0,:,:].unsqueeze(1)
        ref5_g = ref5[:,1,:,:].unsqueeze(1)
        ref5_b = ref5[:,2,:,:].unsqueeze(1)
        ref5_r_0 = pixel_reshuffle(ref5_r, 5)
        ref5_g_0 = pixel_reshuffle(ref5_g, 5)
        ref5_b_0 = pixel_reshuffle(ref5_b, 5)

        ref2 = self.generateFrame2_6(Blurry_r_0, Blurry_g_0, Blurry_b_0, ref4_r_0, ref4_g_0, ref4_b_0, ref3_r_0, ref3_g_0, ref3_b_0) + ref3
        ref6 = self.generateFrame2_6(Blurry_r_0, Blurry_g_0, Blurry_b_0, ref4_r_0, ref4_g_0, ref4_b_0, ref5_r_0, ref5_g_0, ref5_b_0) + ref5
        return ref2, ref6
class F17_N9(nn.Module):
    def __init__(self):
        super(F17_N9, self).__init__()
        self.generateFrame1_7 =  N9_IN()
    def forward(self, Blurry, ref2, ref3, ref5, ref6):

        Blurry_r = Blurry[:,0,:,:].unsqueeze(1)
        Blurry_g = Blurry[:,1,:,:].unsqueeze(1)
        Blurry_b = Blurry[:,2,:,:].unsqueeze(1)
        Blurry_r_0 = pixel_reshuffle(Blurry_r, 5)
        Blurry_g_0 = pixel_reshuffle(Blurry_g, 5)
        Blurry_b_0 = pixel_reshuffle(Blurry_b, 5)

        ref3_r = ref3[:,0,:,:].unsqueeze(1)
        ref3_g = ref3[:,1,:,:].unsqueeze(1)
        ref3_b = ref3[:,2,:,:].unsqueeze(1)
        ref3_r_0 = pixel_reshuffle(ref3_r, 5)
        ref3_g_0 = pixel_reshuffle(ref3_g, 5)
        ref3_b_0 = pixel_reshuffle(ref3_b, 5)

        ref2_r = ref2[:,0,:,:].unsqueeze(1)
        ref2_g = ref2[:,1,:,:].unsqueeze(1)
        ref2_b = ref2[:,2,:,:].unsqueeze(1)
        ref2_r_0 = pixel_reshuffle(ref2_r, 5)
        ref2_g_0 = pixel_reshuffle(ref2_g, 5)
        ref2_b_0 = pixel_reshuffle(ref2_b, 5)

        ref5_r = ref5[:,0,:,:].unsqueeze(1)
        ref5_g = ref5[:,1,:,:].unsqueeze(1)
        ref5_b = ref5[:,2,:,:].unsqueeze(1)
        ref5_r_0 = pixel_reshuffle(ref5_r, 5)
        ref5_g_0 = pixel_reshuffle(ref5_g, 5)
        ref5_b_0 = pixel_reshuffle(ref5_b, 5)

        ref6_r = ref6[:,0,:,:].unsqueeze(1)
        ref6_g = ref6[:,1,:,:].unsqueeze(1)
        ref6_b = ref6[:,2,:,:].unsqueeze(1)
        ref6_r_0 = pixel_reshuffle(ref6_r, 5)
        ref6_g_0 = pixel_reshuffle(ref6_g, 5)
        ref6_b_0 = pixel_reshuffle(ref6_b, 5)

        ref1 = self.generateFrame1_7(Blurry_r_0, Blurry_g_0, Blurry_b_0, ref3_r_0, ref3_g_0, ref3_b_0, ref2_r_0, ref2_g_0, ref2_b_0) + ref2
        ref7 = self.generateFrame1_7(Blurry_r_0, Blurry_g_0, Blurry_b_0, ref5_r_0, ref5_g_0, ref5_b_0, ref6_r_0, ref6_g_0, ref6_b_0) + ref6
        return ref1, ref7
class N8_IN(nn.Module):
    def __init__(self):
        super(N8_IN, self).__init__()
        self.conv1_B  = nn.Conv2d(  25,  64, (5, 5), (1, 1), (2, 2), bias=False)
        self.conv1_S1  = nn.Conv2d( 25,  32, (5, 5), (1, 1), (2, 2), bias=False)
        self.conv1_S2  = nn.Conv2d( 25,  32, (5, 5), (1, 1), (2, 2), bias=False)
        self.conv2  = nn.Conv2d( 3,  32, (3, 3), (1, 1), (1, 1), bias=False)
        
        self.norm3 = nn.InstanceNorm2d(32, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv3  = nn.Conv2d( 32,  3, (3, 3), (1, 1), (1, 1), bias=False)

        self.pixel_shuffle = nn.PixelShuffle(5)
        
        self.LocalGrad1 = self.make_layer(FourConvResBlockIN, 128, 128, 4)
        self.LocalGrad2 = self.make_layer2(128, 128)
        self.LocalGrad3 = self.make_layer(FourConvResBlockIN, 128, 128, 4)

        self.fuse1 = self.make_layer(TwoConvBlockIN, 128, 25, 1)
        self.fuse2 = self.make_layer(TwoConvBlockIN, 128, 25, 1)
        self.fuse3 = self.make_layer(TwoConvBlockIN, 128, 25, 1)

        self.GlobalGrad = self.make_layer(TwoConvBlockIN, 32, 32, 1)
    def forward(self, x1_r_0, x1_g_0, x1_b_0, x2_r_0, x2_g_0, x2_b_0, x3_r_0, x3_g_0, x3_b_0):

        x1_r_1 = self.conv1_B(x1_r_0)
        x2_r_1 = self.conv1_S1(x2_r_0)
        x3_r_1 = self.conv1_S2(x3_r_0)
        x_r_1 = torch.cat( (x1_r_1, x2_r_1, x3_r_1), 1 )

        x_r_1 = self.LocalGrad1(x_r_1)
        x_r_2 = self.LocalGrad2(x_r_1)
        x_r_3 = self.LocalGrad3(x_r_2)

        x1_g_1 = self.conv1_B(x1_g_0)
        x2_g_1 = self.conv1_S1(x2_g_0)
        x3_g_1 = self.conv1_S2(x3_g_0)
        x_g_1 = torch.cat( (x1_g_1, x2_g_1, x3_g_1), 1 )

        x_g_1 = self.LocalGrad1(x_g_1)
        x_g_2 = self.LocalGrad2(x_g_1)
        x_g_3 = self.LocalGrad3(x_g_2)

        x1_b_1 = self.conv1_B(x1_b_0)
        x2_b_1 = self.conv1_S1(x2_b_0)
        x3_b_1 = self.conv1_S2(x3_b_0)
        x_b_1 = torch.cat( (x1_b_1, x2_b_1, x3_b_1), 1 )

        x_b_1 = self.LocalGrad1(x_b_1)
        x_b_2 = self.LocalGrad2(x_b_1)
        x_b_3 = self.LocalGrad3(x_b_2)

        x_r_1 = self.fuse1(x_r_1)
        x_r_2 = self.fuse2(x_r_2)
        x_r_3 = self.fuse3(x_r_3)

        x_g_1 = self.fuse1(x_g_1)
        x_g_2 = self.fuse2(x_g_2)
        x_g_3 = self.fuse3(x_g_3)

        x_b_1 = self.fuse1(x_b_1)
        x_b_2 = self.fuse2(x_b_2)
        x_b_3 = self.fuse3(x_b_3)

        # we only estimate the residual respect to sharp reference image.
        x_r_5 = self.pixel_shuffle(x_r_1 + x_r_2 + x_r_3  + x3_r_0) 
        x_g_5 = self.pixel_shuffle(x_g_1 + x_g_2 + x_g_3  + x3_g_0) 
        x_b_5 = self.pixel_shuffle(x_b_1 + x_b_2 + x_b_3  + x3_b_0) 

        out = torch.cat( (x_r_5, x_g_5, x_b_5), 1)

        out = self.conv2(out)
        out = self.GlobalGrad(out)
        out = self.norm3(out)
        out = self.relu(out)
        out = self.conv3(out)
        return out
    def make_layer(self, block, in_channels, out_channels, blocks):
        layers = []
        for i in range(1, blocks + 1):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)
    def make_layer2(self, in_channels, out_channels):
        layers = []
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 1, 2))
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 4, 8))
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 8, 4))
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 2, 1))
        return nn.Sequential(*layers)
class F35_N8(nn.Module):
    def __init__(self):
        super(F35_N8, self).__init__()
        self.generateFrame3 =  N8_IN()
        self.generateFrame5 =  N8_IN()
    def forward(self, Blurry, ref4):

        Blurry_r = Blurry[:,0,:,:].unsqueeze(1)
        Blurry_g = Blurry[:,1,:,:].unsqueeze(1)
        Blurry_b = Blurry[:,2,:,:].unsqueeze(1)
        Blurry_r_0 = pixel_reshuffle(Blurry_r, 5)
        Blurry_g_0 = pixel_reshuffle(Blurry_g, 5)
        Blurry_b_0 = pixel_reshuffle(Blurry_b, 5)

        ref4_r = ref4[:,0,:,:].unsqueeze(1)
        ref4_g = ref4[:,1,:,:].unsqueeze(1)
        ref4_b = ref4[:,2,:,:].unsqueeze(1)
        ref4_r_0 = pixel_reshuffle(ref4_r, 5)
        ref4_g_0 = pixel_reshuffle(ref4_g, 5)
        ref4_b_0 = pixel_reshuffle(ref4_b, 5)

        ref3 = self.generateFrame3(Blurry_r_0, Blurry_g_0, Blurry_b_0, ref4_r_0, ref4_g_0, ref4_b_0, ref4_r_0, ref4_g_0, ref4_b_0) + ref4

        ref3_r = ref3[:,0,:,:].unsqueeze(1)
        ref3_g = ref3[:,1,:,:].unsqueeze(1)
        ref3_b = ref3[:,2,:,:].unsqueeze(1)
        ref3_r_0 = pixel_reshuffle(ref3_r, 5)
        ref3_g_0 = pixel_reshuffle(ref3_g, 5)
        ref3_b_0 = pixel_reshuffle(ref3_b, 5)

        ref5 = self.generateFrame5(Blurry_r_0, Blurry_g_0, Blurry_b_0, ref3_r_0, ref3_g_0, ref3_b_0, ref4_r_0, ref4_g_0, ref4_b_0) + ref4
        return ref3, ref5