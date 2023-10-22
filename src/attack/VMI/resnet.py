import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import Flatten
from .conv import ConvBlock

# Import the FiLM generator networks directly from the original CNAPs repo.
import cnaps.src.adaptation_networks as adaptors 


class SimpleResidualBlock(nn.Module):
    def __init__(self, indim, outdim, half_res):
        super(SimpleResidualBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(outdim)
        self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
        self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
            self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'


    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out


class FiLMSimpleResidualBlock(SimpleResidualBlock):
    def __init__(self, indim, outdim, half_res):
        super(FiLMSimpleResidualBlock, self).__init__(indim, outdim, half_res)

    def forward(self, x, gamma1=None, beta1=None, gamma2=None, beta2=None):
        is_none = [param is None for param in [gamma1, beta1, gamma2, beta2]]
        if any(is_none):
            if not all(is_none):
                raise ValueError('Expected either all or none of the FiLM '
                  'parameters to be None.')
            # Use the parent's forward pass without any FiLM params.
            return super(FiLMSimpleResidualBlock, self).forward(x)

        # Use the provided FiLM params to modify the forward pass.
        out = self.C1(x)
        out = self.BN1(out)
        out = self._film(out, gamma1, beta1)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self._film(out, gamma2, beta2)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out

    def _film(x, gamma, beta):
      gamma = gamma[None, :, None, None]
      beta = beta[None, :, None, None]
      return gamma * x + beta


class ResNet(nn.Module):
    def __init__(self,block,list_of_num_layers, list_of_out_dims, flatten = True,final_feature_map_width=7):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet,self).__init__()
        assert len(list_of_num_layers)==4, 'Can have only four stages'
        self._list_of_num_layers = list_of_num_layers
        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                            bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):
            for j in range(list_of_num_layers[i]):
                half_res = (i>=1) and (j==0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(final_feature_map_width)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [ indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)

    def forward(self,x):
        out = self.trunk(x)
        return out


class FiLMResNet(ResNet):
    def __init__(self, block, list_of_num_layers, list_of_out_dims, 
        flatten=True, final_feature_map_width=7):
        if block != FiLMSimpleResidualBlock:
            raise ValueError('Expected block to be FiLMSimpleResidualBlock '
              'when using FiLMResNet.')
        super(FiLMResNet, self).__init__(block, list_of_num_layers, 
            list_of_out_dims, flatten=flatten, 
            final_feature_map_width=final_feature_map_width)

    def forward(self, x, param_dict=None):
        """
        :param param_dict: (list::dict::torch.tensor) A dict per block in each
        layer containing the FiLM adaptation params for each conv layer.
        """
        # Initial layer: conv, bn, relu and pool.
        # maybe need another nn.sequential or something here?
        #out = self.trunk[:4](x)
        out = x
        for i in range(4):
          out = self.trunk[i](out)
        
        offset = 4  # offset into self.trunk.
        # list_of_num_layers is e.g. [2,2,2,2].
        for layer_idx, num_layers in enumerate(self._list_of_num_layers):
            for block_idx in range(num_layers):
                block = self.trunk[offset]
                offset += 1

                block_args = [out]
                # Even when using this architecture there is an option to not
                # provide FiLM params, thus not modifying the backbone.
                if param_dict is not None:
                    block_args.extend([
                      param_dict[layer_idx][block_idx]['gamma1'], 
                      param_dict[layer_idx][block_idx]['beta1'], 
                      param_dict[layer_idx][block_idx]['gamma2'], 
                      param_dict[layer_idx][block_idx]['beta2'] 
                    ])

                out = block(*block_args) 

        # Finish the foward pass (final average pooling if applicable).
        out = self.trunk[offset:](out)
        return out


def SmallResNet10( flatten = True):
    # works for im size 64x64
    return ResNet(SimpleResidualBlock, [1,1,1,1],[64,128,256,512], flatten,final_feature_map_width=2)

def SmallResNet18( flatten = True):
    # works for im size 64x64
    return ResNet(SimpleResidualBlock, [1,1,1,1],[64,128,256,512], flatten,final_feature_map_width=2)

def ResNet10( flatten = True):
    return ResNet(SimpleResidualBlock, [1,1,1,1],[64,128,256,512], flatten)

def ResNet18( flatten = True):
    return ResNet(SimpleResidualBlock, [2,2,2,2],[64,128,256,512], flatten)

# Architectures with FiLM.
def define_backbone_and_adaptors(num_blocks_per_layer, num_maps_per_layer, 
    final_feature_map_width, flatten=True): 
    backbone = FiLMResNet(
        FiLMSimpleResidualBlock, 
        num_blocks_per_layer, 
        num_maps_per_layer, 
        flatten,
        final_feature_map_width=final_feature_map_width)
    backbone_adaptor = adaptors.FilmAdaptationNetwork(
          layer=adaptors.FilmLayerNetwork,
          num_maps_per_layer=num_maps_per_layer,
          num_blocks_per_layer=num_blocks_per_layer,
          # The size of the task representation, which I'm currently assuming is computed in terms of averages in the emebdding space.
          z_g_dim=512)
          # Similarly, the classifier adaptor is conditioned on the class prototypes..
    classifier_adaptor = adaptors.LinearClassifierAdaptationNetwork(512)
    return backbone, backbone_adaptor, classifier_adaptor

def FiLMResNet10(flatten = True):
    return define_backbone_and_adaptors(
        [1,1,1,1], 
        [64,128,256,512],
        final_feature_map_width=2,
        flatten=flatten)

def FiLMResNet18(flatten = True):
    return define_backbone_and_adaptors(
        [2,2,2,2], 
        [64,128,256,512], 
        final_feature_map_width=2,
        flatten=flatten)
