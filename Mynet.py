import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, ResNet

class PretrainedResNet50(ResNet):
  def __init__(self):
    super().__init__(Bottleneck, [3, 4, 6, 3], 1000)

  def forward(self, x):
    input_dim = x.size()[2:]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x1 = self.maxpool(x)
    x2 = self.layer1(x1)
    x3 = self.layer2(x2)
    x4 = self.layer3(x3)
    x5 = self.layer4(x4)
    return x1, x2, x3, x4, x5, input_dim


class MyNet(nn.Module):
  def __init__(self, num_classes, pretrained_net):
    super().__init__()
    self.pretrained_net = pretrained_net

    score_5 = nn.Conv2d(1024, num_classes , kernel_size=1)
    score_4 = nn.Conv2d(512, num_classes , kernel_size=1)
    score_3 = nn.Conv2d(512, num_classes , kernel_size=1)
    score_2 = nn.Conv2d(256, num_classes , kernel_size=1)
    score_1 = nn.Conv2d(64, num_classes , kernel_size=1)
    self._normal_initialization(score_1)
    self._normal_initialization(score_2)
    self._normal_initialization(score_3)
    self._normal_initialization(score_4)
    self._normal_initialization(score_5)
    self.score_1 = score_1
    self.score_2 = score_2
    self.score_3 = score_3
    self.score_4 = score_4
    self.score_5 = score_5

  def forward(self, x):
    x1, x2, x3, x4, x5, input_dim = self.pretrained_net(x)
    x1 = self.score_1(x1)
    x2 = self.score_2(x2)
    x3 = self.score_3(x3)
    x4 = self.score_4(x4)
    x5 = self.score_5(x5)
    x1_dim = x1.size()[2:]
    x2_dim = x2.size()[2:]
    x3_dim = x3.size()[2:]
    x4_dim = x4.size()[2:]       
    x4 += F.upsample(x5,size=x4_dim, mode='bilinear')
    x3 += F.upsample(x4, size=x3_dim, mode='bilinear')
    x2 += F.upsample(x3, size=x2_dim, mode='bilinear')
    x1 += F.upsample(x2, size=x1_dim, mode='bilinear')
    x = F.upsample(x1, size=input_dim, mode='bilinear')  
    return x

  def _normal_initialization(self, layer):
        
    layer.weight.data.normal_(0, 0.01)
    layer.bias.data.zero_()

