import torch.nn as nn
from torch.nn import functional as F
from ASPP import ASPP
import torch
from torch import Tensor
from collections import OrderedDict


class ConradDenseAspp(nn.Module):
    """
    Modified model based on a combination of Densenet blocks and ASPP convolution blocks
    """

    def __init__(self, growth_rate=32, num_init_features=64, bn_size=4,
                 drop_rate=0, num_classes=14, memory_efficient=False, include_top=False):
        super(ConradDenseAspp, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features

        block1 = MyDenseBlock(
            num_layers=6,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate,
            memory_efficient=memory_efficient
        )
        self.features.add_module('MyDenseBlock%d' % 1, block1)
        num_features = num_features + 6 * growth_rate

        transition1 = MyTransition(
            num_input_features=num_features,
            num_output_features=num_features // 2
        )
        self.features.add_module('MyTransition%d' % 1, transition1)
        num_features = num_features // 2

        block2 = MyDenseBlock(
            num_layers=12,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate,
            memory_efficient=memory_efficient
        )

        self.features.add_module('MyDenseBlock%d' % 2, block2)
        num_features = num_features + 12 * growth_rate

        transition2 = MyTransition(
            num_input_features=num_features,
            num_output_features=num_features // 2
        )
        self.features.add_module('MyTransition%d' % 2, transition2)
        num_features = num_features // 2

        # block3 = MyDenseBlock(
        #     num_layers=24,
        #     num_input_features=num_features,
        #     bn_size=bn_size,
        #     growth_rate=growth_rate,
        #     drop_rate=drop_rate,
        #     memory_efficient=memory_efficient
        # )
        #
        # self.features.add_module('MyDenseBlock%d' % 3, block3)
        # num_features = num_features + 24 * growth_rate

        atrous_rates = [1,3,6]

        aspp_sequential = nn.Sequential(
            ASPP(in_channels=num_features, out_channels=num_features * 4, atrous_rates=atrous_rates)
        )

        self.features.add_module('MyASPPModule%d' % 1, aspp_sequential)
        num_features = num_features * 4

        transition3 = MyTransition(
            num_input_features=num_features,
            num_output_features=num_features // 2
        )
        self.features.add_module('MyTransition%d' % 3, transition3)
        num_features = num_features // 2

        self.features.add_module('MyPool%d' % 1, nn.AvgPool2d(kernel_size=1, stride=1))
        self.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Softmax(dim=-1) if include_top else nn.Sigmoid()
        )
        # if include_top:
        #     self.classifier = nn.Sequential(
        #         nn.Linear(num_features, num_classes),
        #         nn.Softmax(dim=-1)
        #         # nn.Sigmoid()
        #     )
        # else:
        #     self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class MyDenseBlock(nn.ModuleDict):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(MyDenseBlock, self).__init__()
        for i in range(num_layers):
            layer = MyDenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('MyDenseLayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class MyTransition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(MyTransition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class MyDenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(MyDenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features

if __name__ == '__main__':
    from torchsummary import summary
    dummy = torch.ones(1,3,224,224).cuda()
    model = ConradDenseAspp(num_classes=14, include_top=True).eval()
    print(summary(model, input_data=(3, 224,224)))
    out = model.forward(dummy)
    print(out)