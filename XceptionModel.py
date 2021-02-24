from collections import OrderedDict
from typing import Any

import torch.nn as nn
import torch

file_path = ''


def xception(pretrained=False, **kwargs):
    model = XCeption(**kwargs)

    model = Xception(**kwargs)
    if pretrained:
        model.load_state_dict(file_path)
    return model


class XCeption(nn.Module):
    r"""XCeption Model Class, based on
    `"Xception: Deep Learning with Depthwise Separable Convolutions"` <https://arxiv.org/pdf/1610.02357v3.pdf>
    Args:
        classCount (int) - Num of classes to classify
        isTrained (bool) - If true, return an existing model with loaded weights
    """

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, classCount):
        super(XCeption, self).__init__()

        kernelSize = (3, 3)
        strideSize = (2, 2)

        self.features = nn.Sequential(OrderedDict([
            ('entry_flow', XCeptionEntryFlow(32, 728, kernelSize, strideSize)),
            ('middle_flow', XCeptionMiddleFlow(8, 728, kernelSize)),
            ('exit_flow', XCeptionExitFlow(728, classCount, kernelSize, strideSize))
        ]))

    def forward(self, input):
        x = self.features(input)
        return x


class XCeptionEntryFlow(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, inputSize, outputSize, kernelSize, strideSize):
        super(XCeptionEntryFlow, self).__init__()

        initial_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=3, out_channels=inputSize, kernel_size=kernelSize, stride=strideSize)),
            ('norm0', nn.BatchNorm2d(num_features=inputSize)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(in_channels=inputSize, out_channels=64, kernel_size=kernelSize)),
            ('norm1', nn.BatchNorm2d(num_features=64)),
            ('relu1', nn.ReLU(inplace=True))
        ]))

        self.features = nn.Sequential(OrderedDict([
            ('initial_block', initial_block),
            ('entry_block0', XCeptionEntryFlowBlock(64, 128, kernelSize, strideSize, False)),
            ('entry_block1', XCeptionEntryFlowBlock(128, 256, kernelSize, strideSize, True)),
            ('entry_block2', XCeptionEntryFlowBlock(256, outputSize, kernelSize, strideSize, True))
        ]))

    def forward(self, input):
        return self.features(input)


class XCeptionEntryFlowBlock(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, inputSize, outputSize, kernelSize, strideSize, includeRelu):
        super(XCeptionEntryFlowBlock, self).__init__()

        self.block0 = nn.Sequential(OrderedDict([]))

        if includeRelu:
            self.block0.add_module('relu0', nn.ReLU(inplace=True))

        self.block0.add_module('sep_conv0', SeparableConv2d(inputSize, outputSize, kernelSize))
        self.block0.add_module('norm0', nn.BatchNorm2d(num_features=outputSize))

        self.block1 = nn.Sequential(OrderedDict([
            ('relu1', nn.ReLU(inplace=True)),
            ('sep_conv1', SeparableConv2d(outputSize, outputSize, kernelSize)),
            ('norm1', nn.BatchNorm2d(num_features=outputSize))
        ]))

        self.max_pool = nn.MaxPool2d(kernel_size=kernelSize, stride=strideSize)

        self.res_block = nn.Sequential(OrderedDict([
            ('res_conv',
             nn.Conv2d(in_channels=inputSize, out_channels=outputSize, kernel_size=(1, 1), stride=strideSize)),
            ('res_norm', nn.BatchNorm2d(num_features=outputSize))
        ]))

    def forward(self, input):
        residual = self.res_block(input)
        x = self.block0(input)
        x = self.block1(x)
        x = self.max_pool(x)

        x = torch.cat((x, residual), dim=1)

        return x


class XCeptionMiddleFlow(nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, num_blocks, size, kernel):
        super(XCeptionMiddleFlow, self).__init__()
        self.features = nn.Sequential(OrderedDict([]))

        for i in range(num_blocks):
            block = nn.Sequential(OrderedDict([
                ('relu', nn.ReLU(inplace=True)),
                ('sep_conv', SeparableConv2d(size, size, kernel)),
                ('norm', nn.BatchNorm2d(num_features=size))
            ]))

            self.features.add_module(f"block{i}", block)

    def forward(self, input):
        x = self.features(input)
        return torch.cat((x, input), dim=1)


class XCeptionExitFlow(nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, inputSize, numClasses, kernelSize, strideSize):
        super(XCeptionExitFlow, self).__init__()

        self.block = nn.Sequential(OrderedDict([
            ('relu0', nn.ReLU(inplace=True)),
            ('sep_conv0', SeparableConv2d(inputSize, inputSize, kernelSize)),
            ('norm0', nn.BatchNorm2d(num_features=inputSize)),
            ('relu1', nn.ReLU(inplace=True)),
            ('sep_conv1', SeparableConv2d(inputSize, 1024, kernelSize)),
            ('norm1', nn.BatchNorm2d(num_features=1024)),
            ('max_pool', nn.MaxPool2d(kernel_size=kernelSize, stride=strideSize))
        ]))
        self.residual = nn.Sequential(OrderedDict([
            ('res_conv',
             nn.Conv2d(in_channels=inputSize, out_channels=1024, kernel_size=(1, 1), stride=strideSize)),
            ('res_norm', nn.BatchNorm2d(num_features=1024))
        ]))

        self.exit_block = nn.Sequential(OrderedDict([
            ('sep_conv0', SeparableConv2d(1024, 1536, kernelSize)),
            ('norm0', nn.BatchNorm2d(num_features=1536)),
            ('relu0', nn.ReLU(inplace=True)),
            ('sep_conv1', SeparableConv2d(1536, 2048, kernelSize)),
            ('norm0', nn.BatchNorm2d(num_features=2048)),
            ('relu1', nn.ReLU(inplace=True)),
            ('global_avg_pooling', nn.AdaptiveAvgPool2d(1))
        ]))

        self.classifier = nn.Sequential(
            nn.Linear(2048, numClasses),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.block(input)
        residual = self.residual(input)
        x = torch.cat((x, residual), dim=1)
        x = self.exit_block(x)
        x = self.classifier(x)

        return x


class SeparableConv2d(nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, inputSize, outputSize, kernelSize, bias=False):
        self.in_channels = inputSize
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(inputSize, inputSize, kernel_size=kernelSize,
                                   groups=inputSize, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(inputSize, outputSize,
                                   kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
