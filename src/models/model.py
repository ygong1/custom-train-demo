import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    """A ResNet block."""

    def __init__(self, f_in: int, f_out: int, downsample: bool = False):
        super(Block, self).__init__()

        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(f_out)
        self.conv2 = nn.Conv2d(f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(f_out)
        self.relu = nn.ReLU(inplace=True)

        # No parameters for shortcut connections.
        if downsample or f_in != f_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(f_out),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x: torch.Tensor):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class ResNetCIFAR(nn.Module):
    """A residual neural network as originally designed for CIFAR-10."""

    def __init__(self, outputs: int = 10):
        super(ResNetCIFAR, self).__init__()

        depth = 56
        width = 16
        num_blocks = (depth - 2) // 6

        plan = [(width, num_blocks), (2 * width, num_blocks), (4 * width, num_blocks)]

        self.num_classes = outputs

        # Initial convolution.
        current_filters = plan[0][0]
        self.conv = nn.Conv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(current_filters)
        self.relu = nn.ReLU(inplace=True)

        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(Block(current_filters, filters, downsample))
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)

        # Final fc layer. Size = number of filters in last segment.
        self.fc = nn.Linear(plan[-1][0], outputs)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor):
        out = self.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
