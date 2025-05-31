import torch
import torch.nn as nn
import torch.nn.functional as F

class FlappyDQN(nn.Module):
    def __init__(self, input_channels=4, num_actions=2):
        super(FlappyDQN, self).__init__()
        self.num_actions = num_actions
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.iterations = 1000000
        self.memory_size = 50000
        self.batch_size = 64

        self.conv_layer1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv_layer2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv_layer3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        test_input = torch.zeros(1, input_channels, 80, 80)
        test_output = self._forward_conv(test_input)
        conv_output_size = test_output.view(1, -1).size(1)

        self.dense1 = nn.Linear(conv_output_size, 512)
        self.dense2 = nn.Linear(512, num_actions)

        self._initialize_weights()

    def _forward_conv(self, x):
        x = F.relu(self.bn1(self.conv_layer1(x)))
        x = F.relu(self.bn2(self.conv_layer2(x)))
        x = F.relu(self.bn3(self.conv_layer3(x)))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense1(x))
        return self.dense2(x)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)