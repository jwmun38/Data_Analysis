# CNN 모델 정의 (1D Convolution)
import torch.nn as nn
class CNNModel(nn.Module):
    def __init__(self, input_size, num_filters=64, kernel_size=3):
        super(CNNModel, self).__init__()
        # Conv1d 입력: (batch_size, channels, seq_len)
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        conv_output_length = LOOKBACK - kernel_size + 1
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_filters * conv_output_length, 1)
    
    def forward(self, x):
        # x: (batch_size, seq_len, features)
        x = x.permute(0, 2, 1)  # (batch_size, features, seq_len)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x