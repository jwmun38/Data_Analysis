
import torch.nn as nn# GRU 모델 정의
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=50):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out