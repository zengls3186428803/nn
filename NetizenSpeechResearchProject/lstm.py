import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, class_num, dropout=0.5):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.class_num = class_num
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, device=device,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(in_features=hidden_size * 2, out_features=hidden_size * 2, device=device),
            nn.Tanh(),
            nn.Linear(in_features=hidden_size * 2, out_features=hidden_size * 2, device=device),
            nn.Tanh(),
            nn.Linear(in_features=hidden_size * 2, out_features=class_num, device=device)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        logits = self.fc(out[:, -1, :])
        return logits
