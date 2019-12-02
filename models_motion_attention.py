import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import math
from functools import partial 
import torchvision
from opts import parse_opts_offline

opt = parse_opts_offline()

device = torch.device(opt.torch_device)


class LSTM(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_dim, bidirectional):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self, x):
        x, self.hidden_state = self.lstm(x, self.hidden_state)
        return x



class ConvLSTM(nn.Module):
    def __init__(
        self, num_classes, latent_dim=128, lstm_layers=1, hidden_dim=256, bidirectional=True, attention=True
    ):
        super(ConvLSTM, self).__init__()
        # get pretrained model
        self.encoder = torchvision.models.resnet18(pretrained=True)
        for param in self.encoder.parameters():
            param.requires_grad = False
        num_ftrs = self.encoder.fc.in_features
        self.encoder.fc = nn.Linear(num_ftrs, latent_dim)
        self.lstm = LSTM(latent_dim, lstm_layers, hidden_dim, bidirectional)
        self.output_layers = nn.Sequential(
            nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=-1),
        )
        self.attention = attention
        self.latent_dim = latent_dim
    def forward(self, x, attention_w):
        batch_size, c, seq_length, h, w = x.shape
        x = x.permute(0,2,1,3,4)
        input = torch.zeros(batch_size, seq_length, self.latent_dim).to(device)
        for i_data in range(batch_size):
            input[i_data] = self.encoder(x[i_data])
        x = self.lstm(input)
        x = torch.sum(attention_w.unsqueeze(-1) * x, dim=1)
        return self.output_layers(x)


