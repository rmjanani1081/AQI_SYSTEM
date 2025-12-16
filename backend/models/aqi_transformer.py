import torch
import torch.nn as nn

class AQITransformer(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x.mean(dim=1))
