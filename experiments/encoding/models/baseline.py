import pytorch_lightning as pl
import torch


class Constant(pl.LightningModule):
    def __init__(self, output_dim: int, constant_value: float):
        super().__init__()
        self.output_dim = output_dim
        self.constant_value = constant_value

        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.fill(torch.zeros(x.shape[0], self.output_dim), self.constant_value)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def training_step(self, batch, batch_idx):
        return self.dummy_param
