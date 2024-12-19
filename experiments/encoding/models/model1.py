import pytorch_lightning as pl
import torch


class Model1(pl.LightningModule):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss
