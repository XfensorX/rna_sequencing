import torch
import pytorch_lightning as pl


class AdaptableDenseModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_layer_sizes, output_dim):
        super().__init__()
        layers = []

        current_dim = input_dim
        for hidden_size in hidden_layer_sizes:
            layers.append(torch.nn.Linear(current_dim, hidden_size))
            layers.append(torch.nn.ReLU())
            current_dim = hidden_size

        layers.append(torch.nn.Linear(current_dim, output_dim))

        self.layers = torch.nn.Sequential(*layers)

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)

        self.log("training/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("validation/loss", loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
