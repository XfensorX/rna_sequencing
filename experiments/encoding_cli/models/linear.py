import torch
import pytorch_lightning as pl

from utils.metrics import Metrics
import numpy.typing as npt


class AdaptableDenseModel(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        hidden_layer_sizes,
        output_dim,
        use_dropout: bool = False,
        dropout_prob: float = 0.0,
        use_layer_norm: bool = False,
        use_batch_norm: bool = False,
        learning_rate: float = 0.001,
        activation_function="relu",
    ):
        super().__init__()

        self.learning_rate = learning_rate
        layers = []

        current_dim = input_dim
        for hidden_size in hidden_layer_sizes:
            layers.append(torch.nn.Linear(current_dim, hidden_size))
            if use_dropout:
                layers.append(torch.nn.Dropout(p=dropout_prob))

            if activation_function.lower() == "relu":
                layers.append(torch.nn.ReLU())
            elif activation_function.lower() == "sigmoid":
                layers.append(torch.nn.Sigmoid())
            elif activation_function.lower() == "tanh":
                layers.append(torch.nn.Tanh())

            elif "leaky relu" in activation_function.lower():
                layers.append(
                    torch.nn.LeakyReLU(float(activation_function.split(" ")[-1]))
                )

            if use_layer_norm:
                layers.append(torch.nn.LayerNorm(hidden_size))

            if use_batch_norm:
                layers.append(torch.nn.BatchNorm1d(hidden_size))
            current_dim = hidden_size

        layers.append(torch.nn.Linear(current_dim, output_dim))

        self.layers = torch.nn.Sequential(*layers)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.validation_step_outputs = []

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        logits: npt.NDArray = (
            torch.cat([x["logits"] for x in self.validation_step_outputs])
            .cpu()
            .sigmoid()
            .round()
            .numpy()
        )
        targets: npt.NDArray = (
            torch.cat([x["targets"] for x in self.validation_step_outputs])
            .cpu()
            .numpy()
        )

        metrics = Metrics(logits, targets)
        self.log_dict({f"validation/metrics/{k}": v for k, v in metrics})

        self.validation_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log("val_loss", self.criterion(logits, y), prog_bar=True)
        self.validation_step_outputs.append({"logits": logits, "targets": y})
        return logits

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
