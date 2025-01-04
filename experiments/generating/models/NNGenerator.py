import torch


class AdaptableDenseModel(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_layer_sizes,
        dropout_prob,
        leaky_relu_slope,
        output_dim,
        use_layer_norm: bool = False,
        use_batch_norm: bool = False,
    ):
        super().__init__()
        layers = []

        current_dim = input_dim
        for i, hidden_size in enumerate(hidden_layer_sizes):
            layers.append(torch.nn.Linear(current_dim, hidden_size))
            if leaky_relu_slope is not None:
                layers.append(torch.nn.LeakyReLU(leaky_relu_slope))
            else:
                layers.append(torch.nn.ReLU())
            if use_layer_norm:
                layers.append(torch.nn.LayerNorm(hidden_size))
            if use_batch_norm:
                layers.append(torch.nn.BatchNorm1d(hidden_size))
            current_dim = hidden_size

            if dropout_prob is not None:
                layers.append(torch.nn.Dropout(p=dropout_prob))

        layers.append(torch.nn.Linear(current_dim, output_dim))

        self.layers = torch.nn.Sequential(*layers)

        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        return self.layers(x)
