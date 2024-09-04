import numpy as np
import torch
import torch.nn as nn

# from torchinfo import summary

from models.signal_processing import GCC, Window
from models.layers import Mlp


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                        pool_size=None, dropout_rate=0.0):

        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)

        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.PReLU()

        if pool_size is not None:
            self.pool = nn.MaxPool2d(kernel_size=pool_size)
        
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        x = self.activation(self.bn(self.conv(x)))

        if hasattr(self, "pool"):
            x = self.pool(x)

        if hasattr(self, "dropout"):
            x = self.dropout(x)

        return x


class BinauralLocalizer(nn.Module):
    def __init__(self, n_gcc_bins, params):
        """
        Initialize the CRNN model.
        :param n_gcc_bins: Number of the input gcc_bins.
        :param params: Dictionary of parameters.
        :param n_max_sources: Maximum number of simulataneous sources.
        """

        super().__init__()

        self.use_activity_out = params["use_activity_output"]
        self.bidirectional_rnn = params["bidirectional_rnn"]
        
        self.conv_block_list = nn.ModuleList()
        
        self.conv_agg_mode = params["conv_agg_mode"]

        self.n_input_channels = 1 # Correlation values
                                    # Also tried 2 (respective bins)
        self.output_mode = params["output_mode"]

        # Input batch normalization
        self.use_batch_norm_input = params["use_batch_norm_input"]
        if self.use_batch_norm_input:
            self.input_bn = nn.BatchNorm2d(self.n_input_channels)

        # Convolutional blocks
        n_conv_input = self.n_input_channels

        n_conv_layers = len(params["f_pool_size"])
        for conv_cnt in range(n_conv_layers):
            self.conv_block_list.append(
                ConvBlock(
                    in_channels=params["nb_cnn2d_filt"] if conv_cnt else n_conv_input,
                    out_channels=params["nb_cnn2d_filt"],
                    pool_size=(params["t_pool_size"][conv_cnt], params["f_pool_size"][conv_cnt]),
                    dropout_rate=params["dropout_rate"]
                )
            )
        
        self.n_rnn_features = params["rnn_size"]

        if self.conv_agg_mode == "flatten":
            self.in_gru_size = int(
                params["nb_cnn2d_filt"] * (n_gcc_bins / np.prod(params["f_pool_size"])))
        elif self.conv_agg_mode in ["sum", "mean", "prod", "max"]:
            self.in_gru_size = params["nb_cnn2d_filt"]
        
        self.gru = nn.GRU(input_size=self.in_gru_size, hidden_size=self.n_rnn_features,
                                num_layers=params["nb_rnn_layers"], batch_first=True,
                                dropout=params["dropout_rate"],
                                bidirectional=params["bidirectional_rnn"])

        self.n_output = 3 # 3 coordinates (x, y, z) for a single source

        self.fnn_doa = Mlp(
            self.n_rnn_features, self.n_output,
            params["fnn_doa_size"], params["nb_fnn_layers"],
            activation="prelu",
            output_activation=nn.Tanh())

        # summary(self)

    def forward(self, x):
        """Forward pass of the network

        Args:
            x (dict): dictionary containing the "signal"
                key, which contains the input signal, with
                shape (batch_size, n_samples, n_gcc_bins, n_channels).
        Returns:
            torch.Tensor: output of the network
        """

        x = x["signal"]

        batch_size, n_frames, n_features = x.shape
        x = x.unsqueeze(-1) # Add channel dimension

        x = x.moveaxis(3, 1) # Move channel after batch dimension, expected by conv2d
        if self.use_batch_norm_input:
            x = self.input_bn(x)

        # 1. Apply convolutional blocks
        for conv_block in self.conv_block_list:
            x = conv_block(x)
        """(batch_size, feature_maps, time_steps, n_gcc_bins)"""

        x = x.transpose(1, 2).contiguous()

        if self.conv_agg_mode == "flatten":
            x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        elif self.conv_agg_mode == "sum":
            x = x.sum(dim=-1)
        elif self.conv_agg_mode == "mean":
            x = x.mean(dim=-1)
        elif self.conv_agg_mode == "prod":
            x = x.prod(dim=-1)
        elif self.conv_agg_mode == "max":
            x = x.max(dim=-1)[0]
        """ (batch_size, time_steps, feature_maps):"""

        # 2. Apply GRU block
        (x, _) = self.gru(x)
        x = torch.tanh(x)

        if self.bidirectional_rnn:
            x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]
        """(batch_size, time_steps, feature_maps)"""

        # 3. Apply fully connected layers

        # 2. Apply DOA fully connected branch
        doa = self.fnn_doa(x)
        # (batch_size, time_steps, label_dim)

        if self.output_mode == "mean":
            # Average across time frames
            doa = doa.mean(dim=1)
        elif self.output_mode == "last":
            # Take last time frame
            doa = doa[:, -1]

        return {
            "doa_cart": doa
        }


class BinauralFeatureExtractor(torch.nn.Module):
    def __init__(self, params): 
        super().__init__()
        
        self.mic_pair_sampling_mode = params["mic_pair_sampling_mode"]
        self.n_mic_pairs = params["n_mic_pairs"]
        self.metadata_type = params["neural_srp"]["metadata_type"]

        # 1. Create windowing transform
        self.window = Window(
            params["win_size"],
            int(params["win_size"] * params["hop_rate"]),
            window="hann"
        )

        # 2. Create feature extractor
        self.feature_extractor = GCC(params["win_size"], 
                                    tau_max=params["nb_gcc_bins"] // 2,
                                    transform="phat", concat_bins=False,
                                    center=True)

    def forward(self, x):
        x["signal"] = self.window(x["signal"])
        x["signal"] = self.feature_extractor(x["signal"])
        
        return x
