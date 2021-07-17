import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: Document
# NOTE: From https://github.com/pytorch/pytorch/issues/1333#issuecomment-400338207
class CausalConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True
    ):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


# TODO: Document
class DenseBlock(nn.Module):
    def __init__(self, in_channels, dilation, filters, kernel_size=2):
        super(DenseBlock, self).__init__()
        self.xf_causal_conv = CausalConv1d(in_channels, filters, kernel_size, dilation=dilation)
        self.xg_causal_conv = CausalConv1d(in_channels, filters, kernel_size, dilation=dilation)

    def forward(self, input):
        # input is dimensions (N, in_channels, T)
        xf = self.xf_causal_conv(input)
        xg = self.xg_causal_conv(input)
        activations = torch.tanh(xf) * torch.sigmoid(xg) # shape: (N, filters, T)

        return torch.cat((input, activations), dim=1)


# TODO: Document
class TCBlock(nn.Module):
    def __init__(self, in_channels, seq_length, filters):
        super(TCBlock, self).__init__()
        dense_block_count = int(np.ceil(np.log2(seq_length)))
        self.dense_blocks = nn.ModuleList([
            DenseBlock(
                in_channels=in_channels + i * filters,
                dilation=2 ** (i+1),
                filters=filters,
            )
            for i in range(dense_block_count)
        ])

    def forward(self, input):
        # input is dimensions (N, T, in_channels)
        input = torch.transpose(input, 1, 2)
        for block in self.dense_blocks:
            input = block(input)

        return torch.transpose(input, 1, 2)


# TODO: Document
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, key_size, value_size):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)

    def forward(self, input):
        # input is dim (N, T, in_channels) where N is the batch_size, and T is
        # the sequence length
        mask = np.array([[1 if i>j else 0 for i in range(input.shape[1])] for j in range(input.shape[1])])
        mask = torch.BoolTensor(mask).cuda()

        keys = self.linear_keys(input) # shape: (N, T, key_size)
        query = self.linear_query(input) # shape: (N, T, key_size)
        logits = torch.bmm(query, torch.transpose(keys, 1, 2)) # shape: (N, T, T)
        masked_logits = logits.data.masked_fill(mask, -float('inf'))
        probs = F.softmax(masked_logits / self.sqrt_key_size, dim=1) # shape: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        values = self.linear_values(input) # shape: (N, T, value_size)
        read = torch.bmm(probs, values) # shape: (N, T, value_size)

        return torch.cat((input, read), dim=2) # shape: (N, T, in_channels + value_size)
