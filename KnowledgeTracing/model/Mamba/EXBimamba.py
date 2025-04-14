import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.modules.mamba_simple import Mamba

'''
Copied from Mamba in Speech, the simplest BiMamba structure.
'''


class ExBimamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.hidd_mamba = Mamba(d_model=self.d_model, d_state=self.d_state, d_conv=self.d_conv, expand=self.expand)
        self.diff_mamba = Mamba(d_model=self.d_model, d_state=self.d_state, d_conv=self.d_conv, expand=self.expand)
        # self.output_proj = nn.Linear(2 * self.d_model, self.d_model)

    def forward(self, hidden, diff):
        hidden_out = self.hidd_mamba(hidden)  # [B, L, d_model]
        diff_out = self.diff_mamba(diff)  # [B, L, d_model]

        return hidden_out, diff_out
