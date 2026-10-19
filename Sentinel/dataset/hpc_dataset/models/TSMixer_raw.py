import torch.nn as nn
import torch
import math

class ResBlock(nn.Module):
    def __init__(self, configs):
        super(ResBlock, self).__init__()
        self.temporal = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.seq_len),
            nn.Dropout(configs.dropout)
        )

        self.channel = nn.Sequential(
            nn.Linear(configs.enc_in, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.enc_in),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x, x_hostname_id, x_job_id, x_rack_id, x_cabinet_id, x_node_in_rack_id):
        # x: [B, L, D]
        x = x + self.temporal(x.transpose(2, 3)).transpose(2, 3)
        x = x + self.channel(x)

        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.layer = configs.e_layers
        self.model = nn.ModuleList([ResBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.pred_len = configs.pred_len
        self.projection = nn.Linear(configs.seq_len, configs.pred_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_hostname_id, x_job_id, x_rack_id, x_cabinet_id, x_node_in_rack_id, mask=None):

        # x: [B, L, D]
        for i in range(self.layer):
            x_enc = self.model[i](x_enc,x_hostname_id, x_job_id, x_rack_id, x_cabinet_id, x_node_in_rack_id)
        enc_out = self.projection(x_enc.transpose(2, 3)).transpose(2, 3)

        return enc_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_hostname_id, x_job_id, x_rack_id, x_cabinet_id, x_node_in_rack_id, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast' or self.task_name == 'long_term_forecast_mmd':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, x_hostname_id, x_job_id, x_rack_id, x_cabinet_id, x_node_in_rack_id, mask=mask)
            return dec_out[:, :, -self.pred_len:, :]  # [B, L, D]
        else:
            raise ValueError('Only forecast tasks implemented yet')
