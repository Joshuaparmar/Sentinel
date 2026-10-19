import torch.nn as nn
import torch
import math

class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.full_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.full_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.full_weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, indices=None, jobs=None, racks=None, dim=1):
        # Select a subset of the weight matrix based on the indices. The length of indices should be equal to input. # Input : (batch_size, time, features, len_nodes)
        batch= indices.shape[0]
        num_hnodes = indices.shape[1]
        time_win = indices.shape[2]
        weight = self.full_weight.index_select(1, indices.reshape(-1))
        weight = weight.reshape(self.out_features, batch, num_hnodes, time_win)
        mask = ((jobs[:, :, None, :]==jobs[:, None, :, :]) & (jobs[:, :, None, :]!= 0) & (jobs[:, None, :, :]!= 0)).int()
        mask = mask.permute(1,0,2,3)
        masked_weight= weight * mask
        y = torch.einsum('bijk,lbki->bijl', input, masked_weight)
        return y

class NodeMixingResBlock(nn.Module):
    def __init__(self, in_dim: int , width_batch: int, out_dim: int,dropout: float):
        super(NodeMixingResBlock, self).__init__()
        self.width_batch = width_batch
        self.in_dim = in_dim
        self.norm = nn.LayerNorm(normalized_shape=[in_dim])
        self.lin_1 = MaskLinear(in_features=in_dim, out_features=out_dim)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, entryid: torch.Tensor, jobid: torch.Tensor, rackid: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, len_nodes , time, features)
        y = self.norm(x.transpose(1, 3)).transpose(1, 3)  # Transpose to normalize across node dimension
        batch_size = y.shape[0]
        y = x.permute(0, 2, 3, 1)
        y = self.lin_1(y, indices= entryid, jobs=jobid, racks=rackid)
        y = self.act(y)
        y = self.dropout_1(y)
        y = y.permute(0, 3, 1, 2)
        return x + y


class ResBlock(nn.Module):
    def __init__(self, configs):
        super(ResBlock, self).__init__()
        self.node = NodeMixingResBlock(configs.num_hostname, configs.batch_size, configs.num_hostname, configs.dropout)
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
        x = x + self.node(x, x_hostname_id, x_job_id, x_rack_id)
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
