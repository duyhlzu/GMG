import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7, 11], stride=1, padding='same'):
        super(MultiScaleConv, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=stride, padding=k//2)
            for k in kernel_sizes
        ])

    def forward(self, x):
        return sum(conv(x) for conv in self.convs)

class self_attention_memory_module(nn.Module):  # SAM
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # h(hidden): layer q, k, v
        # m(memory): layer k2, v2
        # layer z, m are for layer after concat(attention_h, attention_m)

        # layer_q, k, v are for h (hidden) layer
        # Layer_ k2, v2 are for m (memory) layer
        # Layer_z, m are using after concatinating attention_h and attention_m layer

        self.layer_q = nn.Conv2d(input_dim, hidden_dim, 1) #对应H的Q
        self.layer_k = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_k2 = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_v = nn.Conv2d(input_dim, input_dim, 1)
        self.layer_v2 = nn.Conv2d(input_dim, input_dim, 1)

        self.layer_z = nn.Conv2d(input_dim * 2, input_dim * 2, 1)
        self.layer_m = nn.Conv2d(input_dim * 3, input_dim * 3, 1)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def forward(self, h, m):
        batch_size, channel, H, W = h.shape
        # feature aggregation
        ##### hidden h attention #####
        K_h = self.layer_k(h)
        Q_h = self.layer_q(h)
        K_h = K_h.view(batch_size, self.hidden_dim, H * W)
        Q_h = Q_h.view(batch_size, self.hidden_dim, H * W)
        Q_h = Q_h.transpose(1, 2)

        A_h = torch.softmax(torch.bmm(Q_h, K_h), dim=-1)  # batch_size, H*W, H*W

        V_h = self.layer_v(h)
        V_h = V_h.view(batch_size, self.input_dim, H * W)
        Z_h = torch.matmul(A_h, V_h.permute(0, 2, 1))

        ###### memory m attention #####
        K_m = self.layer_k2(m)
        V_m = self.layer_v2(m)
        K_m = K_m.view(batch_size, self.hidden_dim, H * W)
        V_m = V_m.view(batch_size, self.input_dim, H * W)
        A_m = torch.softmax(torch.bmm(Q_h, K_m), dim=-1)
        V_m = self.layer_v2(m)
        V_m = V_m.view(batch_size, self.input_dim, H * W)
        Z_m = torch.matmul(A_m, V_m.permute(0, 2, 1))
        Z_h = Z_h.transpose(1, 2).view(batch_size, self.input_dim, H, W)
        Z_m = Z_m.transpose(1, 2).view(batch_size, self.input_dim, H, W)

        ### Z_h & Z_m (from attention) then, concat then computation ####
        W_z = torch.cat([Z_h, Z_m], dim=1)
        Z = self.layer_z(W_z)
        ## Memory Updating (Ref: SA-ConvLSTM)
        combined = self.layer_m(torch.cat([Z, h], dim=1))  # 3 * input_dim
        mo, mg, mi = torch.split(combined, self.input_dim, dim=1)
        ### (Ref: SA-ConvLSTM)
        mi = torch.sigmoid(mi)
        new_m = (1 - mi) * m + mi * torch.tanh(mg)
        new_m = new_m
        new_h = torch.sigmoid(mo) * new_m

        return new_h, new_m

class GlobalFeatureExtractor(nn.Module):
    def __init__(self, in_channels, num_hidden):
        super(GlobalFeatureExtractor, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_hidden, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_hidden, num_hidden)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

class GSAMSpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, att_hidden, height, width, filter_size, stride, layer_norm):
        super(GSAMSpatioTemporalLSTMCell, self).__init__()
        self.num_hidden = num_hidden
        self.att_hidden = att_hidden
        self._forget_bias = 1.0
        padding = filter_size // 2

        self.attention_layer = self_attention_memory_module(num_hidden, att_hidden)
        self.global_feature_extractor = GlobalFeatureExtractor(in_channel, num_hidden)
        self.gate = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, filter_size, stride, padding, bias=False),
                nn.LayerNorm([num_hidden * 7, height, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, filter_size, stride, padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, filter_size, stride, padding, bias=False),
                nn.LayerNorm([num_hidden * 3, height, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, filter_size, stride, padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
            self.conv_g = nn.Sequential(
                # nn.Conv2d(in_channel, num_hidden * 1, filter_size, stride, padding, bias=False),
                MultiScaleConv(in_channel, num_hidden, kernel_sizes=[1, 3, 5], stride=1),
                nn.LayerNorm([num_hidden * 1, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, filter_size, stride, padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, filter_size, stride, padding, bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, filter_size, stride, padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, filter_size, stride, padding, bias=False),
            )
            self.conv_g = nn.Sequential(
                # nn.Conv2d(in_channel, num_hidden * 1, filter_size, stride, padding, bias=False),
                MultiScaleConv(in_channel, num_hidden, kernel_sizes=[1, 3, 5], stride=1),
            )

        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t, motion_highway):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        g_concat = self.conv_g(x_t)

        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)
        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)
        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        m_new_new = self.conv_last(mem)

        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(m_new_new)

        # Global feature extraction and gate mechanism
        global_features = self.global_feature_extractor(x_t)
        global_features = global_features.unsqueeze(2).unsqueeze(3).expand_as(h_new)
        gate = self.gate(torch.cat([h_new, global_features], dim=1))
        # h_new = h_new * gate + global_features * (1 - gate) + h_new

        h_new = h_new * gate + global_features * (1 - gate)
        h_new = h_new * g_concat

        # Attention layer
        h_new, m_new = self.attention_layer(h_new, m_new)

        h_new = h_new + (1 - o_t) * motion_highway
        motion_highway = h_new
        return h_new, c_new, m_new, motion_highway