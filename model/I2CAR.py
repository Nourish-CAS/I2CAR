import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange,reduce,repeat
from .embed import DataEmbedding, TokenEmbedding
from torch_geometric.nn import GCNConv,global_mean_pool,SAGEConv,GraphConv



class nconv(nn.Module):

    def __init__(self, gnn_type):
        super(nconv, self).__init__()
        self.gnn_type = gnn_type

    def forward(self, x, A):
        if self.gnn_type == 'time':
            x = torch.einsum('btc,tl->blc', x, A)
        elif self.gnn_type == 'var':
            x = torch.einsum('btc,cv->btv', x, A)
        return x.contiguous()




class gcn(nn.Module):
   
    def __init__(self, c_in, c_out, dropout, gnn_type,k, order=3):
        super(gcn, self).__init__()
        self.order = order
        self.nconv=nconv(gnn_type)
        self.gnn_type = gnn_type
        self.c_in = (order + 1) * c_in
        self.mlp = nn.Linear(self.c_in, c_out)
        self.cout = c_out
        self.dropout = dropout
        self.act = nn.GELU()
        self.k = k

    def forward(self, x, A):
        out = [x]
        x1 =x
        adj = A

        for i in range(0, self.order):
            x2 = self.nconv(x1, adj)
            out.append(x2)
            x1 = x2

        h = torch.cat(out, dim=-1)
        h = self.mlp(h)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


def get_top_k_adjacency_matrix_var(adj_matrix, k):
  
    N = adj_matrix.size(0)
    mask = torch.zeros_like(adj_matrix)  

    for i in range(N):
        top_k_indices = torch.topk(adj_matrix[i], k=k, largest=True).indices
        mask[i, top_k_indices] = 1 

   
    sparse_adj_matrix = adj_matrix * mask
    sparse_adj_matrix = F.softmax(sparse_adj_matrix, dim=-1) 
    return sparse_adj_matrix


def get_top_k_adjacency_matrix_time(adj_matrix, k):
   
    N = adj_matrix.size(0)
    mask = torch.zeros_like(adj_matrix) 

    for i in range(N):
        top_k_indices = torch.topk(adj_matrix[i], k=k, largest=True).indices
        mask[i, top_k_indices] = 1

        neighbor_indices = (adj_matrix[i] > 0).nonzero(as_tuple=True)[0]
        mask[i, neighbor_indices] = 1

    sparse_adj_matrix = adj_matrix * mask
    return sparse_adj_matrix


def get_top_k_adjacency_matrix(adj_matrix, k):

    if k >= adj_matrix.size(1):
        return adj_matrix

    top_k_values, top_k_indices = torch.topk(adj_matrix, k=k, dim=-1)

    sparse_adj = torch.zeros_like(adj_matrix)

    sparse_adj.scatter_(-1, top_k_indices, top_k_values)

    sparse_adj = F.softmax(sparse_adj, dim=-1)

    return sparse_adj


class single_scale_gnn(nn.Module):
    def __init__(self,enc_in, d_model,dropout,window_size,patch_size):
        super(single_scale_gnn, self).__init__()

        self.seq_len = window_size
        self.enc_in = enc_in
        self.hidden = enc_in
        self.dropout = 0.05
        self.dropout_1 = nn.Dropout(0.08)
        self.patch_size = patch_size
        self.window_size = window_size


        for i, patchsize in enumerate(self.patch_size):
            setattr(self, f"timevec_{i}", nn.Parameter(torch.randn(self.seq_len//patchsize, self.seq_len//patchsize)))
            setattr(self, f"varvec_{i}", nn.Parameter(torch.randn(self.enc_in, self.enc_in)))
            setattr(self, f"shared_matrix_time{i}", nn.Parameter(torch.randn(self.enc_in, self.seq_len)))


        self.time_gcn = nn.ModuleList()
        self.var_gcn = nn.ModuleList()
        self.time_gcn = gcn(d_model, self.hidden, self.dropout, gnn_type='time', k=5)
        self.var_gcn = gcn(self.hidden, self.hidden, self.dropout, gnn_type='var', k=6)


        self.start_linear = nn.Linear(self.enc_in, self.hidden)
        self.output_linear = nn.Linear(self.hidden, 1)
        self.act = nn.GELU()

        self.time_linear = nn.Linear(d_model,enc_in)

        self.embedding_window_size = DataEmbedding(enc_in, d_model, dropout)

    def get_time_adj(self,timevec):
        adj = self.act(torch.einsum('td,dm->tm', timevec, timevec))
        adj = F.softmax(adj, dim=-1)  
        return adj

    def get_var_adj(self,varvec):
       
        adj = self.act(torch.einsum('vd,dn->vn', varvec, varvec))
        adj = F.softmax(adj, dim=-1) 
        return adj

    def aggregate_patches(self,data, patch_size):
       
        length = data.shape[1]
        assert length % patch_size == 0, f"Length {length} can be divided by patch_size {patch_size}"

        data_patched = rearrange(data, 'b (p l) c -> b p l c',
                                 p=patch_size)  # [batch, length/patch_size, patch_size, channels]

        data_aggregated = reduce(data_patched, 'b p l c -> b l c', 'mean')  # [batch, length/patch_size, channels]
        # data_aggregated = reduce(data_patched, 'b p l c -> b l c', 'sum')  # [batch, length/patch_size, channels]
        return data_aggregated


    def forward(self, x):
      
        batch_size = x.shape[0]
       
        time_list = []
        var_list = []
        time_consis_list = []
        var_consis_list = []

        for i, patchsize in enumerate(self.patch_size):
            timevec = getattr(self, f"timevec_{i}")
            varvec = getattr(self, f"varvec_{i}")
            time_adj = self.get_time_adj(timevec)  # [seq_len/patchsize, seq_len/patchsize]
            var_adj = self.get_var_adj(varvec)  # [enc_in, enc_in]
            x_patch = self.aggregate_patches(x,patchsize)
            x_t = self.embedding_window_size(x_patch)
            time_adj = get_top_k_adjacency_matrix_time(time_adj, k=5)
            var_adj = get_top_k_adjacency_matrix_var(var_adj, k=6)
            x_time = self.time_gcn(x_t, time_adj)
            x_var = self.var_gcn(x_patch, var_adj)
            time_consis_list.append(x_time)
            var_consis_list.append(x_var)

            ####shared matrix
            shared_matrix_time = getattr(self, f"shared_matrix_time{i}")
            x_time = torch.einsum('bij,jk->bik', x_time, shared_matrix_time)
            x_var = torch.einsum('bij,jk->bik', x_var, shared_matrix_time)

            x_time = x_time.unsqueeze(1)
            x_var = x_var.unsqueeze(1)

            ####upsampling
            x_time = x_time.repeat(1,1,patchsize,1)
            x_var = x_var.repeat(1,1,patchsize,1)

            x_time = F.softmax(x_time, dim=-1)
            x_var = F.softmax(x_var, dim=-1)

            time_list.append(x_time)
            var_list.append(x_var)



        return time_list,var_list,time_consis_list,var_consis_list







class I2CAR(nn.Module):
    def __init__(self, win_size, enc_in, c_out, n_heads=1, d_model=256, e_layers=3, patch_size=[3,5,7], channel=55, d_ff=512, dropout=0.0, activation='gelu', output_attention=True):
        super(I2CAR, self).__init__()
        self.output_attention = output_attention
        self.patch_size = patch_size
        self.patch_size_len = 1
        self.channel = channel
        self.win_size = win_size
        self.global_pool = global_mean_pool 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       

        self.seq_len = win_size
        self.graph_encs = nn.ModuleList()
        self.enc_layers = self.patch_size_len
        self.anti_ood = 1
        for i in range(self.enc_layers):
            self.graph_encs.append(single_scale_gnn(enc_in=enc_in,d_model=d_model,dropout=dropout,window_size=win_size,patch_size=self.patch_size))

        self.weights_time = nn.Parameter(torch.randn(self.enc_layers)).to(device=self.device) 
        self.weights_var = nn.Parameter(torch.randn(self.enc_layers)).to(device=self.device) 
       
    def forward(self, x):
        B, L, M = x.shape #Batch win_size channel

        for i in range(self.enc_layers):
            x_time,x_val,time_consis_list,var_consis_list = self.graph_encs[i](x)


        if self.output_attention:
            # return series_time_mean, series_var_mean
            return x_time, x_val,time_consis_list,var_consis_list
        else:
            return None
#
