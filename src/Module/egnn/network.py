import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from src.Module.egnn.egnn_pytorch_geometric import EGNN_Sparse
from src.Module.egnn.utils import get_edge_feature_dims, get_node_feature_dims

class nodeEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(nodeEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()
        self.node_feature_dim = get_node_feature_dims()
        for i, dim in enumerate(self.node_feature_dim):
            emb = torch.nn.Linear(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        feature_dim_count = 0
        for i in range(len(self.node_feature_dim)):
            x_embedding += self.atom_embedding_list[i](
                x[:, feature_dim_count:feature_dim_count + self.node_feature_dim[i]])
            feature_dim_count += self.node_feature_dim[i]
        return x_embedding


class edgeEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(edgeEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.edge_feature_dims = get_edge_feature_dims()
        for i, dim in enumerate(self.edge_feature_dims):
            emb = torch.nn.Linear(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        feature_dim_count = 0
        for i in range(len(self.edge_feature_dims)):
            x_embedding += self.atom_embedding_list[i](
                x[:, feature_dim_count:feature_dim_count + self.edge_feature_dims[i]])
            feature_dim_count += self.edge_feature_dims[i]
        return x_embedding


class EGNN(nn.Module):
    def __init__(self, config, input_dim, out_dim):
        super(EGNN, self).__init__()
        self.config = config
        
        if config["embedding"]:
            self.node_embedding = nn.Linear(input_dim, config["node_embedding_dim"])
            input_dim = config["node_embedding_dim"]
            # self.edge_embedding = edgeEncoder(input_dim)
        
        self.mpnn_layes = nn.ModuleList([
            EGNN_Sparse(
                input_dim, 
                m_dim=int(self.config["hidden_channels"]), 
                edge_attr_dim=int(self.config["edge_attr_dim"]), 
                dropout=int(self.config["dropout"]), 
                mlp_num=int(self.config["mlp_num"])) 
            for _ in range(int(self.config["n_layers"]))])
        
        
        
        if config["problem_type"] == "multi_label_classification":
            self.cls_layer = nn.Sequential(
                nn.Linear(input_dim, 2*config["hidden_channels"]), 
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(2*config["hidden_channels"], config["num_labels"])
            )
        self.lin = nn.Linear(input_dim, out_dim)
        self.droplayer = nn.Dropout(float(self.config["dropout"]))
        
    def forward(self, data):
        x, pos, edge_index, edge_attr, batch = (
            data.x, data.pos, 
            data.edge_index,
            data.edge_attr, data.batch
        )
        
        input_x = torch.empty([pos.shape[0], 0]).to(x.device)
        input_x = torch.cat([input_x, x], dim=1)
        mu_r_norm = data.mu_r_norm
        input_x = torch.cat([input_x, mu_r_norm], dim=1)
        
        
        if self.config['embedding']:
            input_x = self.node_embedding(input_x)
            # edge_attr = self.edge_embedding(edge_attr)
        input_x = torch.cat([pos, input_x], dim=1)
        
        for i, layer in enumerate(self.mpnn_layes):
            h = layer(input_x, edge_index, edge_attr, batch)
            if self.config['residual']:
                input_x = input_x + h
            else:
                input_x = h
                
        x = input_x[:, 3:]
        if self.config["problem_type"] == "multi_label_classification":
            x_mean = scatter_mean(x, batch, dim=0)
            x = self.cls_layer(x_mean)
        elif self.config["problem_type"] == "aa_classification":
            x = self.droplayer(x)
            x = self.lin(x)
        return x, x_mean
        
