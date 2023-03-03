import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

class Model(nn.Module):
    def __init__(self, drop_rate):
        super().__init__()
        self.conv1 = gnn.GraphConv(6, 128)
        self.pool1 = gnn.TopKPooling(128, ratio=0.8)
        self.conv2 = gnn.GraphConv(128, 128)
        self.pool2 = gnn.TopKPooling(128, ratio=0.8)
        self.conv3 = gnn.GraphConv(128, 128)
        self.pool3 = gnn.TopKPooling(128, ratio=0.8)
        self.lin1 = nn.Linear(256, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 2)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        gcn1 = F.relu(self.conv1(x, edge_index))
        pool1, edge_index, _, batch, _, _ = self.pool1(gcn1, edge_index, None, batch)
        x1 = torch.cat([gnn.global_max_pool(pool1, batch), gnn.global_mean_pool(pool1, batch)], dim=1)
        
        gcn2 = F.relu(self.conv2(pool1, edge_index))
        pool2, edge_index, _, batch, _, _ = self.pool2(gcn2, edge_index, None, batch)
        x2 = torch.cat([gnn.global_max_pool(pool2, batch), gnn.global_mean_pool(pool2, batch)], dim=1)
        
        gcn3 = F.relu(self.conv3(pool2, edge_index))
        pool3, edge_index, _, batch, _, _ = self.pool3(gcn3, edge_index, None, batch)
        x3 = torch.cat([gnn.global_max_pool(pool3, batch), gnn.global_mean_pool(pool3, batch)], dim=1)
        
        x = x1 + x2 + x3
        fc1 = self.dropout(F.relu(self.bn1(self.lin1(x))))
        fc2 = F.relu(self.bn2(self.lin2(fc1)))
        fc3 = F.softmax(self.lin3(fc2), dim=1)

        return fc3