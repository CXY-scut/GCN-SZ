import torch
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


def BrainGraphBuild1(txt_path, fmri_feature_path, smri_feature_path, sparsity):

    """
    Description: build "AAL: FBM & sMRI + fMRI" or "AAL: GMM & sMRI + fMRI" brain graph
    
    Param txt_path : str, the storage address of the connection matrix file, the address should end with "//"
    Param fmri_feature_path: str, the storage address of fMRI node features (ALFF、ReHo、fDC) .csv file
    Param smri_feature_path：str, the storage address of sMRI node features (GMV、WMV、sDC) .csv file
    Param sparsity : keep the top percent sparsity edges of the network

    return data_list
    data_list: list, all samples data, one element represents one brain graph
    """

    data_list = []
    node_f1 = pd.read_csv(fmri_feature_path)
    node_f2 = pd.read_csv(smri_feature_path)
    
    for i in range(345):  # 345 subjects
        x_ALFF = torch.tensor(node_f1.iloc[i,2:92]).view(90,1)
        x_ReHo = torch.tensor(node_f1.iloc[i,92:182]).view(90,1)
        x_fDC = torch.tensor(node_f1.iloc[i,182:272]).view(90,1)
        x_GMV = torch.tensor(node_f2.iloc[i,2:92]).view(90,1)
        x_WMV = torch.tensor(node_f2.iloc[i,92:182]).view(90,1)
        x_sDC = torch.tensor(node_f2.iloc[i,182:272]).view(90,1)
        x = torch.cat((x_ALFF, x_ReHo, x_fDC, x_GMV, x_WMV, x_sDC), 1)
        
        # node_f1.iloc[i, 0] was the subject's index
        if node_f1.iloc[i, 0] != node_f2.iloc[i, 0]:
            print("feature error")
            break
            
        samp_index = "ROICorrelation_" + node_f1.iloc[i,0] + ".txt"
        edge_attr = np.loadtxt(txt_path + samp_index)
        
        target = 0 if node_f1.iloc[i,0][:2] == "NC" else 1
        edges = edge_attr[np.triu(edge_attr, 1) != 0]
        threshold = np.percentile(abs(edges), (1-sparsity)*100)
        
        G = nx.Graph()
        nodes = list(range(len(edge_attr)))
        G.add_nodes_from(nodes)
        for m in  range(len(edge_attr)):
            for n in range(m+1, len(edge_attr)):
                if abs(edge_attr[m][n]) > threshold:
                    G.add_edge(m,n)
        
        edge_index = torch.from_numpy(np.array(G.edges)).t().contiguous().long()  
        data = Data(x=x, edge_attr = edge_attr, edge_index = edge_index, y = target)
        data_list.append(data)
    return data_list


def BrainGraphBuild2(fbm_path, gmm_path, fmri_feature_path, smri_feature_path, sparsity):
    """
    Description: build "AAL: FBM-GMM & sMRI + fMRI" brain graph
    
    Param fbm_path : str, the storage address of the connection matrix file (fMRI), the address should end with "//"
    aram gmm_path : str, the storage address of the connection matrix file (sMRI), the address should end with "//"
    Param fmri_feature_path: str, the storage address of fMRI node features (ALFF、ReHo、fDC) .csv file
    Param smri_feature_path：str, the storage address of sMRI node features (GMV、WMV、sDC) .csv file
    Param sparsity : keep the top percent sparsity edges of the network

    return data_list
    data_list: list, all samples data, one element represents one brain graph
    """
    data_list = []
    node_f1 = pd.read_csv(fmri_feature_path)
    node_f2 = pd.read_csv(smri_feature_path)
    
    for i in range(345): # 345 subjects
        x_ALFF = torch.tensor(node_f1.iloc[i,2:92]).view(90,1)
        x_ReHo = torch.tensor(node_f1.iloc[i,92:182]).view(90,1)
        x_fDC = torch.tensor(node_f1.iloc[i,182:272]).view(90,1)
        x_GMV = torch.tensor(node_f2.iloc[i,2:92]).view(90,1)
        x_WMV = torch.tensor(node_f2.iloc[i,92:182]).view(90,1)
        x_sDC = torch.tensor(node_f2.iloc[i,182:272]).view(90,1)
        x = torch.cat((x_ALFF, x_ReHo, x_fDC, x_GMV, x_WMV, x_sDC), 1)
        
        # node_f1.iloc[i, 0] was the subject's index
        if node_f1.iloc[i, 0] != node_f2.iloc[i, 0]:
            print("feature error")
            break
            
        fbn_samp_index = "ROICorrelation_" + node_f1.iloc[i,0] + ".txt"
        gmn_samp_indx = "KLS_AAL90" + node_f1.iloc[i,0] + "_mwc1" + node_f1.iloc[i,0] + ".txt"
        fbn_edge_attr = np.loadtxt(fbm_path + fbn_samp_index)
        gmn_edge_attr = np.loadtxt(gmn_path + gmn_samp_indx)
        for j in range(90):
            gmn_edge_attr[j][j] = 1
        scaler = MinMaxScaler()
        fbn_edge_attr_nor = scaler.fit_transform(fbn_edge_attr.reshape(-1,1)).reshape(90,90)
        scaler = MinMaxScaler()
        gmn_edge_attr_nor = scaler.fit_transform(gmn_edge_attr.reshape(-1,1)).reshape(90,90)
        edge_attr = fbn_edge_attr_nor + gmn_edge_attr_nor
        
        target = 0 if node_f1.iloc[i,0][:2] == "NC" else 1
        edges = edge_attr[np.triu(edge_attr, 1) != 0]
        threshold = np.percentile(abs(edges), (1-sparsity)*100)
        
        G = nx.Graph()
        nodes = list(range(len(edge_attr)))
        G.add_nodes_from(nodes)
        
        for m in  range(len(edge_attr)):
            for n in range(m+1, len(edge_attr)):
                if abs(edge_attr[m][n]) > threshold:
                    G.add_edge(m,n)
        
        edge_index = torch.from_numpy(np.array(G.edges)).t().contiguous().long()  
        data = Data(x=x, edge_attr = edge_attr, edge_index = edge_index, y = target)
        data_list.append(data)
    return data_list


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)


class EarlyStopping:  
    def __init__(self, patience=500, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta  
    
    def __call__(self, val_loss, model):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                print("\tEarly Stopping Actived")
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop

