import torch
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
from torch.utils.data import Dataset


def BrainGraphBuild(txt_path, fmri_feature_path, smri_feature_path, sparsity):

    """
    Description: build "AAL: FBM & sMRI + fMRI" orbrain graph
    
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
fbn_path = "//home//diu//diu//data//Network//345_FC//AAL90(from116)_FC_345//"  # 功能邻接矩阵
gmn_path = "//home//diu//diu//data//Network//345_MWC//KLS_mwc1_AAL90_345_txtdata//"  # 结构sMRI邻接矩阵
save_dir_path = "//home//diu//diu//data//Network//Fusion//FBN_GMN_fs_sparse//"  # 保存构建好的脑拓扑图，一个文件就包含所有样本的脑拓扑图
feature_path1 = "//home//diu//diu//data//Node_features//345//DC_ALFF_ReHo_90.csv"  # 读取的节点特征，这个是功能MRI三个特征，没有进行归一化或者标准化
feature_path2 = "//home//diu//diu//data//Node_features//345//DC_GMV_WMV_90.csv"


# 将功能和结构网络，边权值进行归一化后相加构建新网络
# 节点特征暂定使用 功能特征
def Graph_fusion_dataset(fbm_path, gmn_path, feature_path1, feature_path2, sparsity):
    data_list = []
    node_f1 = pd.read_csv(feature_path1) # 功能特征 DC_ReHo_ALFF
    node_f2 = pd.read_csv(feature_path2) 
    
    for i in range(345):
        x1 = torch.tensor(node_f1.iloc[i,2:92]).view(90,1)
        x2 = torch.tensor(node_f1.iloc[i,92:182]).view(90,1)
        x3 = torch.tensor(node_f1.iloc[i,182:272]).view(90,1)
        x4 = torch.tensor(node_f2.iloc[i,2:92]).view(90,1)
        x5 = torch.tensor(node_f2.iloc[i,92:182]).view(90,1)
        x6 = torch.tensor(node_f2.iloc[i,182:272]).view(90,1)
        x = torch.cat((x1, x2, x3, x4, x5, x6), 1)
        
        if node_f1.iloc[i, 0] != node_f2.iloc[i, 0]:
            print("feature error")
            break
        fbn_samp_index = "ROICorrelation_" + node_f1.iloc[i,0] + ".txt"
        gmn_samp_indx = "KLS_AAL90" + node_f1.iloc[i,0] + "_mwc1" + node_f1.iloc[i,0] + ".txt"
        fbn_edge_attr = np.loadtxt(fbm_path + fbn_samp_index)
        gmn_edge_attr = np.loadtxt(gmn_path + gmn_samp_indx)
        # GMN对角线元素是0，FBN对角线元素是1，将GMN的也改为1
        for j in range(90):
            gmn_edge_attr[j][j] = 1
        # 各自归一化后相加
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
        
        edge_index = torch.from_numpy(np.array(G.edges)).t().contiguous().long()  # 深拷贝
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

