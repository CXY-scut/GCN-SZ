import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from model import Model
from Correlation_function import BrainGraphBuild1, BrainGraphBuild2, MyDataset, EarlyStopping


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=777, help="seed for initialization")  
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--batch_size", type=int, default=30, help="batch_size")
parser.add_argument("--drop_rate", type=float, default=0.5, help="dropout rate")
parser.add_argument("--patience", type=int, default=500, help="patience of earlystopping")
parser.add_argument("--max_epoch", type=int, default=1000, help="max epoch")
parser.add_argument("--sparsit", type=list, default=0.10, help="is used to build brain graph")
parser.add_argument("--train_size", type=int, default=300, help="train_data_size")
parser.add_argument("--test_size", type=int, default=45, help="test_data_szie")
args = parser.parse_args(args = [])

args.device = "cpu"
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = "cuda:0"

# build the brain graph 
txt_path = "xxx"
fmri_feature_path = "xxx"
smri_feature_path = "xxx"
data_list = BrainGraphBuild1(txt_path, fmri_feature_path, smri_feature_path, args.sparsity)
# data_list = BrainGraphBuild2(fbm_path, gmm_path, fmri_feature_path, smri_feature_path, args.sparsity)

def TrainOnce(model, criterion, optimizer, train_loader):
    """
    Description: One epoch in train process
    return train_acc, train_loss
    """
    loss_train, correct_train, train_number = 0, 0, args.train_size
    for train_batch in train_loader:
        model.train()
        model.double()
        optimizer.zero_grad()
        logit = model(train_batch.to(args.device))
        pred_train = logit.argmax(dim=1)
        loss = criterion(logit, train_batch.y.to(args.device))
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        correct_train += pred_train.eq(train_batch.y.to(args.device)).sum().item()  
    
    train_acc = correct_train/train_number
    train_loss = loss_train/train_number
    return train_acc, train_loss
    

def TestOnce(model, criterion, test_loader):
    """
    Description: One epoch in test process
    return test_acc, test_loss, auc, recall, specificity, f1, precision
    """
    loss_test, correct_test, test_number = 0, 0, args.test_size
    for test_batch in test_loader:
        model.eval()
        logit = model(test_batch.to(args.device))
        pred_test = logit.argmax(dim=1)
        loss = criterion(logit, test_batch.y.to(args.device))
        loss_test += loss.item()
        correct_test += pred_test.eq(test_batch.y.to(args.device)).sum().item()
    
    test_acc = correct_test/test_number
    test_loss = loss_test/test_number
    
    pred_test = pred_test.cpu().numpy()
    test_batch.y = test_batch.y.cpu().numpy()
    confusion = confusion_matrix(test_batch.y, pred_test)  
    auc = roc_auc_score(test_batch.y, logit[:,1].tolist())
    recall = recall_score(test_batch.y, pred_test)
    specificity = confusion[0][0]/(confusion[0][0] + confusion[0][1])
    f1 = f1_score(test_batch.y, pred_test)
    precision = precision_score(test_batch.y, pred_test)

    return test_acc, test_loss, auc, recall, specificity, f1, precision
     


for random_state in range(0, 1000, 100):  # random_state was the seed for train/test splitting
    
    model = Model(args.drop_rate).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    train_data, test_data  = train_test_split(data_list, test_size = args.test_size/(args.train_size + args.test_size), shuffle=True, random_state=random_state)
    train_set, test_set = MyDataset(train_data), MyDataset(test_data)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.test_size, shuffle=True)
    
    Train_acc, Test_acc, Train_loss, Test_loss = [], [], [], []
    AUC, Recall, Specificity, F1, Precision = [], [], [], [], []

    
    for epoch in range(args.max_epoch):
        train_acc, train_loss = TrainOnce(model, criterion, optimizer, train_loader)
        test_acc, test_loss, auc, recall, specificity, f1, precision = TestOnce(model, criterion, test_loader)
        
        Train_acc.append(train_acc)
        Test_acc.append(test_acc)
        Train_loss.append(train_loss)
        Test_loss.append(test_loss)
        AUC.append(auc)
        Recall.append(recall)
        Specificity.append(specificity)
        F1.append(f1)
        Precision.append(precision)
        
        early_stopping(test_loss)
        if early_stopping.early_stop:
            break
    
    # save the result
    save_path = "xxx"
    result = pd.DataFrame({"train_loss":Train_loss, "test_loss": Test_loss, "train_acc": Train_acc, "test_acc":Test_acc,
                           "auc": AUC, "recall": Recall, "specificity":Specificity, "f1":F1, "precision":Precision})
    result.to_csv(save_path + "result-seed%s.csv"%(random_state))
    




    
    
    
    
    
    
    
    
    
