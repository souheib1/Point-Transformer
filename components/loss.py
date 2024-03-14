import torch 

def basic_loss(y_pred,y_true):
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(y_pred, y_true.long())