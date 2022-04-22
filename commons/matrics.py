import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def compute_loss(pos_score, neg_score, device):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

def accuracy(ground_labels,pred,device,threshold=0.5):
    labels_pred = pred > threshold
    return sum(ground_labels.to(device) == labels_pred)/len(ground_labels)

def recall(positive_indexs,pred,threshold=0.5):
    labels_pred = pred > threshold
    return sum(labels_pred[positive_indexs]) /len(positive_indexs)