import torch
import torch.nn as nn
import torch.nn.functional as F

class MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user, pos_item, neg_item):
        user_emb = self.user_embedding(user)
        pos_item_emb = self.item_embedding(pos_item)
        neg_item_emb = self.item_embedding(neg_item)

        pos_score = torch.sum(user_emb * pos_item_emb, dim = -1)
        neg_score = torch.sum(user_emb * neg_item_emb, dim = -1)

        return pos_score, neg_score

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        loss = -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score)))
        return loss, pos_score

class trajectoryLoss(nn.Module):
    def __init__(self, epoch_window):
        super(trajectoryLoss, self).__init__()
        self.epoch_window = epoch_window

    def forward(self, pos_score, before_pos_score, current_epoch):
        loss = torch.norm(pos_score - before_pos_score)
        return loss