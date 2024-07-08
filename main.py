import argparse
import time
from torch.utils.data import DataLoader
import torch
from dataloader import MovielensDataset
from utils import *
from model import MF, BPRLoss, trajectoryLoss

parser = argparse.ArgumentParser()
parser.add_argument("--total_epoch", type = int, default = 100, help = "the total number of epoch to train")
parser.add_argument("--start_epoch", type = int, default = 20, help = "first epochs to warm up BPR")
parser.add_argument("--epoch_window", type = int, default = 3, help = "epoch window")
parser.add_argument("--batch_size", type = int, default = 64, help = "size of batch in dataloader")
parser.add_argument("--lamda", type = float, default = 0.1, help = "coefficient lambda in loss function")
parser.add_argument("--embed_dim", type = int, default=128, help = "Embedding dimensions of users and items")
parser.add_argument("--trajectory_loss", type = bool, default = True, help = "Turn on and off the trajectory loss")
parser.add_argument("--top_k", type = int, default = 10, help = "number of k items to measure metric in evaluation")
opt = parser.parse_args()
print(opt)

user_file_path = "../movielens/users.dat"
rating_file_path = "../movielens/ratings.dat"
movie_file_path = "../movielens/movies.dat"

df = merge_movielens_data(user_file_path, rating_file_path, movie_file_path)
df, num_users, num_items = make_index(df)
train_df, test_df = make_train_test_dataset(df)
train_dataset = MovielensDataset(train_df, num_items)
test_dataset = MovielensDataset(test_df, num_items)

print("len of train dataset:", len(train_dataset))
print("len of test dataset:",len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size = opt.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle = False)

model = MF(num_users, num_items, opt.embed_dim)
criterion_bpr = BPRLoss()
criterion_trj = trajectoryLoss(opt.epoch_window)
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)

train_cache_pos_scores = {}
test_cache_pos_scores = {}
for epoch in range(opt.total_epoch):
    train_loss, cache_pos_scores = train_model(epoch, model, train_loader, optimizer, criterion_bpr, criterion_trj, train_cache_pos_scores, opt)
    test_loss, test_cache_pos_scores = evaluate_model(epoch, model, test_loader, criterion_bpr, criterion_trj, test_cache_pos_scores, opt)
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
torch.save(model, f"./tj_loss_{opt.trajectory_loss}")




