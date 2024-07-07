import torch
from torch.utils.data import Dataset

class MovielensDataset(Dataset):
    def __init__(self, df, num_items):
        self.users = torch.tensor(df['user_id'].values, dtype = torch.long)
        self.items = torch.tensor(df['movie_id'].values, dtype = torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype = torch.float32)
        self.num_items = num_items

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.items[idx]
        neg_item = torch.randint(0, self.num_items, (1,)).item()
        while neg_item in self.items[self.users == user]:
            neg_item = torch.randint(0, self.num_items, (1,)).item()

        return user, pos_item, neg_item

