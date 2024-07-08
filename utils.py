import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def merge_movielens_data(user_file_path, rating_file_path, movie_file_path):
    u_names = ['user_id', 'gender', 'age', 'occupation', 'zip']
    users = pd.read_table(user_file_path, sep='::', header=None, names=u_names, encoding='latin-1',
                          engine='python')
    r_names = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_table(rating_file_path, sep='::', header=None, names=r_names, encoding='latin-1',
                            engine='python')
    m_names = ['movie_id', 'title', 'genres']
    movies = pd.read_table(movie_file_path, sep='::', header=None, names=m_names, encoding='latin-1',
                           engine='python')
    unique_user_ids = ratings['user_id'].unique()
    unique_movie_ids = ratings['movie_id'].unique()
    user_movie_comb = pd.MultiIndex.from_product([unique_user_ids, unique_movie_ids], names = ['user_id', 'movie_id']).to_frame(index = False)
    merged_df = pd.merge(user_movie_comb, ratings, on = ['user_id', 'movie_id'], how = 'left')
    return merged_df[['user_id', 'movie_id', 'rating']]

def make_index(df):
    user_ids = df['user_id'].unique()
    movie_ids = df['movie_id'].unique()
    user2idx = {user_id:idx for idx, user_id in enumerate(user_ids)}
    movie2idx = {movie_id:idx for idx, movie_id in enumerate(movie_ids)}
    num_users = len(user2idx)
    num_items = len(movie2idx)

    df['user_id'] = df['user_id'].apply(lambda x:user2idx[x])
    df['movie_id'] = df['movie_id'].apply(lambda x:movie2idx[x])

    return df, num_users, num_items

def make_train_test_dataset(df):
    positive_interaction = df[df['rating'].notnull()]
    train_df, test_df = train_test_split(positive_interaction, test_size = 0.3, random_state = 42)
    return train_df, test_df

def train_model(current_epoch, model, train_loader, optimizer, criterion_bpr, criterion_trj, train_cache_pos_scores, opt):
    model.train()
    train_loss = 0
    train_cache_pos_scores[current_epoch] = []
    for idx, (users, pos_items, neg_items) in enumerate(train_loader):
        optimizer.zero_grad()
        pos_score, neg_score = model(users, pos_items, neg_items)

        if current_epoch > opt.start_epoch or opt.trajectory_loss == False:
            bpr_loss, before_pos_score = criterion_bpr(pos_score, neg_score)
            train_cache_pos_scores[current_epoch].append(before_pos_score)

            before_pos_score_list = train_cache_pos_scores[current_epoch - opt.epoch_window]
            trj_loss = criterion_trj(pos_score, before_pos_score_list[idx], current_epoch)
            loss = bpr_loss + ((opt.lamda/current_epoch) * trj_loss)
        else:
            loss, before_pos_score = criterion_bpr(pos_score, neg_score)
            train_cache_pos_scores[current_epoch].append(before_pos_score)

        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        train_loss +=loss.item()
    return train_loss / len(train_loader), train_cache_pos_scores

def evaluate_model(current_epoch, model, test_loader, criterion_bpr, criterion_trj, test_cache_pos_scores, opt):
    model.eval()
    test_loss = 0
    test_cache_pos_scores[current_epoch] = []
    with torch.no_grad():
        for idx,(users, pos_items, neg_items) in enumerate(test_loader):
            pos_score, neg_score = model(users, pos_items, neg_items)
            if current_epoch > opt.start_epoch or opt.trajectory_loss == False:
                bpr_loss, before_pos_score = criterion_bpr(pos_score, neg_score)
                test_cache_pos_scores[current_epoch].append(before_pos_score)

                before_pos_score_list = test_cache_pos_scores[current_epoch - opt.epoch_window]
                trj_loss = criterion_trj(pos_score, before_pos_score_list[idx], current_epoch)
                loss = bpr_loss + ((opt.lamda/current_epoch) * trj_loss)
            else:
                loss, before_pos_score = criterion_bpr(pos_score, neg_score)
                test_cache_pos_scores[current_epoch].append(before_pos_score)
            test_loss +=loss.item()

    return test_loss / len(test_loader), test_cache_pos_scores
