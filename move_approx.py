
import pandas as pd
import torch
import pytorch_lightning as pl
from tqdm import tqdm
import torchmetrics
import math
from urllib.request import urlretrieve
from zipfile import ZipFile
import os
import torch.nn as nn
import numpy as np
from math import sqrt
from scipy.sparse import csr_matrix
from config import get_config
import pickle
import time

"""## Settings"""

# WINDOW_SIZE = 20
config, unparsed = get_config()
max_n = config.max_n

"""## Data"""
group_size = config.group_size
train_ratio = config.train_ratio
val_ratio = config.val_ratio
ending_ts = config.ending_ts
n = config.n
interval = config.interval
num_heads = config.num_heads

ending_ts_date = config.ending_ts.split(' ')[0]
if config.O == 0 and config.T == 0:
    groups_f = open('data/p_labels_exactbased_'+ending_ts_date+'_n'+str(n)+'_max_n'+str(max_n)+'interval'+str(interval)+'_group_size'+str(group_size)+'.pkl','rb')
    p_f = open('data/passengers_exactbased_'+ending_ts_date+'_n'+str(n)+'_max_n'+str(max_n)+'interval'+str(interval)+'_group_size'+str(group_size)+'.pkl','rb')
    p_trajs_f = open('data/p_trajs_exactbased_'+ending_ts_date+'_n'+str(n)+'_max_n'+str(max_n)+'interval'+str(interval)+'_group_size'+str(group_size)+'.pkl','rb')
elif config.O != 0:
    groups_f = open(
        'data/p_labels_exactbased_' + ending_ts_date + '_n' + str(n) + '_max_n' + str(max_n) + 'interval' + str(
            interval) + '_group_size' + str(group_size) +'_O'+str(config.O)+ '.pkl', 'rb')
    p_f = open(
        'data/passengers_exactbased_' + ending_ts_date + '_n' + str(n) + '_max_n' + str(max_n) + 'interval' + str(
            interval) + '_group_size' + str(group_size) +'_O'+str(config.O)+ '.pkl', 'rb')
    p_trajs_f = open(
        'data/p_trajs_exactbased_' + ending_ts_date + '_n' + str(n) + '_max_n' + str(max_n) + 'interval' + str(
            interval) + '_group_size' + str(group_size) +'_O'+str(config.O)+ '.pkl', 'rb')
elif config.T != 0:
    groups_f = open(
        'data/p_labels_exactbased_' + ending_ts_date + '_n' + str(n) + '_max_n' + str(max_n) + 'interval' + str(
            interval) + '_group_size' + str(group_size) +'_T'+str(config.T)+ '.pkl', 'rb')
    p_f = open(
        'data/passengers_exactbased_' + ending_ts_date + '_n' + str(n) + '_max_n' + str(max_n) + 'interval' + str(
            interval) + '_group_size' + str(group_size) +'_T'+str(config.T)+ '.pkl', 'rb')
    p_trajs_f = open(
        'data/p_trajs_exactbased_' + ending_ts_date + '_n' + str(n) + '_max_n' + str(max_n) + 'interval' + str(
            interval) + '_group_size' + str(group_size) +'_T'+str(config.T)+ '.pkl', 'rb')
labels = pickle.load(groups_f)
groups_f.close()
print(len(labels))

np.sum(labels, axis = 0)

passengers = pickle.load(p_f)
p_f.close()
print(len(passengers))

p_trajs = pickle.load(p_trajs_f)
p_trajs_f.close()
print(len(p_trajs))


p_stations = {}
p_timeintervals = {}
for p in p_trajs:
    stations = []
    timeintervals = []
    trjs = p_trajs[p]
    for tripsi in range(len(trjs)):
        stations.append(trjs[tripsi][0])
        stations.append(trjs[tripsi][1])
        timeintervals.append(trjs[tripsi][2])
        timeintervals.append(trjs[tripsi][3])
    p_stations[p] = stations
    p_timeintervals[p] = timeintervals

p_stations_list = []
p_timeintervals_list = []
for i in range(len(passengers)):
    p_stations_list.append(p_stations[passengers[i]])
    p_timeintervals_list.append(p_timeintervals[passengers[i]])

ps = set()
timeintervals = set()
for p_stations in p_stations_list:
    ps.update(p_stations)
for p_timeintervals in p_timeintervals_list:
    timeintervals.update(p_timeintervals)

ps_f = open('data/ps_'+ending_ts_date+'_n'+str(n)+'_max_n'+str(max_n)+'interval'+str(interval)+'_group_size'+str(group_size)+'.pkl', 'wb')
pickle.dump(list(ps), ps_f, protocol=4)
ps_f.close()

timeintervals_f = open('data/timeintervals_'+ending_ts_date+'_n'+str(n)+'_max_n'+str(max_n)+'interval'+str(interval)+'_group_size'+str(group_size)+'.pkl', 'wb')
pickle.dump(list(timeintervals), timeintervals_f, protocol=4)
timeintervals_f.close()

users = np.asarray(list(passengers.keys()))
ps_f = open('data/ps_'+ending_ts_date+'_n'+str(n)+'_max_n'+str(max_n)+'interval'+str(interval)+'_group_size'+str(group_size)+'.pkl','rb')
ps = np.asarray(pickle.load(ps_f))
ps_f.close()
timeintervals_f = open('data/timeintervals_'+ending_ts_date+'_n'+str(n)+'_max_n'+str(max_n)+'interval'+str(interval)+'_group_size'+str(group_size)+'.pkl','rb')
timeintervals = np.asarray(pickle.load(timeintervals_f))
timeintervals_f.close()

def find_num_heads(num,h):
    for i in range(h+1, num+1):
        j = h - (i - h)
        if num % i == 0:
            return i
        if j > 1 and num % j == 0:
            return j


def padding_tmp(data_t):
    max_len = max(len(x) for x in data_t)
    dataset_padding = [x + [0] * (max_len - len(x)) for x in data_t]
    dataset_mask = [[1.0] * len(x) + [0.0] * (max_len - len(x)) for x in data_t]

    return dataset_padding, dataset_mask

p_stations_list_padding, p_stations_list_mask = padding_tmp(p_stations_list)
p_timeinterval_list_padding, _ = padding_tmp(p_timeintervals_list)

sequence_temp = [','.join(str(x) for x in p_stations_list_padding[i]) for i in range(len(p_stations_list_padding))]
rating_temp = [','.join(str(x) for x in labels[i]) for i in range(len(labels))]
time_temp = [','.join(str(x) for x in p_timeinterval_list_padding[i]) for i in range(len(p_timeinterval_list_padding))]
p_stations_list_mask_temp = [','.join(str(x) for x in p_stations_list_mask[i]) for i in range(len(p_stations_list_mask))]

ratings_data_transformed = pd.DataFrame(list(zip(list(passengers.keys()), sequence_temp,rating_temp,time_temp, p_stations_list_mask_temp)),
                                        columns = ['user_id', 'sequence_movie_ids', 'sequence_ratings', 'time_interval_ids', 'p_stations_list_mask'])

seed_sample = np.random.rand(len(ratings_data_transformed.index))
random_selection_train = seed_sample <= train_ratio
random_selection_val = seed_sample >= (train_ratio+val_ratio)
random_selection_test = (~random_selection_train) & (~random_selection_val)
train_data = ratings_data_transformed[random_selection_train]
val_data = ratings_data_transformed[random_selection_val]
test_data = ratings_data_transformed[random_selection_test]

train_data.to_csv("data/train_data_max_"+ending_ts_date+'_n'+str(n)+'_max_n'+str(max_n)+'interval'+str(interval)+'_group_size'+str(group_size)+".csv", index=False, sep=",")
tmp = train_data.sequence_ratings.tolist()
train_labels = [[float(v_t) for v_t in t.split(',')] for t in tmp]
train_labels_pos = np.sum(train_labels, axis = 0)
train_labels_pos = np.asarray([tmp if tmp!=0 else 0.1 for tmp in train_labels_pos])
weight_class = (np.shape(train_labels)[0] - train_labels_pos) / train_labels_pos

pos_weight = torch.LongTensor(weight_class)

val_data.to_csv("data/val_data_max_"+ending_ts_date+'_n'+str(n)+'_max_n'+str(max_n)+'interval'+str(interval)+'_group_size'+str(group_size)+".csv", index=False, sep=",")
test_data.to_csv("data/test_data_max_"+ending_ts_date+'_n'+str(n)+'_max_n'+str(max_n)+'interval'+str(interval)+'_group_size'+str(group_size)+".csv", index=False, sep=",")

import pytorch_lightning as pl
import torchmetrics
import math
import os
import torch.nn as nn
import numpy as np

import pandas as pd
import torch
import torch.utils.data as data
class MovieDataset(data.Dataset):
    def __init__(
        self, ratings_file,test=False
    ):
        self.ratings_frame = pd.read_csv(
            ratings_file,
            delimiter=",",
        )
        self.test = test

    def __len__(self):
        return len(self.ratings_frame)

    def __getitem__(self, idx):
        data = self.ratings_frame.iloc[idx]
        user_id = data.user_id
        
        movie_history = eval(data.sequence_movie_ids)
        movie_history_ratings = eval(data.sequence_ratings)
        target_movie_id = movie_history[-1:][0]
        # target_movie_rating = movie_history_ratings[-1:][0]
        target_movie_rating = movie_history_ratings
        target_movie_rating = torch.LongTensor(target_movie_rating)
        
        movie_history = torch.LongTensor(movie_history[:-1])
        # movie_history_ratings = torch.LongTensor(movie_history_ratings[:-1])

        time_interval_ids = eval(data.time_interval_ids)
        time_interval_ids_historay = torch.LongTensor(time_interval_ids[:-1])
        target_time_interval_ids = time_interval_ids[-1:][0]

        p_stations_list_mask = eval(data.p_stations_list_mask)
        p_stations_list_mask = torch.FloatTensor(p_stations_list_mask)
        return user_id, movie_history, target_movie_id, target_movie_rating, time_interval_ids_historay, target_time_interval_ids, p_stations_list_mask
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)

class approx(pl.LightningModule):
    def __init__(
        self, args=None,
    ):
        super().__init__()
        super(approx, self).__init__()
        
        self.save_hyperparameters()
        self.args = args
        self.embeddings_user_id = nn.Embedding(
            int(users.max())+1, int(math.sqrt(users.max()))+1
        )
        self.embeddings_movie_id = nn.Embedding(
            int(ps.max())+1, int(math.sqrt(ps.max()))+1
        )
        self.embedding_timeinterval_id = nn.Embedding(
            int(timeintervals.max())+1, int(math.sqrt(timeintervals.max()))+1
        )
        transform_d1 = int(math.sqrt(ps.max()))+1 + int(math.sqrt(timeintervals.max()))+1

        if transform_d1 % config.num_heads != 0:
            config.num_heads = find_num_heads(transform_d1, config.num_heads)
        print(transform_d1, config.num_heads)
        if  config.ablation_flag == 0 or config.ablation_flag == 2:
            self.transfomerlayer = nn.TransformerEncoderLayer(transform_d1, config.num_heads, dropout=0.2)
        self.linear = nn.Sequential(
            nn.LazyLinear(
                config.hidden_d,
            ),
            nn.LeakyReLU(),
            nn.Linear(config.hidden_d, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, np.shape(labels)[1]),
        )
        if config.ablation_flag != 2:
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError()

    def encode_input(self,inputs):
        user_id, movie_history, target_movie_id, target_movie_rating, time_interval_ids_historay, target_time_interval_ids, p_stations_list_mask = inputs
               
        movie_history = self.embeddings_movie_id(movie_history)
        target_movie = self.embeddings_movie_id(target_movie_id)
         
        target_movie = torch.unsqueeze(target_movie, 1)
        transfomer_features = torch.cat((movie_history, target_movie),dim=1)

        user_id = self.embeddings_user_id(user_id)

        time_intervals_historay = self.embedding_timeinterval_id(time_interval_ids_historay)
        target_time_interval = self.embedding_timeinterval_id(target_time_interval_ids)

        target_time_interval = torch.unsqueeze(target_time_interval, 1)
        transfomer_time_intervals = torch.cat((time_intervals_historay, target_time_interval),dim=1)

        user_features = user_id
        
        return transfomer_features, user_features, target_movie_rating.float(), transfomer_time_intervals, p_stations_list_mask
    
    def forward(self, batch):
        transfomer_features, user_features, target_movie_rating, transfomer_time_intervals, p_stations_list_mask = self.encode_input(batch)
        transfomer_features = torch.cat((transfomer_features, transfomer_time_intervals), dim=2)
        p_stations_list_mask = torch.transpose(p_stations_list_mask,0,1)
        if config.ablation_flag == 0 or config.ablation_flag == 2:
            transformer_output = self.transfomerlayer(transfomer_features, src_key_padding_mask = p_stations_list_mask)
            transformer_output = torch.flatten(transformer_output,start_dim=1)
        else:
            transformer_output = torch.flatten(transfomer_features, start_dim=1)
        
        features = torch.cat((transformer_output,user_features),dim=1)

        output = self.linear(features)
        return output, target_movie_rating
        
    def training_step(self, batch, batch_idx):
        out, target_movie_rating = self(batch)
        loss = self.criterion(out, target_movie_rating)
        
        mae = self.mae(out, target_movie_rating)
        mse = self.mse(out, target_movie_rating)
        rmse =torch.sqrt(mse)
        self.log(
            "train/mae", mae, on_step=True, on_epoch=False, prog_bar=False
        )
        
        self.log(
            "train/rmse", rmse, on_step=True, on_epoch=False, prog_bar=False
        )
        
        self.log("train/step_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        out, target_movie_rating = self(batch)
        loss = self.criterion(out, target_movie_rating)

        mae = self.mae(out, target_movie_rating)
        mse = self.mse(out, target_movie_rating)
        rmse = torch.sqrt(mse)

        return {"val_loss": loss, "mae": mae.detach(), "rmse": rmse.detach()}
    
    def test_step(self, batch, batch_idx):
        out, target_movie_rating = self(batch)
        loss = self.criterion(out, target_movie_rating)
        
        mae = self.mae(out, target_movie_rating)
        mse = self.mse(out, target_movie_rating)
        rmse =torch.sqrt(mse)

        return {"test_loss": loss, "mae": mae.detach(), "rmse":rmse.detach(), "top14":out, "users":batch[0], "ground_truth":target_movie_rating}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_mae = torch.stack([x["mae"] for x in outputs]).mean()
        avg_rmse = torch.stack([x["rmse"] for x in outputs]).mean()
        
        self.log("val/loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/mae", avg_mae, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/rmse", avg_rmse, on_step=False, on_epoch=True, prog_bar=False)

    def _check_targets(self,y_true, y_pred):
        type_true = 'multilabel-indicator'
        type_pred = 'multilabel-indicator'

        y_type = {type_true, type_pred}
        if y_type == {"binary", "multiclass"}:
            y_type = {"multiclass"}

        if len(y_type) > 1:
            raise ValueError("Classification metrics can't handle a mix of {0} "
                             "and {1} targets".format(type_true, type_pred))

        y_type = y_type.pop()

        if (y_type not in ["binary", "multiclass", "multilabel-indicator"]):
            raise ValueError("{0} is not supported".format(y_type))

        if y_type.startswith('multilabel'):
            y_true = csr_matrix(y_true)
            y_pred = csr_matrix(y_pred)
            y_type = 'multilabel-indicator'

        return y_type, y_true, y_pred

    def new_hamming_loss(self,y_true, y_pred, labels=None, sample_weight=None):
        y_type, y_true, y_pred = self._check_targets(y_true, y_pred)

        if sample_weight is None:
            weight_average = 1.
        else:
            weight_average = np.mean(sample_weight)

        if y_type.startswith('multilabel'):
            y_dif = np.sum(np.abs(y_true - y_pred), axis=1)
            y_t = np.sum(y_true, axis=1)
            sum_y = 0.0
            count = 0.0
            for i in range(y_dif.shape[0]):
                if y_t[i][0] != 0:
                    sum_y += y_dif[i, 0] / y_t[i, 0]
                    count+=1.0
                    if y_dif[i, 0] == 0 and y_t[i, 0] >= 1:
                        print(i)
            n_differences = sum_y / count

            print("y_true.shape[0],y_true.shape[1]", y_true.shape[0], y_true.shape[1])
            return n_differences

    def test_epoch_end(self, outputs):
        users = torch.cat([x["users"] for x in outputs])
        y_hat = torch.cat([x["top14"] for x in outputs])
        users = users.tolist()
        y_hat = y_hat.tolist()
        ground_truth = torch.cat([x["ground_truth"] for x in outputs])
        ground_truth = ground_truth.tolist()

        predicted = [[0 if y_hat[i][j]<0.5 else 1 for j in range(len(y_hat[0]))] for i in range(len(y_hat))]
        data = {"users": users, "predicted": predicted, "y_hat": y_hat, "ground_truth": ground_truth}

        # evaluate
        nhl = self.new_hamming_loss(ground_truth, predicted)

        df = pd.DataFrame.from_dict(data)
        print(len(df))
        df.to_csv("lightning_logs/predict_max_"+ending_ts_date+'_n'+str(n)+'_max_n'+str(max_n)+'interval'+str(interval)+'_group_size'+str(group_size)+".csv", index=False)

        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_mae = torch.stack([x["mae"] for x in outputs]).mean()
        avg_rmse = torch.stack([x["rmse"] for x in outputs]).mean()

        self.log("test/loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/mae", avg_mae, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/rmse", avg_rmse, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/nhl", nhl, on_step=False, on_epoch=True, prog_bar=False)
        return nhl

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.0005)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.01)
        return parser

    def setup(self, stage=None):
        print("Loading datasets")
        self.train_dataset = MovieDataset("data/train_data_max_"+ending_ts_date+'_n'+str(n)+'_max_n'+str(max_n)+'interval'+str(interval)+'_group_size'+str(group_size)+".csv")
        self.val_dataset = MovieDataset("data/val_data_max_"+ending_ts_date+'_n'+str(n)+'_max_n'+str(max_n)+'interval'+str(interval)+'_group_size'+str(group_size)+".csv")
        self.test_dataset = MovieDataset("data/test_data_max_"+ending_ts_date+'_n'+str(n)+'_max_n'+str(max_n)+'interval'+str(interval)+'_group_size'+str(group_size)+".csv")
        print("Done")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
        )


if __name__ == "__main__":
    ave_training_time = 0.0
    ave_test_time = 0.0
    ave_nhl = 0.0
    log_rsl = open("output_mine_eff_231125_" + str(config.file_name) + ".csv", "a")
    for i in range(config.times_n):
        model = approx()

        ts = time.time()
        trainer = pl.Trainer(gpus=[config.device], max_epochs=config.max_epochs)
        trainer.fit(model)
        nhl_tuple = trainer.test(dataloaders=model.test_dataloader())
        nhl = float(nhl_tuple[0]['test/nhl'])
