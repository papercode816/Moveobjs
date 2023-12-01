import numpy as np
import pickle
from config import get_config

from scipy.sparse import csr_matrix
import random

config, unparsed = get_config()
group_size = config.group_size

max_n = config.max_n


def new_hamming_loss(y_true, y_pred, labels=None, sample_weight=None):
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)

    if sample_weight is None:
        weight_average = 1.
    else:
        weight_average = np.mean(sample_weight)

    if y_type.startswith('multilabel'):
        y_dif = np.sum(np.abs(y_true - y_pred), axis=1)
        y_t = np.sum(y_true,axis=1)
        sum_y = 0
        for i in range(y_dif.shape[0]):
            if y_t[i][0] != 0:
                sum_y += y_dif[i,0] / y_t[i,0]

        n_differences = sum_y / y_dif.shape[0]
        print(y_dif)
        print(y_t)
        print(n_differences)

        print("y_true.shape[0],y_true.shape[1]",y_true.shape[0],y_true.shape[1])
        return n_differences

def _check_targets(y_true, y_pred):
    type_true = 'multilabel-indicator'
    type_pred = 'multilabel-indicator'

    y_type = {type_true, type_pred}
    if y_type == {"binary", "multiclass"}:
        y_type = {"multiclass"}

    if len(y_type) > 1:
        raise ValueError("Classification metrics can't handle a mix of {0} "
                         "and {1} targets".format(type_true, type_pred))

    # We can't have more than one value on y_type => The set is no more needed
    y_type = y_type.pop()

    # No metrics support "multiclass-multioutput" format
    if (y_type not in ["binary", "multiclass", "multilabel-indicator"]):
        raise ValueError("{0} is not supported".format(y_type))
    if y_type.startswith('multilabel'):
        y_true = csr_matrix(y_true)
        y_pred = csr_matrix(y_pred)
        y_type = 'multilabel-indicator'

    return y_type, y_true, y_pred
ending_ts = config.ending_ts
n = config.n
interval = config.interval

ending_ts_date = config.ending_ts.split(' ')[0]
groups_f_exact = 'data/groups_exact_'+ending_ts_date+'_n'+str(n)+'_max_n'+str(max_n)+'interval'+str(interval)+'.pkl'
groups_exact = pickle.load(open(groups_f_exact, 'rb'))

p_trajs_f = 'data/p_trajs_'+ending_ts_date+'_n'+str(n)+'_max_n'+str(max_n)+'interval'+str(interval)+'.pkl'
p_trajs = pickle.load(open(p_trajs_f, 'rb'))

label_dict = {}
p_dict = {}

exact_p_dict = {}
p_pos_n = 0
for key in groups_exact.keys():
    if len(groups_exact[key]) >= group_size:
        for p in groups_exact[key]:
            if p not in exact_p_dict.keys():
                exact_p_dict[p] = []
            if key not in exact_p_dict[p]:
                exact_p_dict[p].append(key)

            if key not in label_dict.keys():
                tmp = len(label_dict)
                label_dict[key] = tmp
            if p not in p_dict.values():
                p_dict[len(p_dict)] = p
            p_pos_n+=1
print(p_pos_n)

neg_ratio = config.neg_ratio
p_neg_n = (neg_ratio+2)*p_pos_n
print(p_neg_n)

l = list(groups_exact.items())
random.shuffle(l)
groups_exact = dict(l)

for key in groups_exact.keys():
    if len(groups_exact[key]) < group_size:
        if p_neg_n <= 0:
            break
        for p in groups_exact[key]:
            if p not in exact_p_dict.keys():
                exact_p_dict[p] = []
            if p not in p_dict.values():
                p_dict[len(p_dict)] = p
            p_neg_n-=1

exact_label = np.zeros((len(p_dict), len(label_dict)))
for i in range(np.shape(exact_label)[0]):
    passenger_id = p_dict[i]
    groups_pi = exact_p_dict[passenger_id]
    for g in groups_pi:
        group_id = label_dict[g]
        exact_label[i][group_id] = 1
pos_ps_num = np.sum(exact_label,axis=1)
pos_ps_indexes = [n for n,x in enumerate(pos_ps_num) if x>0]
neg_ps_indexes = list(set(list(range(0, np.shape(exact_label)[0]))) - set(pos_ps_indexes))
sampled_neg_ps_indexes = np.random.choice(neg_ps_indexes, len(pos_ps_indexes)*neg_ratio, replace=False)
sampled_indexes = list(set(pos_ps_indexes).union(list(sampled_neg_ps_indexes)))
np.random.shuffle(sampled_indexes)

import pickle
exact_label_sampled = exact_label[sampled_indexes]
exact_label_sum = np.sum(exact_label_sampled,axis=0)
exact_label_ps_indexes = [n for n,x in enumerate(exact_label_sum) if x>0]
exact_label_sampled = exact_label_sampled[:,exact_label_ps_indexes]
groups_f = open('data/p_labels_exactbased_'+ending_ts_date+'_n'+str(n)+'_max_n'+str(max_n)+'interval'+str(interval)+'_group_size'+str(group_size)+'.pkl', 'wb')
pickle.dump(exact_label_sampled, groups_f, protocol=4)
groups_f.close()
print(len(exact_label_sampled))

passengers_sampled = [p_dict[ind] for ind in sampled_indexes]
p_dict_sampled = {}
p_trajs_sampled = {}
for p in passengers_sampled:
    p_dict_sampled[len(p_dict_sampled)] = p
    p_trajs_sampled[p] = p_trajs[p]
p_f = open('data/passengers_exactbased_'+ending_ts_date+'_n'+str(n)+'_max_n'+str(max_n)+'interval'+str(interval)+'_group_size'+str(group_size)+'.pkl', 'wb')
pickle.dump(p_dict_sampled, p_f, protocol=4)
p_f.close()
print(len(p_dict_sampled))

p_trajs_f = open('data/p_trajs_exactbased_'+ending_ts_date+'_n'+str(n)+'_max_n'+str(max_n)+'interval'+str(interval)+'_group_size'+str(group_size)+'.pkl', 'wb')
pickle.dump(p_trajs_sampled, p_trajs_f, protocol=4)
p_trajs_f.close()
print(len(p_trajs_sampled))