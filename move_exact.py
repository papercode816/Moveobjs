import sys
import os
parent_path = os.path.dirname(sys.path[0])
sys.path.append(parent_path)
print(sys.path)

from sqlalchemy import create_engine

import pandas as pd
import time
import numpy as np
import itertools
from config import get_config
def fx(x):
    return ((x - pd.to_datetime(starting_ts)) // (60 * interval))

if __name__ == '__main__':
    config, unparsed = get_config()

    starting_ts = config.starting_ts
    ending_ts = config.ending_ts
    min_n = config.min_n
    max_n = config.max_n
    n = config.n
    limit_d = config.limit_d

    times_n = config.times_n
    raw_n = config.raw_n
    output_groups_str = ''

    interval = config.interval

    log_rsl = open("output_mine_eff.csv", "a")
    log_rsl.write(str(config) + '\n')
    log_rsl.flush()

    if not os.path.exists('data/Nov20_simplified_' + str(raw_n) + '.csv'):
        engine = create_engine("mysql+pymysql://root:MTRdata@localhost:3306/MTRdata", echo=True)

        # data from Nov
        tablename = 'Nov20_simplified'
        conn = engine.connect()
        sql_mtr = 'select * from ' + tablename + ' order by ENTRY_TIME asc limit ' + str(limit_d)
        df_mtr = pd.read_sql(sql_mtr, conn)
        df_mtr.to_csv('data/Nov20_simplified_'+str(raw_n)+'.csv')

    df_mtr = pd.read_csv('data/Nov20_simplified_' + str(raw_n) + '.csv')
    df_mtr = df_mtr[df_mtr["TRAIN_ENTRY_STN"] != df_mtr["TXN_LOC"]]
    df_mtr = df_mtr[df_mtr["ENTRY_TIME"].notna()]
    df_mtr = df_mtr[df_mtr["EXIT_TIME"].notna()]

    df_mtr = df_mtr.sort_values(by="ENTRY_TIME", key=pd.to_datetime)
    df_mtr['ENTRY_TIME'] = pd.to_datetime(df_mtr['ENTRY_TIME'], unit='ns', origin='unix')
    df_mtr['EXIT_TIME'] = pd.to_datetime(df_mtr['EXIT_TIME'], unit='ns', origin='unix')
    df_mtr = df_mtr.loc[(df_mtr["ENTRY_TIME"] >= pd.to_datetime(starting_ts, unit='ns', origin='unix')) &
                        (df_mtr["ENTRY_TIME"] < pd.to_datetime(ending_ts, unit='ns', origin='unix'))]
    df_mtr = df_mtr.reset_index(drop=True)

    for t_n in range(times_n):
        interval = int(interval)
        count = 0
        total_group_size = 0

        ts = time.time()
        station_pair_dict = {}
        bin_list = []
        bin_o_list = []

        obj2trajs = {}
        obj2trajs_T = {}
        for index, row in df_mtr.iterrows():
            passenger_id = row["CSC_PHY_ID"]
            s1, s2 = row["TRAIN_ENTRY_STN"], row["TXN_LOC"]
            t1, t2 = row["ENTRY_TIME"], row["EXIT_TIME"]

            if passenger_id not in obj2trajs.keys():
                obj2trajs[passenger_id] = []

            obj2trajs[passenger_id].append( (s1,s2,t1,t2) )

            tmp_datetime = row['ENTRY_TIME'] - pd.to_datetime(starting_ts, unit='ns', origin='unix')
            # tmp_datetime = row['ENTRY_TIME'] - pd.to_datetime('2020-12-01 00:00:00', unit='ns', origin='unix')
            T1 = int(tmp_datetime.total_seconds() / 60 / interval)
            tmp_datetime = row['EXIT_TIME'] - pd.to_datetime(starting_ts, unit='ns', origin='unix')
            # tmp_datetime = row['EXIT_TIME'] - pd.to_datetime('2020-12-01 01:00:00', unit='ns', origin='unix')
            T2 = int(tmp_datetime.total_seconds() / 60 / interval)

            if passenger_id not in obj2trajs_T.keys():
                obj2trajs_T[passenger_id] = []
            obj2trajs_T[passenger_id].append( (s1,s2,T1,T2) )

        p_trajs = {}
        for obj in obj2trajs_T.keys():
            trajs = obj2trajs_T[obj]

            if len(trajs) < min_n or len(trajs) > max_n: continue
            if obj not in p_trajs.keys():
                p_trajs[obj] = trajs

            combinations_obj = itertools.combinations(list(range(len(trajs))), n)
            ids_n = list(combinations_obj)

            for ids in ids_n:
                trajs_sampled = np.asarray(trajs)[list(ids)]
                trajs_sampled_tuple = tuple()
                for e_tmp in trajs_sampled:
                    trajs_sampled_tuple = trajs_sampled_tuple + tuple(e_tmp)

                # 3. put passengers into bins
                if trajs_sampled_tuple not in station_pair_dict.keys():
                    station_pair_dict[trajs_sampled_tuple] = []
                if obj not in station_pair_dict[trajs_sampled_tuple]:
                    station_pair_dict[trajs_sampled_tuple].append(obj)

