import random
import coral
import pandas as pd
from enum import Enum
import numpy as np
from tensorflow.keras.utils import to_categorical
import tca
from sklearn.model_selection import train_test_split
female_path = 'FEMALE.csv'
male_path = 'MALE.csv'
mixed_path = 'MIXED.csv'
NORMAL = False
SPLIT_SIZE = 0.7
TCA = False
CORAL = False
TRN_SIZE = 100

class Domain(Enum):
    FEMALE = 1
    MALE = 2
    MIXED = 3

def feature_normalize(data):
    mu = np.mean(data,axis=0) # 均值
    std = np.std(data,axis=0) # 标准差
    return (data - mu)/std

def feature_unnormalize(data, arr):
    mu = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return arr * std + mu



def read_file(file_path):

    df = pd.read_csv(file_path, sep=',')
    df = df.sample(frac = 1)

    # one hot encoding:
    year = to_categorical(df.values[:, 0])[:, 1:]
    fsm = df.values[:, 1:2]
    vr1_band = df.values[:, 2:3] #155

    vr_band = to_categorical(
        np.where(df.values[:, 3] == 0, random.choice([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]), df.values[:, 3]))[
              :, 1:] #116
    ethnic = to_categorical(df.values[:, 4])[:, 1:] #161
    denomination = to_categorical(df.values[:, 5])[:, 1:] #159

    y = df.values[:, -1:]


    # normalize
    if NORMAL:
        # print('Normalize...')
        year = feature_normalize(year)
        fsm = feature_normalize(fsm)
        vr1_band = feature_normalize(vr1_band)
        vr_band = feature_normalize(vr_band)
        ethnic = feature_normalize(ethnic)
        denomination = feature_normalize(denomination)
        y = feature_normalize(y)
    # else:
        # print('Not Normalize...')
    X = np.hstack((year, fsm, vr1_band, vr_band, ethnic, denomination))

    return X, y



# import csv
# def write_file(X, file_path):
#     file = open(file_path, 'a', encoding='utf-8', newline='')
#     csv_writer = csv.writer(file, dialect='excel')
#     for i in range(X.shape[0]):
#         csv_writer.writerow(X[i])
#     assert False
#
def split_data(data):
    split_size = SPLIT_SIZE
    trn, test = data[:int(split_size * len(data))], data[int(split_size * len(data)):]
    dev, test = test[:int(0.5 * len(test))], test[int(0.5 * len(test)):]
    return trn, dev, test
#

def get_sampled_data(data, size):
    # random.seed=1
    random.shuffle(data)

    use_data = data[:min(size, len(data))]
    left_data = data[min(size, len(data)):]
    return use_data, left_data


def get_data(tgt_domain):
    c = coral.CORAL()
    t = tca.TCA(dim=20,lamb=1, gamma=1)
    female_X, female_y = read_file(female_path)
    male_X, male_y = read_file(male_path)
    mixed_X, mixed_y = read_file(mixed_path)
    # print('\nData Shape:')
    # print('Female:',female_X.shape)
    # print('Male:',male_X.shape)
    # print('Mixed:',mixed_X.shape)

    if tgt_domain == Domain.FEMALE:
        if CORAL:
            # print('CORAL')
            male_X, mixed_X = c.fit(male_X, female_X), c.fit(mixed_X, female_X)
        if TCA:
            male_X, female_X = t.fit(male_X, female_X)
            mixed_X, female_X_2 = t.fit(mixed_X, female_X)
            female_X = np.vstack((female_X, female_X_2))
            female_y = np.vstack((female_y, female_y))
    elif tgt_domain == Domain.MALE:
        if CORAL:
            female_X, mixed_X = c.fit(female_X, male_X), c.fit(mixed_X, male_X)
        if TCA:
            # ????????????????????????????????????????
            new_X = np.vstack((female_X, mixed_X))
            new_X, male_X = t.fit(new_X, male_X)
            female_X = new_X[:female_X.shape[0]]
            mixed_X = new_X[-male_X.shape[0]:]
            # print(male_X.shape)
            # assert False
    elif tgt_domain == Domain.MIXED:
        if CORAL:
            # print('CORAL')
            female_X, male_X = c.fit(female_X, mixed_X), c.fit(male_X, mixed_X)
        if TCA:
            female_X, mixed_X = t.fit(female_X, mixed_X)
            male_X, mixed_X_2 = t.fit(male_X, mixed_X)
            mixed_X = np.vstack((mixed_X, mixed_X_2))
            mixed_y = np.vstack((mixed_y, mixed_y))

    female = list(zip(female_X, female_y))
    male = list(zip(male_X, male_y))
    mixed = list(zip(mixed_X, mixed_y))


    female_trn, female_dev, female_test = split_data(female)
    male_trn, male_dev, male_test = split_data(male)
    mixed_trn, mixed_dev, mixed_test = split_data(mixed)
    # print('\nSplit Data to trn/dev/test...')
    # print('female:',len(female_trn), len(female_dev), len(female_test))
    # print('male:',len(male_trn), len(male_dev), len(male_test))
    # print('mixed',len(mixed_trn), len(mixed_dev), len(mixed_test))


    if tgt_domain == Domain.FEMALE:

        src_0 = male_trn
        src_1 = mixed_trn
        src_X_0, src_y_0 = zip(*src_0)
        src_X_1, src_y_1 = zip(*src_1)

        src_X = [np.array(src_X_0), np.array(src_X_1)]
        src_y = [np.array(src_y_0), np.array(src_y_1)]

        tgt, left = get_sampled_data(female_trn, TRN_SIZE)
        dev, left_dev = get_sampled_data(female_dev, 100)
        test = female_test

        assert len(left) + len(tgt) + len(dev) + len(test) == len(female) - len(female_dev) + 100
        assert src_X[0].shape[0] == int(SPLIT_SIZE*len(male))
        assert src_X[1].shape[0] == int(SPLIT_SIZE*len(mixed))

        if len(left)==0:
            left = left_dev



    elif tgt_domain == Domain.MALE:

        src_0 = female_trn
        src_1 = mixed_trn
        src_X_0, src_y_0 = zip(*src_0)
        src_X_1, src_y_1 = zip(*src_1)
        src_X = [np.array(src_X_0), np.array(src_X_1)]
        src_y = [np.array(src_y_0), np.array(src_y_1)]

        tgt, left = get_sampled_data(male_trn, TRN_SIZE)
        dev, left_dev = get_sampled_data(male_dev, 100)
        test =  male_test

        assert len(left) + len(tgt) + len(dev) + len(test) == len(male) - len(male_dev) +100
        assert src_X[0].shape[0] == int(SPLIT_SIZE*len(female))
        assert src_X[1].shape[0] == int(SPLIT_SIZE*len(mixed))
        if len(left)==0: left = left_dev

    elif tgt_domain == Domain.MIXED:
        src_0 = female_trn
        src_1 = male_trn
        src_X_0, src_y_0 = zip(*src_0)
        src_X_1, src_y_1 = zip(*src_1)
        src_X = [np.array(src_X_0), np.array(src_X_1)]
        src_y = [np.array(src_y_0), np.array(src_y_1)]



        tgt, left = get_sampled_data(mixed_trn, TRN_SIZE)
        dev, left_dev = get_sampled_data(mixed_dev, 100)
        test = mixed_test

        # print(len(tgt), len(left), len(dev),len(test))
        # print(src_X[0].shape[0],int(SPLIT_SIZE*len(female)))
        assert len(left) + len(tgt) + len(dev) + len(test) == len(mixed)-len(mixed_dev) + 100
        assert src_X[0].shape[0] == int(SPLIT_SIZE*len(female))
        assert src_X[1].shape[0] == int(SPLIT_SIZE*len(male))
        if len(left)==0: left = left_dev

    tgt_X, tgt_y = zip(*tgt)
    dev_X, dev_y = zip(*dev)
    test_X, test_y = zip(*test)
    left_X , left_y = np.array([]),np.array([])
    if len(left)>0:
        left_X, left_y = zip(*left)

    tgt_X, tgt_y = np.array(tgt_X), np.array(tgt_y)
    dev_X, dev_y = np.array(dev_X), np.array(dev_y)
    test_X, test_y = np.array(test_X), np.array(test_y)
    left_X, left_y = np.array(left_X), np.array(left_y)
    # print('\nSummary:')
    # print('tgt domain:',tgt_domain)
    # print('src',[src_X[0].shape, src_X[1].shape],'tgt trn:', tgt_X.shape,'dev:', dev_X.shape, 'test:', test_X.shape, 'left:',left_X.shape)
    # print('***************************************************************************************************************')
    return src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y, left_X, left_y



















