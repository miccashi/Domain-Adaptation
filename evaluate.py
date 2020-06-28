import pandas as pd
import initialize
from initialize import Domain
import numpy as np
from sklearn.metrics import mean_squared_error


def predict(model, tgt_domain, trn_X, trn_y, dev_X, dev_y, test_X, test_y):
    model.fit(trn_X, trn_y)
    trn_mse = get_MSE(tgt_domain, model.predict(trn_X), trn_y, train=True)
    dev_mse = get_MSE(tgt_domain, model.predict(dev_X), dev_y)
    test_mse = get_MSE(tgt_domain, model.predict(test_X), test_y)

    return trn_mse, dev_mse, test_mse

def get_real_y(file_path):
    df = pd.read_csv(file_path, sep=',')
    df = df.sample(frac=1, random_state=2020)
    return df.values[:, -1:]


female_real_y = get_real_y(initialize.female_path)
male_real_y = get_real_y(initialize.male_path)
mixed_real_y = get_real_y(initialize.mixed_path)
female_trn_num = int(initialize.TRN_SIZE * len(female_real_y))
male_trn_num = int(initialize.TRN_SIZE * len(male_real_y))
mixed_trn_num = int(initialize.TRN_SIZE * len(mixed_real_y))


def get_train_loss(tgt, y_pred, y_true):
    # y_pred = y_pred.data.numpy()
    # y_true = y_true.data.numpy()
    trn_size = initialize.TRN_SIZE
    if tgt == Domain.FEMALE:
        
        y_pred = np.vstack((initialize.feature_unnormalize(male_real_y, y_pred[:male_trn_num]),
                            initialize.feature_unnormalize(mixed_real_y, y_pred[male_trn_num:-trn_size]),
                            initialize.feature_unnormalize(female_real_y, y_pred[-trn_size:])))
        y_true = np.vstack((initialize.feature_unnormalize(male_real_y, y_true[:male_trn_num]),
                            initialize.feature_unnormalize(mixed_real_y, y_true[male_trn_num:-trn_size]),
                            initialize.feature_unnormalize(female_real_y, y_true[-trn_size:])))

    elif tgt == Domain.MALE:
        y_pred = np.vstack((initialize.feature_unnormalize(female_real_y, y_pred[:female_trn_num]),
                            initialize.feature_unnormalize(mixed_real_y, y_pred[female_trn_num:-trn_size]),
                            initialize.feature_unnormalize(male_real_y, y_pred[-trn_size:])))
        y_true = np.vstack((initialize.feature_unnormalize(female_real_y, y_true[:female_trn_num]),
                            initialize.feature_unnormalize(mixed_real_y, y_true[female_trn_num:-trn_size]),
                            initialize.feature_unnormalize(male_real_y, y_true[-trn_size:])))
    elif tgt == Domain.MIXED:
        y_pred = np.vstack((initialize.feature_unnormalize(female_real_y, y_pred[:female_trn_num]),
                            initialize.feature_unnormalize(male_real_y, y_pred[female_trn_num:-trn_size]),
                            initialize.feature_unnormalize(mixed_real_y, y_pred[-trn_size:])))
        y_true = np.vstack((initialize.feature_unnormalize(female_real_y, y_true[:female_trn_num]),
                            initialize.feature_unnormalize(male_real_y, y_true[female_trn_num:-trn_size]),
                            initialize.feature_unnormalize(mixed_real_y, y_true[-trn_size:])))

    return ((y_pred - y_true) ** 2).mean()


def get_test_loss(tgt, y_pred, y_true):
    # y_pred = y_pred.data.numpy()
    # y_true = y_true.data.numpy()
    if tgt == Domain.FEMALE:
        y_pred = initialize.feature_unnormalize(female_real_y, y_pred)
        y_true = initialize.feature_unnormalize(female_real_y, y_true)
    elif tgt == Domain.MALE:
        y_pred = initialize.feature_unnormalize(male_real_y, y_pred)
        y_true = initialize.feature_unnormalize(male_real_y, y_true)
    elif tgt == Domain.MIXED:
        y_pred = initialize.feature_unnormalize(mixed_real_y, y_pred)
        y_true = initialize.feature_unnormalize(mixed_real_y, y_true)
    return ((y_pred - y_true) ** 2).mean()


def get_MSE(tgt, y_pred, y_true, train=False):
    if initialize.NORMAL:
        if train:
            return get_train_loss(tgt, y_pred, y_true)
        return get_test_loss(tgt, y_pred, y_true)
    else:
        return mean_squared_error(y_true, y_pred)