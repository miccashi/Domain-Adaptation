import initialize
from initialize import Domain
import numpy as np
from data import SRCONLY, TGTONLY, ALL, WEIGHTED,PRED, LININT, AUGMENT, EAPLUS
from sklearn.linear_model import Ridge
import evaluate
from tqdm import tqdm
def test(d, tgt_domain):
    # print('Trn Size', d.trn_X.shape, d.trn_y.shape, 'Test Size', d.test_X.shape, d.test_y.shape)

    alphas = [0.001, 0.1, 1, 10, 100, 1000]
    # alphas = [1]
    models = [Ridge(alpha=a) for a in alphas]

    trn_mses, dev_mses, test_mses = [], [], []
    for i, model in enumerate(models):
        trn_mse, dev_mse, test_mse = evaluate.predict(model, tgt_domain, d.trn_X, d.trn_y, d.dev_X, d.dev_y, d.test_X,
                                                      d.test_y)
        trn_mses.append(trn_mse)
        dev_mses.append(dev_mse)
        test_mses.append(test_mse)
    return trn_mses, dev_mses, test_mses, models

tgt_domains = [Domain.MALE]

for tgt_domain in tgt_domains:
    print('************************* ', tgt_domain, ' *************************')
    CONSTANTS_1 = [1,5,10,100]
    CONSTANTS_2 = [1,5,10]
    CONSTANTS_3 = [1,5,10]
    CONSTANTS_4 = [1]
    record = []
    for i in tqdm(range(200)):
        src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y, left_X, left_y = initialize.get_data(tgt_domain)
        constant_record = []
        for C1 in CONSTANTS_1:
            for C2 in CONSTANTS_2:
                for C3 in CONSTANTS_3:
                    for C4 in CONSTANTS_4:
                        d = AUGMENT(src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y,left_X, left_y,[C1,C2,C3,C4])
                        trn_mses, dev_mses, test_mses, models = test(d, tgt_domain)
                        constant_record.append([min(trn_mses), min(dev_mses),min(test_mses)])
        record.append(constant_record)

    avg_record = np.average(record, axis=0)

    min_dev, min_test = min(avg_record[:,1]), min(avg_record[:,2])
    k = 0
    for i in range(avg_record.shape[0]):
        for C1 in tqdm(CONSTANTS_1):
            for C2 in CONSTANTS_2:
                for C3 in CONSTANTS_3:
                    for C4 in CONSTANTS_4:

                        dev_color = test_color = '\033[0;30m'
                        if avg_record[k][1] == min_dev: dev_color = '\033[1;34m'
                        if avg_record[k][2] == min_test: test_color = '\033[1;31m'
                        print('K',k,'C:', C1,C2,C3,C4, ': Train MSE:', avg_record[k][0], dev_color, ' Dev MSE:', avg_record[k][1], test_color,
                              ' Test MSE:', avg_record[k][2], '\033[0m')
                        k += 1
                        if k == avg_record.shape[0]:
                            break

    assert False
#