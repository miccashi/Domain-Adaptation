import initialize
from initialize import Domain
import numpy as np
from data import SRCONLY, TGTONLY, ALL, WEIGHTED,PRED, LININT, AUGMENT, EAPLUS
from sklearn.linear_model import Ridge
import evaluate
from tqdm import tqdm
def test(d, tgt_domain):
    # print('Trn Size', d.trn_X.shape, d.trn_y.shape, 'Test Size', d.test_X.shape, d.test_y.shape)

    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
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
import csv
def write_file(best_models, file_path):
    file = open(file_path, 'a', encoding='utf-8', newline='')
    csv_writer = csv.writer(file, dialect='excel')
    for i in range(len(best_models)):
        csv_writer.writerow(best_models[i].coef_[0])

tgt_domains = [Domain.FEMALE, Domain.MALE, Domain.MIXED]

for tgt_domain in tgt_domains:

    best_records = []
    best_performance = float('inf')
    best_models = None
    for epoch in tqdm(range(100)):

        src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y,left_X, left_y= initialize.get_data(tgt_domain)
        # REGs = [SRCONLY, TGTONLY, ALL, WEIGHTED, PRED, LININT, AUGMENT, EAPLUS]
        REGs = [AUGMENT]

        reg_records = []
        for REG in REGs:

            # print('\n************************',REG.__name__,'*******************************')

            d = REG(src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y,left_X, left_y)
            trn_mses, dev_mses, test_mses, models = test(d, tgt_domain)

            if REG == PRED:
                pred_trn_y = models[dev_mses.index(min(dev_mses))].predict(d.tgt_X)
                pred_dev_y = models[dev_mses.index(min(dev_mses))].predict(d.dev_X)
                pred_test_y = models[dev_mses.index(min(dev_mses))].predict(d.test_X)
                d.trn_X, d.trn_y = np.c_[d.tgt_X, pred_trn_y], d.tgt_y
                d.dev_X = np.c_[d.dev_X, pred_dev_y]
                d.test_X = np.c_[d.test_X, pred_test_y]
                trn_mses, dev_mses, test_mses, models = test(d, tgt_domain)
            elif REG == LININT:
                src_trn_y = models[dev_mses.index(min(dev_mses))].predict(d.tgt_X)
                src_dev_y = models[dev_mses.index(min(dev_mses))].predict(d.dev_X)
                src_test_y = models[dev_mses.index(min(dev_mses))].predict(d.test_X)

                d = TGTONLY(src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y,left_X, left_y)
                _, dev_mses, _, models = test(d, tgt_domain)
                tgt_trn_y = models[dev_mses.index(min(dev_mses))].predict(d.tgt_X)
                tgt_dev_y = models[dev_mses.index(min(dev_mses))].predict(d.dev_X)
                tgt_test_y = models[dev_mses.index(min(dev_mses))].predict(d.test_X)

                w1 = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3]

                y_preds = [[src_dev_y[i] + (tgt_dev_y[i] - src_dev_y[i]) * w for i in range(len(src_dev_y))] for w in w1]
                dev_mses = [evaluate.get_MSE(tgt_domain, dev_y, y_pred) for y_pred in y_preds]
                y_preds = [[src_test_y[i] + (tgt_test_y[i] - src_test_y[i]) * w for i in range(len(src_test_y))] for w in w1]
                test_mses = [evaluate.get_MSE(tgt_domain, test_y, y_pred) for y_pred in y_preds]

                w2 = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 2]
                y_preds = [[tgt_dev_y[i] + (src_dev_y[i] - tgt_dev_y[i]) * w for i in range(len(tgt_dev_y))] for w in w2]
                dev_mses  += [evaluate.get_MSE(tgt_domain, dev_y, y_pred) for y_pred in y_preds]
                y_preds = [[tgt_test_y[i] + (src_test_y[i] - tgt_test_y[i]) * w for i in range(len(tgt_test_y))] for w in w2]
                test_mses += [evaluate.get_MSE(tgt_domain, test_y, y_pred) for y_pred in y_preds]

                trn_mses = [-1 for i in range(len(test_mses))]

            # for i in range(len(trn_mses)):
            #     dev_color = test_color = '\033[0;30m'
            #     if dev_mses[i] == min(dev_mses): dev_color = '\033[1;34m'
            #     if test_mses[i] == min(test_mses): test_color = '\033[1;31m'
            #     print('Turn', i,': Train MSE:', trn_mses[i], dev_color,' Dev MSE:', dev_mses[i], test_color,' Test MSE:', test_mses[i],'\033[0m')

            reg_records.append([min(trn_mses), min(dev_mses), min(test_mses)])
            if min(test_mses) < best_performance:
                best_performance = min(test_mses)
                best_models = models
                best_records = [trn_mses, dev_mses, test_mses]


    print('************************* ', tgt_domain, ' *************************')
    trn_mses, dev_mses, test_mses = best_records
    for i in range(len(trn_mses)):
        dev_color = test_color = '\033[0;30m'
        if dev_mses[i] == min(dev_mses): dev_color = '\033[1;34m'
        if test_mses[i] == min(test_mses): test_color = '\033[1;31m'
        print('Turn', i,': Train MSE:', trn_mses[i], dev_color,' Dev MSE:', dev_mses[i], test_color,' Test MSE:', test_mses[i],'\033[0m','alpha:',models[i].alpha)

    write_file(best_models,str(tgt_domain)+'_hyperParameter.csv')

    # for j,model in enumerate(best_models):
    #     print('****************',j,'*****************')
    #     print(model.alpha)
    #     print(model.coef_)
    # assert False
    print()
    print()

