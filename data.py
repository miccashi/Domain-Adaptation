from sklearn.metrics import mean_squared_error
import numpy as np
import initialize

CONSTANT_1 = [1, 10, 100, 1000]



class DataPrepare:
    def __init__(self, src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y, left_X, left_y):
        self.src_X, self.tgt_X, self.src_y, self.tgt_y, self.dev_X, self.dev_y, self.test_X, self.test_y, \
        self.left_X, self.left_y = src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y, left_X, left_y
        self.trn_X, self.trn_y = None, None


class EAPLUS(DataPrepare):
    def __init__(self, src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y, left_X, left_y, constants=None):
        super().__init__(src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y, left_X, left_y)
        if constants is None:
            C1, C2, C3, C4 = 1, 1, 1, 1
        else:
            C1, C2, C3, C4 = constants

        src_X_1 = np.hstack(
            (self.src_X[0]*C1, self.src_X[0]*C2, np.zeros((self.src_X[0].shape)), np.zeros((self.src_X[0].shape))))
        src_X_2 = np.hstack(
            (self.src_X[1]*C1, np.zeros((self.src_X[1].shape)), self.src_X[1]*C3, np.zeros((self.src_X[1].shape))))

        self.src_X = np.vstack((src_X_1, src_X_2))
        self.src_y = np.vstack(tuple(self.src_y))

        self.tgt_X = np.hstack((self.tgt_X*C1, np.zeros((self.tgt_X.shape)), np.zeros((self.tgt_X.shape)), self.tgt_X*C4))
        self.dev_X = np.hstack((self.dev_X*C1, np.zeros((self.dev_X.shape)), np.zeros((self.dev_X.shape)), self.dev_X*C4))
        self.test_X = np.hstack(
            (self.test_X*C1, np.zeros((self.test_X.shape)), np.zeros((self.test_X.shape)), self.test_X*C4))

        self.left_X1 = np.hstack(
            (np.zeros((self.left_X.shape)), np.zeros((self.left_X.shape)), self.left_X*C3, -self.left_X*C4))
        self.left_X2 = np.hstack(
            (np.zeros((self.left_X.shape)), self.left_X*C2, np.zeros((self.left_X.shape)), -self.left_X*C4))

        # print(self.src_X.shape, self.tgt_X.shape, self.left_X1.shape, self.left_X2.shape)
        self.trn_X = np.vstack((self.src_X, self.tgt_X, self.left_X1, self.left_X2))
        self.trn_y = np.vstack((self.src_y, self.tgt_y, np.zeros((self.left_y.shape)), np.zeros((self.left_y.shape))))


class AUGMENT(DataPrepare):
    def __init__(self, src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y, left_X, left_y,constants=None):
        super().__init__(src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y, left_X, left_y)
        if constants is None: C1, C2,C3,C4 = 1,1,1,1
        else: C1, C2, C3, C4 = constants

        src_X_1 = np.hstack(
            (self.src_X[0]*C1, self.src_X[0]*C2, np.zeros((self.src_X[0].shape)), np.zeros((self.src_X[0].shape))))
        src_X_2 = np.hstack(
            (self.src_X[1]*C1, np.zeros((self.src_X[1].shape)), self.src_X[1]*C3, np.zeros((self.src_X[1].shape))))

        self.src_X = np.vstack((src_X_1, src_X_2))
        self.src_y = np.vstack(tuple(self.src_y))

        self.tgt_X = np.hstack((self.tgt_X*C1, np.zeros((self.tgt_X.shape)), np.zeros((self.tgt_X.shape)), self.tgt_X*C4))
        self.dev_X = np.hstack((self.dev_X*C1, np.zeros((self.dev_X.shape)), np.zeros((self.dev_X.shape)), self.dev_X*C4))
        self.test_X = np.hstack(
            (self.test_X*C1, np.zeros((self.test_X.shape)), np.zeros((self.test_X.shape)), self.test_X*C4))

        self.trn_X = np.vstack((self.src_X, self.tgt_X))
        self.trn_y = np.vstack((self.src_y, self.tgt_y))


class SRCONLY(DataPrepare):
    def __init__(self, src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y, left_X, left_y):
        super().__init__(src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y, left_X, left_y)
        self.src_X = np.vstack(tuple(self.src_X))
        self.src_y = np.vstack(tuple(self.src_y))

        self.trn_X, self.trn_y = self.src_X, self.src_y


class TGTONLY(DataPrepare):
    def __init__(self, src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y, left_X, left_y):
        super().__init__(src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y, left_X, left_y)
        self.src_X = np.vstack(tuple(self.src_X))
        self.src_y = np.vstack(tuple(self.src_y))
        self.trn_X, self.trn_y = self.tgt_X, self.tgt_y


class ALL(DataPrepare):
    def __init__(self, src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y, left_X, left_y):
        super().__init__(src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y, left_X, left_y)
        self.src_X = np.vstack(tuple(self.src_X))
        self.src_y = np.vstack(tuple(self.src_y))
        self.trn_X = np.vstack((self.src_X, self.tgt_X))
        self.trn_y = np.vstack((self.src_y, self.tgt_y))


class WEIGHTED(DataPrepare):
    def __init__(self, src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y, left_X, left_y):
        super().__init__(src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y, left_X, left_y)
        self.src_X = np.vstack(tuple(self.src_X))
        self.src_y = np.vstack(tuple(self.src_y))

        weight = round(len(self.src_X) / len(self.tgt_X))
        # print(weight)
        trn_X, trn_y = self.src_X, self.src_y
        for _ in range(weight):
            trn_X = np.vstack((trn_X, self.tgt_X))
            trn_y = np.vstack((trn_y, self.tgt_y))
        self.trn_X, self.trn_y = trn_X, trn_y

class PRED(SRCONLY):
    def __init__(self, src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y, left_X, left_y):
        super().__init__(src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y, left_X, left_y)

class LININT(SRCONLY):
    def __init__(self, src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y, left_X, left_y):
        super().__init__(src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y, left_X, left_y)
