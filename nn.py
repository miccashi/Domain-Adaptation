import csv
from tensorflow.keras import layers, Input, Model, regularizers
import matplotlib.pyplot as plt
import initialize
from initialize import Domain
import numpy as np
from data import SRCONLY, TGTONLY, ALL, WEIGHTED,PRED, LININT, AUGMENT, EAPLUS

from sklearn.metrics import mean_squared_error
import random


tgt_domain = [Domain.FEMALE, Domain.MALE, Domain.MIXED][0]
REG = AUGMENT

test_records = []
def plot_history(*historys):
    colors = 'brg'
    for i, (history, color) in enumerate(list(zip(historys, list(colors)))):
        history_dict = history.history
        mse = history_dict['mse']
        val_mse = history_dict['val_mse']
        epochs = range(1, len(mse) + 1)
        plt.plot(epochs, mse, color+'o', label='Training loss'+str(i))
        plt.plot(epochs, val_mse, color, label='Validation loss'+str(i))
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
def model(x_train, y_train, x_val, y_val, params):
    main_input = Input(shape=x_train[0].shape,dtype='float')
    # print(main_input)
    dim = main_input.shape[1]

    gen = main_input[:, 0:22]
    dom1 = main_input[:, 22:44]
    dom2 = main_input[:, 44:66]
    dom3 = main_input[:, 66:88]
    gen_y = layers.Dense(1, kernel_regularizer=regularizers.l2(params['general_regularizer']))(gen)
    dom1_y = layers.Dense(1, kernel_regularizer=regularizers.l2(params['SR1_regularizer']))(dom1)
    dom2_y = layers.Dense(1, kernel_regularizer=regularizers.l2(params['SR2_regularizer']))(dom2)
    dom3_y = layers.Dense(1, kernel_regularizer=regularizers.l2(params['SR3_regularizer']))(dom3)

    # gen_y = layers.Dense(params['neuron'],activation=params['activation'], kernel_regularizer=regularizers.l2(params['general_regularizer']))(gen)
    # dom1_y = layers.Dense(params['neuron'],activation=params['activation'], kernel_regularizer=regularizers.l2(params['SR1_regularizer']))(dom1)
    # dom2_y = layers.Dense(params['neuron'],activation=params['activation'],kernel_regularizer=regularizers.l2(params['SR2_regularizer']))(dom2)
    # dom3_y = layers.Dense(params['neuron'],activation=params['activation'],kernel_regularizer=regularizers.l2(params['SR3_regularizer']))(dom3)

    conc = layers.concatenate([gen_y, dom1_y, dom2_y, dom3_y], axis=-1)
    conc = layers.Dense(params['neuron'],activation=params['activation'])(conc)
    # conc = layers.Dropout(params['dropout'])(conc)
    main_output = layers.Dense(1)(conc)
    m = Model(inputs=main_input, outputs=main_output)
    # model.summary()

    m.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mse'])
    # history = m.fit(
    #     x_train, y_train,
    #     batch_size=64,
    #     epochs=10,
    #     validation_data=(x_val, y_val),
    #     callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)],
    #     verbose=0,
    # )
    history = m.fit(
        x_train, y_train,
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        validation_data=(x_val, y_val),
        # callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)],
        verbose=0,
    )
    # plot_history(history)
    return m

p = {'lr':(1/1000, 1/100, 5/100, 1/10, 5/10),
     'neuron':[4],
    'activation' : ['relu'],
    'regularizer':[100, 1, 0.01],
     # 'dropout':[.2, .3, .4, .5],
    'batch_size': [64],
     'epochs':[200]
}
P = []
for i in enumerate(range(10)):

    src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y, left_X, left_y = initialize.get_data(tgt_domain)
    d = REG(src_X, tgt_X, src_y, tgt_y, dev_X, dev_y, test_X, test_y, left_X, left_y)
    x, y, x_val, y_val = d.trn_X, d.trn_y, d.dev_X, d.dev_y
    for j in range(10):
        params = {'neuron':random.choice(p['neuron']),
                  'activation': random.choice(p['activation']),
                  'batch_size': random.choice(p['batch_size']),
                    'epochs':random.choice(p['epochs']),
                  # 'dropout': random.choice(p['dropout']),
                  'general_regularizer': random.choice([0.01]),
                  'SR1_regularizer': random.choice(p['regularizer']),
                  'SR2_regularizer': random.choice(p['regularizer']),
                  'SR3_regularizer':random.choice(p['regularizer']),
        }
        m = model(x,y,x_val, y_val,params)
        print('***************************************************')
        print(params)
        dev_mse = mean_squared_error(y_val, m.predict(x_val))
        test_mse = mean_squared_error(d.test_y, m.predict(d.test_X))
        print('TRN MSE:', mean_squared_error(d.trn_y, m.predict(d.trn_X)))

        print('DEV MSE:', dev_mse)
        print('TEST MSE:', test_mse)
        params['dev_mse'] = dev_mse
        params['test_mse'] = test_mse
        P.append(list(params.values()))
print(P)

