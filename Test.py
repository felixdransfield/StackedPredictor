import os
import json

from Models.LSTMAutoEncoder.LSTMAutoEncoder import LSTMAutoEncoder
from Models.LSTMAutoEncoder.Utils import process_data, flatten
from Models.MetaClassifier.DecisionMaker import DecisionMaker
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from pylab import rcParams

from keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.metrics import auc, roc_curve

from numpy.random import seed
seed(7)

rcParams['figure.figsize'] = 8, 6
LABELS = ["0","1"]

from Utils.Data import scale, impute

def main():
    configs = json.load(open('Configuration.json', 'r'))

    epochs = configs['training']['epochs']
    grouping = configs['data']['grouping']
    dynamic_features = configs['data']['dynamic_columns']

    outcomes = configs['data']['classification_outcome']
    lookback = configs['data']['batch_size']
    timeseries_path = configs['paths']['data_path']

    ##read, impute and scale dataset
    non_smotedtime_series = pd.read_csv(timeseries_path + "TimeSeriesAggregatedUpto0.csv")
    non_smotedtime_series[dynamic_features] = impute(non_smotedtime_series, dynamic_features)
    normalized_timeseries = scale(non_smotedtime_series, dynamic_features)
    normalized_timeseries.insert(0, grouping, non_smotedtime_series[grouping])

    ##start working per outcome
    for outcome in outcomes:
        decision_maker = DecisionMaker()

        X_train_y0, y_train1, X_valid_y0, X_valid_y1, X_valid, y_val1, X_test, y_test1, timesteps, n_features=\
            process_data(normalized_timeseries, non_smotedtime_series, outcome, grouping, non_smotedtime_series[grouping], lookback)


if __name__ == '__main__':
    main()
