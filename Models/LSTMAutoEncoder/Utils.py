import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import Models
SEED = 123 #used to help randomly select the data points

def process_data(dynamic_series, full_series, outcome, grouping, lookback):
    # train/test and validation sets
    dynamic_series.insert(len(dynamic_series.columns), outcome, full_series[outcome])
    dynamic_timeseries = curve_shift(dynamic_series, grouping, outcome, shift_by=lookback - 1)
    X_cols = (dynamic_timeseries.columns).tolist()
    X_cols.remove(outcome)
    X_cols.remove(grouping)

    input_X = dynamic_timeseries.loc[:,
              dynamic_timeseries.columns.isin(X_cols)].values  # converts the df to a numpy array
    input_y = dynamic_timeseries[outcome].values

    n_features = input_X.shape[1]  # number of features

    X, y = temporalize(X=input_X, y=input_y, lookback=lookback)

    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.33,
                                                        random_state=SEED, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.33,
                                                          random_state=SEED, stratify=y_train)

    X_train_y0 = X_train[y_train == 0]
    X_train_y1 = X_train[y_train == 1]
    X_valid_y0 = X_valid[y_valid == 0]
    X_valid_y1 = X_valid[y_valid == 1]

    X_train = X_train.reshape(X_train.shape[0], lookback, n_features)
    X_train_y0 = X_train_y0.reshape(X_train_y0.shape[0], lookback, n_features)
    X_train_y1 = X_train_y1.reshape(X_train_y1.shape[0], lookback, n_features)
    X_valid = X_valid.reshape(X_valid.shape[0], lookback, n_features)
    X_valid_y0 = X_valid_y0.reshape(X_valid_y0.shape[0], lookback, n_features)
    X_valid_y1 = X_valid_y1.reshape(X_valid_y1.shape[0], lookback, n_features)
    X_test = X_test.reshape(X_test.shape[0], lookback, n_features)

    scaler = StandardScaler().fit(flatten(X_train_y0))
    X_train_y0_scaled = Models.LSTMAutoEncoder.Utils.scale(X_train_y0, scaler)

    a = flatten(X_train_y0_scaled)
    print('colwise mean', np.mean(a, axis=0).round(6))
    print('colwise variance', np.var(a, axis=0))

    X_valid_scaled = Models.LSTMAutoEncoder.Utils.scale(X_valid, scaler)
    X_valid_y0_scaled = Models.LSTMAutoEncoder.Utils.scale(X_valid_y0, scaler)
    X_test_scaled = Models.LSTMAutoEncoder.Utils.scale(X_test, scaler)

    timesteps = X_train_y0_scaled.shape[1]  # equal to the lookback
    n_features = X_train_y0_scaled.shape[2]  # 59
    return X_train_y0_scaled, X_valid_y0_scaled, X_valid_scaled, y_valid, X_test_scaled, y_test, timesteps, n_features

def temporalize(X, y, lookback):
    '''
    Inputs
    X         A 2D numpy array ordered by time of shape:
              (n_observations x n_features)
    y         A 1D numpy array with indexes aligned with
              X, i.e. y[i] should correspond to X[i].
              Shape: n_observations.
    lookback  The window size to look back in the past
              records. Shape: a scalar.

    Output
    output_X  A 3D numpy array of shape:
              ((n_observations-lookback-1) x lookback x
              n_features)
    output_y  A 1D array of shape:
              (n_observations-lookback-1), aligned with X.
    '''
    output_X = []
    output_y = []
    for i in range(len(X) - lookback - 1):
        t = []
        for j in range(1, lookback + 1):
            # Gather the past records upto the lookback period
            t.append(X[[(i + j + 1)], :])
        output_X.append(t)
        output_y.append(y[i + lookback + 1])
    return np.squeeze(np.array(output_X)), np.array(output_y)


def flatten ( X ) :
    '''
    Flatten a 3D array.

    Input
    X            A 3D array for lstm, where the array is sample x timesteps x features.

    Output
    flattened_X  A 2D array, sample x features.
    '''
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]) :
        flattened_X[i] = X[i, (X.shape[1] - 1), :]
    return (flattened_X)


def scale ( X, scaler ) :
    '''
    Scale 3D array.

    Inputs
    X            A 3D array for lstm, where the array is sample x timesteps x features.
    scaler       A scaler object, e.g., sklearn.preprocessing.StandardScaler, sklearn.preprocessing.normalize

    Output
    X            Scaled 3D array.
    '''
    for i in range(X.shape[0]) :
        X[i, :, :] = scaler.transform(X[i, :, :])

    return X


sign = lambda x : (1, -1)[x < 0]


def curve_shift ( df, grouping, outcome, shift_by ) :
    for patient_id in df[grouping]:
        if ((df.loc[df[grouping]==patient_id, outcome]).tolist())[0] ==1:
            patientFrame = df.loc[df[grouping]==patient_id, outcome]
            patientFrame.iloc[0:shift_by] = 0
            df.loc[df[grouping]==patient_id, outcome] = [x for x in patientFrame.values]


    return df

