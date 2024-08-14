
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

class MinMaxScaler:
    def __init__(self, feature_axis=None, minmax_range=(0, 1)):
        """
        Initialize the MinMaxScaler.
        Args:
        feature_axis (int, optional): The axis that represents the feature dimension if applicable.
                                      Use only for 3D data to specify which axis is the feature axis.
                                      Default is None, automatically managed based on data dimensions.
        """
        self.feature_axis = feature_axis
        self.min_ = None
        self.max_ = None
        self.scale_ = None
        self.minmax_range = minmax_range # Default range for scaling (min, max)

    def fit(self, X):
        """
        Fit the scaler to the data based on its dimensionality.
        Args:
        X (np.array): The data to fit the scaler on.
        """
        if X.ndim == 3 and self.feature_axis is not None:  # 3D data
            axis = tuple(i for i in range(X.ndim) if i != self.feature_axis)
            self.min_ = np.min(X, axis=axis)
            self.max_ = np.max(X, axis=axis)
        elif X.ndim == 2:  # 2D data
            self.min_ = np.min(X, axis=0)
            self.max_ = np.max(X, axis=0)
        elif X.ndim == 1:  # 1D data
            self.min_ = np.min(X)
            self.max_ = np.max(X)
        else:
            raise ValueError("Data must be 1D, 2D, or 3D.")

        self.scale_ = self.max_ - self.min_
        return self

    def transform(self, X):
        """
        Transform the data using the fitted scaler.
        Args:
        X (np.array): The data to transform.
        Returns:
        np.array: The scaled data.
        """
        X_scaled = (X - self.min_) / self.scale_
        X_scaled = X_scaled * (self.minmax_range[1] - self.minmax_range[0]) + self.minmax_range[0]
        return X_scaled

    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        Args:
        X (np.array): The data to fit and transform.
        Returns:
        np.array: The scaled data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        """
        Inverse transform the scaled data to original data.
        Args:
        X_scaled (np.array): The scaled data to inverse transform.
        Returns:
        np.array: The original data scale.
        """
        X = (X_scaled - self.minmax_range[0]) / (self.minmax_range[1] - self.minmax_range[0])
        X = X * self.scale_ + self.min_
        return X

def to_tensor(data, dtype=torch.float32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.tensor(data, dtype=dtype).to(device)

def load_crypto(path="data.parquet"):
    df = pd.read_parquet(path)
    df = df[(df.index >= pd.Timestamp('2020-01-01')) & (df.index < pd.Timestamp('2023-01-01'))]
    assets = ['BTC', 'ETH', 'ADA', 'XMR', 'EOS', 'MATIC', 'TRX', 'FTM', 'BNB', 'XLM', 'ENJ', 'CHZ', 'BUSD', 'ATOM', 'LINK', 'ETC', 'XRP', 'BCH', 'LTC']
    df = df[[c for c in df.columns if 'quote asset volume' in c and any(asset in c for asset in assets)]]
    df.columns = [c.replace(' quote asset volume', '') for c in df.columns]
    return df

def load_known_inputs(df, columns=['hour', 'dayofweek']):
    return pd.DataFrame(
        index=df.index, 
        data=np.array([
            df.reset_index()['group'].apply(lambda x: (x[c])).values
            for c in columns
            
        ]).T, 
        columns = ['hour', 'dayofweek'])

def generate(df, sequence_length, n_ahead = 1, window=24*14):
    #Case without known inputs
    scaler_df = df.copy().shift(n_ahead).rolling(window).median()
    tmp_df = df.copy() / scaler_df
    tmp_df = tmp_df.iloc[window + n_ahead:].fillna(0.)
    scaler_df = scaler_df.iloc[window + n_ahead:].fillna(0.)

    def prepare_sequences(df, scaler_df, n_history, n_future):
        X, y, y_scaler = [], [], []

        # Iterate through the DataFrame to create sequences
        for i in range(n_history, len(df) - n_future + 1):
            # Extract the sequence of past observations
            X.append(df.iloc[i - n_history:i].values)
            # Extract the future values of the first column
            y.append(df.iloc[i:i + n_future,0:1].values)
            y_scaler.append(scaler_df.iloc[i:i + n_future,0:1].values)
        
        X, y, y_scaler = np.array(X), np.array(y), np.array(y_scaler)
        return X, y, y_scaler
    
    # Prepare sequences
    X, y, y_scaler = prepare_sequences(tmp_df, scaler_df, sequence_length, n_ahead)
    
    # Split the dataset into training and testing sets
    train_test_separation = int(len(X) * 0.8)
    X_train_unscaled, X_test_unscaled = X[:train_test_separation], X[train_test_separation:]
    y_train_unscaled, y_test_unscaled = y[:train_test_separation], y[train_test_separation:]
    y_scaler_train, y_scaler_test = y_scaler[:train_test_separation], y_scaler[train_test_separation:]
    
    # Generate the data
    X_scaler = MinMaxScaler(feature_axis=2)
    X_train = to_tensor(X_scaler.fit_transform(X_train_unscaled))
    X_test = to_tensor(X_scaler.transform(X_test_unscaled))
    
    y_scaler = MinMaxScaler(feature_axis=2)
    y_train = to_tensor(y_scaler.fit_transform(y_train_unscaled))
    y_test = to_tensor(y_scaler.transform(y_test_unscaled))
    
    y_train = to_tensor(y_train.reshape(y_train.shape[0], -1))
    y_test = to_tensor(y_test.reshape(y_test.shape[0], -1))    

    return X_scaler, X_train, X_test, \
        X_train_unscaled, X_test_unscaled, \
            y_scaler, y_train, y_test, \
                y_train_unscaled, y_test_unscaled, \
                    y_scaler_train, y_scaler_test


def generate_data_w_known_inputs(
        df: pd.DataFrame, 
        known_input_df: pd.DataFrame, 
        sequence_length: int, 
        n_ahead: int = 1,
        window: int = 24*14
) -> tuple:
    #Case without known inputs - fill with 0 the unknown features future values in X
    scaler_df = df.copy().shift(n_ahead).rolling(window).median()
    tmp_df = df.copy() / scaler_df
    tmp_df = tmp_df.iloc[window + n_ahead:].fillna(0.)
    scaler_df = scaler_df.iloc[window + n_ahead:].fillna(0.)
    tmp_known_input_df = known_input_df.iloc[window + n_ahead:].copy()
    
    def prepare_sequences(df, known_input_df, scaler_df, n_history, n_future):
        Xu, Xk, y, y_scaler = [], [], [], []
        
        # Iterate through the DataFrame to create sequences
        for i in range(n_history, len(df) - n_future + 1):
            # Extract the sequence of past observations
            Xu.append(np.concatenate((df.iloc[i - n_history:i].values, np.zeros((n_future, df.shape[1]))), axis=0))
            Xk.append(known_input_df.iloc[i - n_history:i+n_future].values)
            # Extract the future values of the first column
            y.append(df.iloc[i:i + n_future,0:1].values)
            y_scaler.append(scaler_df.iloc[i:i + n_future,0:1].values)
        
        Xu, Xk, y, y_scaler = np.array(Xu), np.array(Xk), np.array(y), np.array(y_scaler)
        return Xu, Xk, y, y_scaler
    
    # Prepare sequences
    Xu, Xk, y, y_scaler = prepare_sequences(tmp_df, tmp_known_input_df, scaler_df, sequence_length, n_ahead)

    X = np.concatenate((Xu, Xk), axis=-1)
    
    # Split the dataset into training and testing sets
    train_test_separation = int(len(X) * 0.8)
    X_train_unscaled, X_test_unscaled = X[:train_test_separation], X[train_test_separation:]
    y_train_unscaled, y_test_unscaled = y[:train_test_separation], y[train_test_separation:]
    y_scaler_train, y_scaler_test = y_scaler[:train_test_separation], y_scaler[train_test_separation:]
    
    # Generate the data
    X_scaler = MinMaxScaler(feature_axis=2)
    X_train = to_tensor(X_scaler.fit_transform(X_train_unscaled))
    X_test = to_tensor(X_scaler.transform(X_test_unscaled))
    
    y_scaler = MinMaxScaler(feature_axis=2)
    y_train = to_tensor(y_scaler.fit_transform(y_train_unscaled))
    y_test = to_tensor(y_scaler.transform(y_test_unscaled))
    
    y_train = to_tensor(y_train.reshape(y_train.shape[0], -1))
    y_test = to_tensor(y_test.reshape(y_test.shape[0], -1))
    
    return X_scaler, X_train, X_test, \
        X_train_unscaled, X_test_unscaled, \
            y_scaler, y_train, y_test, \
                y_train_unscaled, y_test_unscaled, \
                    y_scaler_train, y_scaler_test