from models import *
from data.timeseries import load as load_timeseries_data, generate_data


def train():
    X_scaler, X_train, X_test, \
        X_train_unscaled, X_test_unscaled, \
            y_scaler, y_train, y_test, \
                y_train_unscaled, y_test_unscaled, \
                    y_scaler_train, y_scaler_test = generate_data(load_timeseries_data())
    
    
    model = TKAT(sequence_length=10, 
                 num_unknown_features=5, 
                 num_known_features=3, 
                 num_embedding=32, 
                 num_hidden=64, 
                 num_heads=4, 
                 n_ahead=5, 
                 use_tkan=True)