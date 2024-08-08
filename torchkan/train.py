from torch import nn
import torch.optim

from models import *
from data.timeseries import load as load_timeseries_data, generate_data


def train():
    X_scaler, X_train, X_test, \
        X_train_unscaled, X_test_unscaled, \
            y_scaler, y_train, y_test, \
                y_train_unscaled, y_test_unscaled, \
                    y_scaler_train, y_scaler_test = generate_data(load_timeseries_data())
    
    

    train_losses = []
    test_losses = []

    epochs = 30
    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = validate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, '
            f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}')

def main(model_type="kan"):    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "cheby" in model_type:
        model = ChebyKAN().to(device)
    elif "jacobi" in model_type:
        model = ChebyKAN().to(device)
    elif "tkat" in model_type:
        model = TKAT(
            sequence_length=10, 
            num_unknown_features=5, 
            num_known_features=3, 
            num_embedding=32, 
            num_hidden=64, 
            num_heads=4, 
            n_ahead=5, 
            use_tkan=True
        )
    elif 'tkan' in model_type:
        model = nn.Sequential([
            TKAN(X_train.shape[1:], tkan_activations=[{'grid_size': 3} for i in range(5)], sub_kan_output_dim = 20, sub_kan_input_dim = 1, return_sequences=True),
            TKAN(100, tkan_activations=[{'grid_size': 3} for i in range(5)], sub_kan_output_dim = 20, sub_kan_input_dim = 1, return_sequences=False),
            nn.Linear(units=n_ahead, activation='linear')
        ])
    elif 'gru' in model_type:
        model = nn.Sequential([
            nn.Input(shape=X_train.shape[1:]),
            nn.GRU(100, return_sequences=True),
            nn.GRU(100, return_sequences=False),
            nn.Dense(units=n_ahead, activation='linear')
        ])
    elif 'LSTM' in model_type:
        model = nn.Sequential([
            nn.Input(shape=X_train.shape[1:]),
            nn.LSTM(100, return_sequences=True),
            nn.LSTM(100, return_sequences=False),
            nn.Dense(units=n_ahead, activation='linear')
        ])
    elif 'MLP' in model_type:
        model = nn.Sequential([
            Input(shape=X_train.shape[1:]),
            nn.Flatten(),
            Dense(100, activation='relu'),
            Dense(100, activation='relu'),
            Dense(units=n_ahead, activation='linear')
        ])
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    criterion = nn.CrossEntropyLoss()

    # LBFGS is really slow
    # optimizer = optim.LBFGS(model.parameters(), lr=0.01)
    # Adam works with very low lr
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        if isinstance(optimizer, optim.LBFGS):
            def closure():
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
        else:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            
        total_loss += loss
        
    return total_loss / len(train_loader)

def validate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    return total_loss / len(test_loader), correct / len(test_loader.dataset)    