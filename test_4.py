import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Trans_utils import *
from Network import *

import torch.optim as optim
from torch.utils.data import DataLoader, StackDataset, Dataset
import h5py
import pandas as pd
from transformers import BertModel, BertConfig, AutoTokenizer, AutoModel
import optuna
import json
# add 10 fold cross valid---------
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Subset
import json
import time

# add visualization---------
# %matplotlib notebook
import matplotlib.pyplot as plt
import datetime
# add hyperparams  tunning--------
enc_seq_len = 6
dec_seq_len = 2
output_sequence_length = 1

dim_val = 10
dim_attn = 5
lr = 0.002
num_epochs = 20

n_heads = 3 

n_decoder_layers = 3
n_encoder_layers = 3

batch_size = 15



# ---------1. train and evaluate model---------


# Load data
def load_data(datadir):
    trainfile = h5py.File(f'{datadir}/train.h5', 'r')
    validfile = h5py.File(f'{datadir}/valid.h5', 'r')
    testfile = h5py.File(f'{datadir}/test.h5', 'r')

  # 合并训练集和验证集
    X_trainhalflife = torch.from_numpy(np.concatenate((trainfile['data'][:], validfile['data'][:]))).float()
    X_trainpromoter = torch.from_numpy(np.concatenate((trainfile['promoter'][:], validfile['promoter'][:]))).float()
    y_train = torch.from_numpy(np.concatenate((trainfile['label'][:], validfile['label'][:]))).float().unsqueeze(1)

    X_testhalflife =  torch.from_numpy(testfile['data'][:] ).float()
    X_testpromoter = torch.from_numpy(testfile['promoter'][:]).float()# float()
    y_test =  torch.from_numpy( testfile['label'][:] ).float().unsqueeze(1)

    return (X_trainhalflife, X_trainpromoter, y_train), (X_testhalflife, X_testpromoter, y_test)


# Define a custom Dataset
class GeneDataset(Dataset):
    def __init__(self, halflife_data, promoter_data, labels):
        
        self.halflife_data = halflife_data
        self.promoter_data = promoter_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.promoter_data[idx], self.halflife_data[idx]), self.labels[idx]

# Transformer-based Model
class GeneExpressionTransformer(nn.Module):
    def __init__(self, halflife_dim, dim_val=10, dim_attn=5, n_heads=3, n_decoder_layers=3, n_encoder_layers=3, num_classes=1):
        super(GeneExpressionTransformer, self).__init__()
        self.transformer = Transformer(
            dim_val=dim_val,
            dim_attn=dim_attn,
            input_size=4,  
            dec_seq_len=2, # 
            out_seq_len=1, 
            n_decoder_layers=n_decoder_layers,
            n_encoder_layers=n_encoder_layers,
            n_heads=n_heads
        )
        self.fc1 = nn.Linear(halflife_dim + 1, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def forward(self, promoter, halflife):
        transformer_output = self.transformer(promoter)
        x = torch.cat((transformer_output, halflife), dim=1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Training loop
def train_model(model, train_loader, valid_loader, optimizer, num_epochs=2, save_path='../mrna_pred_trans/mode/best_model.pth', patience=5):
    criterion = nn.MSELoss()
    best_valid_loss = float('inf')
    device = torch.device('cuda:1')
    model.to(device)
    
    epochs_no_improve = 0
    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for (promoter, halflife), labels in train_loader:
            promoter, halflife, labels = promoter.to(device), halflife.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(promoter, halflife)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validate the model
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for (promoter, halflife), labels in valid_loader:
                promoter, halflife, labels = promoter.to(device), halflife.to(device), labels.to(device)
                outputs = model(promoter, halflife)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

        avg_valid_loss = valid_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)
        print(f'Epoch {epoch + 1}, Validation Loss: {avg_valid_loss}')

        # Check if this is the best model so far
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), save_path)
            print(f'Model saved with validation loss: {best_valid_loss}')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Check for early stopping
        if epochs_no_improve == patience:
            print('Early stopping triggered.')
            break

    # Plot the training and validation losses
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    #plt.show()
    plt.savefig(f'../mrna_pred_trans/doc/train_loss_{timestamp}.png')
    plt.close() 



def evaluate_model(model, valid_loader):
    criterion = nn.MSELoss()
    device = torch.device('cuda:1')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    valid_loss = 0
    predictions = []
    actuals = []
    with torch.no_grad():
        for (promoter, halflife), labels in valid_loader:
            promoter, halflife, labels = promoter.to(device), halflife.to(device), labels.to(device)
            outputs = model(promoter, halflife)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(labels.cpu().numpy())   
    avg_valid_loss = valid_loss / len(valid_loader)

     # Plot predictions vs actuals
    plt.figure()
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Actuals')
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='red')  # Line y=x
    #plt.show()
    plt.savefig('../mrna_pred_trans/doc/evaluate_model_predicted_vs_actual.png')
    plt.close() 

    return predictions, actuals, avg_valid_loss


class HyperparameterOptimization:
    def __init__(self, full_dataset, X_trainhalflife, n_splits=10):
        self.full_dataset = full_dataset
        self.X_trainhalflife = X_trainhalflife
        self.n_splits = n_splits

    def objective(self, trial):
        # Suggest hyperparameters
        batch_size = trial.suggest_categorical('batch_size', [2, 4, 8, 16])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

        # Suggest transformer-specific hyperparameters
        dim_val = 10 # trial.suggest_int('dim_val', 8, 64, step=8)
        dim_attn = 5 # trial.suggest_int('dim_attn', 4, 16, step=2)
        n_heads = 3 # trial.suggest_int('n_heads', 1, 8)
        n_decoder_layers = 3 # trial.suggest_int('n_decoder_layers', 1, 6)
        n_encoder_layers = 3 # trial.suggest_int('n_encoder_layers', 1, 6)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_idx, valid_idx) in enumerate(kf.split(self.full_dataset)):
            train_subset = Subset(self.full_dataset, train_idx)
            valid_subset = Subset(self.full_dataset, valid_idx)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)

            # Define model with suggested hyperparameters
            model = GeneExpressionTransformer(
                halflife_dim=self.X_trainhalflife.shape[1],
                dim_val=dim_val,
                dim_attn=dim_attn,
                n_heads=n_heads,
                n_decoder_layers=n_decoder_layers,
                n_encoder_layers=n_encoder_layers
            )
            model.dropout = nn.Dropout(dropout_rate)

            # Define optimizer
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Train model
            train_model(model, train_loader, valid_loader, optimizer)

            # Evaluate model on validation subset
            _, _, valid_loss = evaluate_model(model, valid_loader)
            fold_results.append(valid_loss)

        # Return the average validation loss across all folds
        avg_loss = np.mean(fold_results)
        return avg_loss


def save_best_params(study, filename='best_params.json'):
    best_params = study.best_params
    with open(filename, 'w') as f:
        json.dump(best_params, f)
    print(f"Best hyperparameters saved to {filename}")


def main(datadir):
    start_time = time.time()
    (X_trainhalflife, X_trainpromoter, y_train), (X_testhalflife, X_testpromoter, y_test) = load_data(datadir)

    full_dataset = GeneDataset(X_trainhalflife, X_trainpromoter, y_train)

    optimizer = HyperparameterOptimization(full_dataset, X_trainhalflife)

    study = optuna.create_study(direction='minimize')
    study.optimize(optimizer.objective, n_trials=1)  # Increase n_trials for better results
    print('Best hyperparameters: ', study.best_params)
    save_best_params(study, filename='best_params.json')

    # 计算训练时间
    training_time =( time.time() - start_time)/3600
    print(f'Training time: {training_time:.2f} hours')

    # Load the best model----------------------
    model_load_start_time = time.time()
    with open('best_params.json', 'r') as f:
        best_params = json.load(f)
 
    model = GeneExpressionTransformer(
        halflife_dim=X_trainhalflife.shape[1]
        # dim_val=best_params['dim_val'],
        # dim_attn=best_params['dim_attn']
        # n_heads=best_params['n_heads'],
        # n_decoder_layers=best_params['n_decoder_layers'],
        # n_encoder_layers=best_params['n_encoder_layers']
    )
    model.load_state_dict(torch.load('../mrna_pred_trans/mode/best_model.pth', weights_only=True))
    # Optionally evaluate on test set
    test_dataset = GeneDataset(X_testhalflife, X_testpromoter, y_test)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    predictions, actuals, avg_test_loss = evaluate_model(model, test_loader)
    # 计算评估时间
    evaluation_time = (time.time() - model_load_start_time)/3600
    print(f'Evaluation time: {evaluation_time:.2f} hours')


datadir = "../../pM10Kb_1KTest" # ../../pM10Kb_1KTest4_from1_2000seq"
main(datadir)


