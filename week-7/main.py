import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

def loss_to_weight(loss, dataset):
    return torch.sqrt(loss) * dataset.scaler_weight.scale_[0]

# NN model
class MyData(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        le = LabelEncoder()
        df['Gender'] = le.fit_transform(df['Gender'])
        
        self.scaler_height = StandardScaler()
        self.scaler_weight = StandardScaler()
        
        df[['Height']] = self.scaler_height.fit_transform(df[['Height']])
        df[['Weight']] = self.scaler_weight.fit_transform(df[['Weight']])
        
        self.X = torch.tensor(df[['Gender', 'Height']].values, dtype=torch.float32)
        self.y = torch.tensor(df['Weight'].values, dtype=torch.float32).unsqueeze(1)
        
        self.weight_mean = self.scaler_weight.mean_[0]
        self.weight_std = self.scaler_weight.scale_[0]
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.X)
    
# Define Custom Dataset Class
class TitanicDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
        df.fillna({'Age': df['Age'].median(), 'Fare': df['Fare'].median()}, inplace=True)
        df.fillna({'Embarked': df['Embarked'].mode()[0]}, inplace=True)
        df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
        df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True, errors='ignore')
        
        # Feature Engineering
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        df['Fare'] = np.log1p(df['Fare'])  # Log transform Fare to reduce skew
        df = pd.get_dummies(df, columns=['Pclass', 'Embarked'])  # One-Hot Encoding
        
        # Scaling
        scaler = MinMaxScaler()
        self.X = scaler.fit_transform(df[['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']].values)
        self.y = df['Survived'].values.astype(np.float32).reshape(-1, 1)
        
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.X)

# Define Neural Network Class
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(2, 2)
        self.hidden2 = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1) 

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.output(x)  
        return x

class TitanicNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(TitanicNeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.hidden2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.hidden3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.hidden1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.hidden2(x)))
        x = self.dropout(x)
        x = torch.relu(self.hidden3(x))
        x = torch.sigmoid(self.output(x))
        return x
    # Below ACC 76%
    # def __init__(self, input_size):
    #     super(TitanicNeuralNetwork, self).__init__()
    #     self.hidden1 = nn.Linear(input_size, 32)
    #     self.hidden2 = nn.Linear(32, 16)
    #     self.hidden3 = nn.Linear(16, 8)
    #     self.output = nn.Linear(8, 1)
        
    # def forward(self, x):
    #     x = torch.relu(self.hidden1(x))
    #     x = torch.relu(self.hidden2(x))
    #     x = torch.relu(self.hidden3(x))
    #     x = torch.sigmoid(self.output(x))
    #     return x

# Function to split dataset into train, validation, and test loaders
def split_data(dataset, batch_size=100, test_size=0.1, val_size=0.3, random_state=7):
    train_val_X, test_X, train_val_y, test_y = train_test_split(dataset.X, dataset.y, test_size=test_size, random_state=random_state)
    train_X, val_X, train_y, val_y = train_test_split(train_val_X, train_val_y, test_size=val_size, random_state=random_state)
    
    train_set = list(zip(train_X, train_y))
    val_set = list(zip(val_X, val_y))
    test_set = list(zip(test_X, test_y))
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_task1(dataset,epoch: int = 80, batch_size: int = 1000, learning_rate: float = 0.01, patience: int = 20):
    dataset = MyData("gender-height-weight.csv")
    train_loader, val_loader, test_loader = split_data(dataset, batch_size=batch_size)
    
    model = MyNeuralNetwork()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    best_loss = float("inf")
    no_improve_count = 0
    
    print("----- Task1: Training Procedure -----")
    print(dataset.scaler_weight.mean_[0],dataset.scaler_weight.scale_[0])
    for epoch_idx in tqdm(range(epoch)):
        model.train()
        loss_sum = 0
        count = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            count += X_batch.size(0)
            loss_sum += loss.item() * X_batch.size(0)
        
        avg_loss = loss_sum / count
        avg_weight_loss = loss_to_weight(torch.tensor(avg_loss), dataset)
        
        # Validation Step
        model.eval()
        val_loss_sum = 0
        val_count = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                outputs = model(X_val)
                val_loss = criterion(outputs, y_val)
                val_loss_sum += val_loss.item() * X_val.size(0)
                val_count += X_val.size(0)
        val_loss_avg = val_loss_sum / val_count
        val_weight_loss = loss_to_weight(torch.tensor(val_loss_avg), dataset)
        
        if val_loss_avg < best_loss:
            best_loss = val_loss_avg
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count >= patience:
            learning_rate *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            no_improve_count = 0
        
        # tqdm.write(f'Epoch {epoch_idx + 1}, Train Loss: {avg_weight_loss.item():.6f}, Val Loss: {val_weight_loss.item():.6f}')
    
    return model, test_loader, dataset

def evaluate_task1(model, test_loader, dataset):
    criterion = nn.MSELoss()
    
    print("----- Evaluating Procedure -----")
    loss_sum = 0
    count = 0
    
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss_sum += loss.item() * X_batch.size(0)
            count += X_batch.size(0)
    
    avg_loss = loss_sum / count
    avg_weight_loss = loss_to_weight(torch.tensor(avg_loss), dataset)
    print(f'Final Test Loss in Weight Scale: {avg_weight_loss.item():.6f}')

# Training Procedure
def train_task2(dataset:TitanicDataset, epoch: int = 200, batch_size: int = 128, learning_rate: float = 0.001, patience: int = 30):
    train_loader, val_loader, test_loader = split_data(dataset, batch_size=batch_size)
    
    model = TitanicNeuralNetwork(input_size=7)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    best_acc = 0
    no_improve_count = 0
    
    print("----- Task2: Training Procedure -----")
    for epoch_idx in tqdm(range(epoch)):
        model.train()
        correct_count = 0
        total_samples = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            predictions = (outputs > 0.5).float()
            correct_count += (predictions == y_batch).sum().item()
            total_samples += y_batch.size(0)
        
        train_acc = correct_count / total_samples
        
        # Validation Step
        model.eval()
        correct_count = 0
        total_samples = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                outputs = model(X_val)
                predictions = (outputs > 0.5).float()
                correct_count += (predictions == y_val).sum().item()
                total_samples += y_val.size(0)
        val_acc = correct_count / total_samples
        
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count >= patience:
            learning_rate *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            no_improve_count = 0
        
        # tqdm.write(f'Epoch {epoch_idx + 1}, Train Acc: {train_acc:.6f}, Val Acc: {val_acc:.6f}')
    
    return model, test_loader, dataset

# Evaluating Procedure
def evaluate_task2(model, test_loader):
    print("----- Evaluating Procedure -----")
    correct_count = 0
    total_samples = 0
    
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predictions = (outputs > 0.5).float()
            correct_count += (predictions == y_batch).sum().item()
            total_samples += y_batch.size(0)
    
    test_acc = correct_count / total_samples
    print(f'Final Test Accuracy: {test_acc:.6f}')

class TaskHandler:
    @staticmethod
    def run_Task1(epoch: int = 80, batch_size: int = 16, learning_rate: float = 0.01, patience: int = 20):
        # Load Data
        dataset = MyData("gender-height-weight.csv")
        # Training Procedure
        model, test_loader, dataset = train_task1(dataset = dataset, epoch = 80, batch_size = 16, learning_rate = 0.01, patience = 20)
        # Evaluating Procedure
        evaluate_task1(model, test_loader, dataset)

    @staticmethod
    def run_Task2(epoch: int = 200):
        dataset = TitanicDataset("titanic.csv")
        # Neural Network Setup
        model, test_loader, dataset = train_task2(dataset)
        evaluate_task2(model, test_loader)


if __name__ == "__main__":
    TaskHandler.run_Task1()
    TaskHandler.run_Task2()


# ----- Task1: Training Procedure -----
# 161.44035683283076 32.106833544431716
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 80/80 [00:23<00:00,  3.35it/s]
# ----- Evaluating Procedure -----
# Final Test Loss in Weight Scale: 10.262472

# ----- Task2: Training Procedure -----
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:03<00:00, 61.48it/s]
# ----- Evaluating Procedure -----
# Final Test Accuracy: 0.844444