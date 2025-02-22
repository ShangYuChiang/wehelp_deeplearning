import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from tqdm import tqdm

class BaseFunction():
    def __repr__(self):
        return f"{self.__class__.__name__}()"

class ActivationFunctions(BaseFunction):
    @staticmethod
    def linear(x: np.ndarray) -> np.ndarray:
        return x
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    @classmethod
    def get(cls, name: str = None):
        if name is None or not hasattr(cls, name):
            return cls.linear 
        return getattr(cls, name)


class LossFunction:
    def get_total_loss(self, groundtruths: np.ndarray, outputs: np.ndarray):
        raise NotImplementedError

    def get_output_losses(self, groundtruths: np.ndarray, outputs: np.ndarray):
        raise NotImplementedError

class MSE(LossFunction):
    def get_total_loss(self, groundtruths: np.ndarray, outputs: np.ndarray):
        return np.mean((groundtruths - outputs) ** 2)

    def get_output_losses(self, groundtruths: np.ndarray, outputs: np.ndarray):
        n = outputs.shape[0]
        return (2 / n) * (outputs - groundtruths)

class BinaryCrossEntropy(LossFunction):
    def get_total_loss(self, groundtruths: np.ndarray, outputs: np.ndarray):
        return -np.sum(
            groundtruths * np.log(outputs + 1e-15) + (1 - groundtruths) * np.log(1 - outputs + 1e-15)
        )

    def get_output_losses(self, groundtruths: np.ndarray, outputs: np.ndarray):
        return -(groundtruths / outputs) + (1 - groundtruths) / (1 - outputs)

class DerivativeFunctions(BaseFunction):
    @staticmethod
    def linear(x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return ActivationFunctions.sigmoid(x) * (1 - ActivationFunctions.sigmoid(x))
    
    @classmethod
    def get(cls, name: str = None):
        if name is None or not hasattr(cls, name):
            return cls.linear 
        return getattr(cls, name)


class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, hidden_activations, output_activation):
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = [np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * np.sqrt(2 / self.layer_sizes[i]) for i in range(len(self.layer_sizes)-1)]
        self.biases = [np.random.randn(1, self.layer_sizes[i+1]) for i in range(len(self.layer_sizes)-1)]
        self.hidden_activations = [ActivationFunctions.get(act) for act in hidden_activations]
        self.output_activation = ActivationFunctions.get(output_activation)
    
    def forward(self, x):
        self.hidden_layer_inputs = []
        self.hidden_layer_outputs = [x]
        
        for weight, bias, activation in zip(self.weights[:-1], self.biases[:-1], self.hidden_activations):
            hidden_layer_input = np.dot(self.hidden_layer_outputs[-1], weight) + bias
            self.hidden_layer_inputs.append(hidden_layer_input)
            self.hidden_layer_outputs.append(activation(hidden_layer_input))
        
        self.output_layer_input = np.dot(self.hidden_layer_outputs[-1], self.weights[-1]) + self.biases[-1]
        self.final_output = self.output_activation(self.output_layer_input)
        return self.final_output
    
    def backward(self, output_losses):
        delta = output_losses * (self.final_output * (1 - self.final_output))  # Assuming sigmoid derivative
        self.dW = [np.dot(self.hidden_layer_outputs[-1].T, delta)]
        self.db = [np.sum(delta, axis=0, keepdims=True)]
        
        for i in reversed(range(len(self.weights)-1)):
            delta = np.dot(delta, self.weights[i+1].T) * (self.hidden_layer_outputs[i+1] > 0)
            self.dW.insert(0, np.dot(self.hidden_layer_outputs[i].T, delta))
            self.db.insert(0, np.sum(delta, axis=0, keepdims=True))
    
    def zero_grad(self, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.dW[i]
            self.biases[i] -= learning_rate * self.db[i]

class TaskHandler:
    @staticmethod
    def run_Task1(epoch: int = 1000):
        pass

    @staticmethod
    def run_Task2(epoch: int = 200):
        # Load and preprocess Titanic dataset
        def load_titanic_data(filename):
            df = pd.read_csv(filename)
            df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
            df.fillna({'Age': df['Age'].median()}, inplace=True)
            df.fillna({'Fare': df['Fare'].median()}, inplace=True)
            df.fillna({'Embarked': df['Embarked'].mode()[0]}, inplace=True)
            df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
            df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True, errors='ignore')
            return df[['Sex','Pclass','SibSp','Parch','Embarked','Survived']].values

        # Load Data
        data = load_titanic_data("titanic.csv")
        X = data[:, :-1]
        y = data[:, -1].reshape(-1, 1)

        # Neural Network Setup
        nn = NeuralNetwork(input_size=X.shape[1], hidden_sizes=[16,8], output_size=1, hidden_activations=["linear","linear"], output_activation="sigmoid")
        loss_fn = MSE()
        learning_rate = 0.001
        best_acc, no_improve_count, patience = 0, 0, 20

        threshold = 0.5

        # Training Procedure
        print("----- Task2: Training Procedure -----")
        for i in tqdm(range(300)):  # REPEAT_TIMES
            correct_count = 0
            for x, e in zip(X, y):
                x = x.reshape(1, -1)
                outputs = nn.forward(x)
                loss = loss_fn.get_total_loss(e, outputs)
                output_losses = loss_fn.get_output_losses(e, outputs)
                nn.backward(output_losses)
                nn.zero_grad(learning_rate)
                survival_status = 1 if outputs > threshold else 0
                if survival_status == e:
                    correct_count += 1
            correct_rate = correct_count / len(X)
            # if correct_rate > best_acc:
            #     nn.save_weights()
            # print(f"{i} Model Accuracy: {correct_rate * 100:.2f}%")

        # Evaluating Procedure
        print("----- Task2: Evaluating Procedure -----")
        correct_count = 0
        threshold = 0.5
        for x, e in zip(X, y):
            x = x.reshape(1, -1)
            output = nn.forward(x)
            survival_status = 1 if output > threshold else 0
            if survival_status == e:
                correct_count += 1

        correct_rate = correct_count / len(X)
        print(f"Model Accuracy: {correct_rate * 100:.2f}%")
    
    @staticmethod
    def run_Task_pytroch():
        print("----- Task3: Pytorch Procedure -----")
        # 1. Build a tensor from the original list and print shape & dtype
        tensor1 = torch.tensor([[2, 3, 1], [5, -2, 1]])
        print("Shape of Tensor 1:", tensor1.shape)
        print("Dtype of Tensor 1:", tensor1.dtype)


        # 2. Build a 3x4x2 tensor filled with random float numbers on [0,1]
        tensor2 = torch.rand((3, 4, 2))
        print("\nShape of Tensor 2:", tensor2.shape)
        print("Tensor 2 (random float numbers on [0,1]):\n", tensor2)

        # 3. Build a 2x1x5 tensor filled with ones
        tensor3 = torch.ones((2, 1, 5))
        print("\nShape of Tensor 3:", tensor3.shape)
        print("Tensor 3 (filled with 1s):\n", tensor3)

        # 4. Matrix multiplication of two tensors
        tensor4 = torch.tensor([[1, 2, 4], [2, 1, 3]])  # Shape: (2,3)
        tensor5 = torch.tensor([[5], [2], [1]])         # Shape: (3,1)
        result_matrix_mul = torch.matmul(tensor4, tensor5)  # (2,3) x (3,1) -> (2,1)
        print("Matrix Multiplication Result:\n", result_matrix_mul)

        # 5. Element-wise multiplication of two tensors
        tensor6 = torch.tensor([[1, 2], [2, 3], [-1, 3]])  # Shape: (3,2)
        tensor7 = torch.tensor([[5, 4], [2, 1], [1, -5]])  # Shape: (3,2)
        result_elementwise_mul = tensor6 * tensor7  # Element-wise multiplication
        print("Element-wise Multiplication Result:\n", result_elementwise_mul)


if __name__ == "__main__":
    # TaskHandler.run_Task1()
    TaskHandler.run_Task2()
    TaskHandler.run_Task_pytroch()

