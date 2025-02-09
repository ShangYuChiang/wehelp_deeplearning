import numpy as np

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


class LossFunctions(BaseFunction):
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Prevent log(0)
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Prevent log(0)
        return -np.sum(y_true * np.log(y_pred), axis=-1)

class Network:
    def __init__(self, input_size: int, hidden_layer_sizes: list, output_size: int, 
                 activation: str = "relu", output_activation: str = None, **kwargs):
        self.input_size = input_size  
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size

        self.activation = ActivationFunctions.get(activation) 
        self.output_activation = ActivationFunctions.get(output_activation) if output_activation else None
        
        self.bias = kwargs.get('bias', True)
        self.weights = kwargs.get('weights', None)
        self.biases = kwargs.get('biases', None)
        
        self.weights = self.weights if self.weights else self.initialize_weights()
        self.biases = self.biases if self.biases else self.initialize_biases()
    
    def initialize_weights(self):
        layer_sizes = [self.input_size] + self.hidden_layer_sizes + [self.output_size]
        return [np.ones((layer_sizes[i], layer_sizes[i+1])) for i in range(len(layer_sizes) - 1)]

    def initialize_biases(self):
        if not self.bias:
            return []
        layer_sizes = self.hidden_layer_sizes + [self.output_size]
        return [np.ones(size) for size in layer_sizes]
    
    def forward(self, inputs: np.ndarray):
        layer_output = np.array(inputs)
        
        for i in range(len(self.hidden_layer_sizes) + 1):
            weights = np.array(self.weights[i])
            biases = np.array(self.biases[i]) if self.bias else np.zeros(weights.shape[1])
            layer_output = np.dot(layer_output, weights) + biases
            if i < len(self.hidden_layer_sizes):
                layer_output = self.activation(layer_output)

        if self.output_activation:
            layer_output = self.output_activation(layer_output)

        return layer_output[:self.output_size]


def evaluate(nn, inputs, groundtruths, loss_fn=LossFunctions.mse):
    for inp, gt in zip(inputs, groundtruths):
        predict = nn.forward(inp)
        loss = loss_fn(gt, predict)
        print("Output",predict)
        print("Total Loss", loss)

class TaskHandler:
    @staticmethod
    def run_Regression_Task():
        print("\n------ Task 1: Regression  ------")
        config = {
            'bias': True,
            'weights': [np.array([[0.5, 0.6], [0.2, -0.6]]), np.array([[0.8,0.4], [-0.5,0.5]])], 
            'biases': [np.array([0.3, 0.25]), np.array([0.6,-0.25])]
        }

        input_size = config['weights'][0].shape[0]
        output_size = config['biases'][-1].shape[0]
        hidden_layer_sizes = [2]
        nn = Network(input_size, hidden_layer_sizes, output_size, \
                     activation = "relu", output_activation = "linear", **config)

        inputs = [[1.5, 0.5], [0, 1]]
        groundtruths = [np.array([0.8, 1]), np.array([0.5, 0.5])]
        evaluate(nn, inputs, groundtruths, LossFunctions.mse)
    
    @staticmethod
    def run_Binary_Classification_Task():
        print("\n------ Task 2: Binary Classification  ------")
        config = {
            'bias': True,
            'weights': [
                np.array([[0.5, 0.6], [0.2, -0.6]]),
                np.array([[0.8, 0.0], [0.4, 0.0]]),
            ],
            'biases': [np.array([0.3, 0.25]),np.array([-0.5])]
        }
        input_size = config['weights'][0].shape[0]
        output_size = config['biases'][-1].shape[0]
        hidden_layer_sizes = [2]
        nn = Network(input_size, hidden_layer_sizes, output_size, \
                     activation = "relu", output_activation = "sigmoid", **config)
        
        inputs = [[0.75, 1.25], [-1, 0.5]]
        groundtruths = [np.array([1]), np.array([0])]
        evaluate(nn, inputs, groundtruths, LossFunctions.binary_cross_entropy)

    @staticmethod
    def run_Multilabel_Classification_Task():
        print("\n------ Task 3: Multilabel Classification ------")
        config = {
            'bias': True,
            'weights': [
                np.array([[0.5, 0.6], [0.2, -0.6]]),
                np.array([[0.8, 0.5, 0.3], [-0.4, 0.4, 0.75]])
            ],
            'biases': [np.array([0.3, 0.25]), np.array([0.6, 0.5, -0.5])]
        }
        input_size = config['weights'][0].shape[0]
        output_size = config['biases'][-1].shape[0]
        hidden_layer_sizes = [2]
        nn = Network(input_size, hidden_layer_sizes, output_size,\
                      activation = "relu", output_activation = "sigmoid", **config)
        
        inputs = [[1.5, 0.5], [0, 1]]
        groundtruths = [np.array([1, 0, 1]), np.array([1, 1, 0])]
        evaluate(nn, inputs, groundtruths, LossFunctions.binary_cross_entropy)
        
    
    @staticmethod
    def run_Multiclass_Classification_Task():
        print("\n------ Task 4: Multiclass Classification ------")
        config = {
            'bias': True,
            'weights': [
                np.array([[0.5, 0.6], [0.2, -0.6]]),
                np.array([[0.8, 0.5, 0.3], [-0.4, 0.4, 0.75]])
            ],
            'biases': [np.array([0.3, 0.25]), np.array([0.6, 0.5, -0.5])]
        }
        input_size = config['weights'][0].shape[0]
        output_size = config['biases'][-1].shape[0]
        hidden_layer_sizes = [2]
        nn = Network(input_size, hidden_layer_sizes, output_size,\
                      activation = "relu", output_activation = "softmax", **config)
        
        inputs = [[1.5, 0.5], [0, 1]]
        groundtruths = [np.array([1, 0, 0]), np.array([0, 0, 1])]
        evaluate(nn, inputs, groundtruths, LossFunctions.categorical_cross_entropy)
        

def Testing_Functions():
    x = np.array([-2, -1, 0, 1, 2])
    print("Linear:", ActivationFunctions.linear(x))
    print("ReLU:", ActivationFunctions.relu(x))
    print("Sigmoid:", ActivationFunctions.sigmoid(x))
    print("Softmax:", ActivationFunctions.softmax(x))
    
    y_true = np.array([1, 0, 1])
    y_pred = np.array([0.9, 0.1, 0.8])
    print("MSE:", LossFunctions.mse(y_true, y_pred))
    print("Binary Cross Entropy:", LossFunctions.binary_cross_entropy(y_true, y_pred))
    
    y_true_cat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred_cat = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
    print("Categorical Cross Entropy:", LossFunctions.categorical_cross_entropy(y_true_cat, y_pred_cat))

if __name__ == "__main__":
    #Testing_Functions()
    TaskHandler.run_Regression_Task()
    TaskHandler.run_Binary_Classification_Task()
    TaskHandler.run_Multilabel_Classification_Task()
    TaskHandler.run_Multiclass_Classification_Task()