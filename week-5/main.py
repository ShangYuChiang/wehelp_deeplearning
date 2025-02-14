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

class Network:
    def __init__(self, hidden_activations, output_activation, weights, biases, **kwargs):
        self.weights = weights
        self.biases = biases
        self.hidden_activations = [ActivationFunctions.get(act) for act in hidden_activations]
        self.output_activation = ActivationFunctions.get(output_activation)


    def forward(self, input_values):
        self.hidden_layer_inputs = []
        self.hidden_layer_outputs = [input_values]

        for weight, bias, activation in zip(self.weights[:-1], self.biases[:-1], self.hidden_activations):    
            hidden_layer_input = np.dot(weight, self.hidden_layer_outputs[-1]) + bias
            self.hidden_layer_inputs.append(hidden_layer_input)
            self.hidden_layer_outputs.append(activation(hidden_layer_input))

        self.output_layer_input = (
            np.dot(self.weights[-1], self.hidden_layer_outputs[-1]) + self.biases[-1]
        )
        self.output_layer_output = self.output_activation(self.output_layer_input)
        return self.output_layer_output
    
    def backward(self, output_losses: np.ndarray):
        output_layer_delta = output_losses * DerivativeFunctions.get(self.output_activation.__name__)(self.output_layer_input)
        self.output_layer_weight_gradient = np.dot(
            output_layer_delta, self.hidden_layer_outputs[-1].T
        )
        self.output_bias_gradient = output_layer_delta
  
        self.hidden_layer_weight_gradients = []
        self.hidden_layer_bias_gradients = []

        hidden_layer_delta = np.dot(self.weights[-1].T, output_layer_delta)

        for i in reversed(range(len(self.weights[:-1]))):
            hidden_layer_delta *= DerivativeFunctions.get(self.hidden_activations[i].__name__)(self.hidden_layer_inputs[i])
            self.hidden_layer_weight_gradients.append(np.dot(hidden_layer_delta, self.hidden_layer_outputs[i].T))
            self.hidden_layer_bias_gradients.append(hidden_layer_delta)
            if i > 0:
                hidden_layer_delta = np.dot(self.weights[i].T, hidden_layer_delta)

        self.hidden_layer_weight_gradients.reverse()
        self.hidden_layer_bias_gradients.reverse()
        

    def zero_grad(self, learning_rate: float):
        for i in range(len(self.weights[:-1])):
            self.weights[i] -= learning_rate * self.hidden_layer_weight_gradients[i]
            self.biases[i] -= learning_rate * self.hidden_layer_bias_gradients[i]
        
        self.weights[-1] -= learning_rate * self.output_layer_weight_gradient
        self.biases[-1] -= learning_rate * self.output_bias_gradient

        self.hidden_layer_weight_gradients = [np.zeros_like(w) for w in self.weights]
        self.hidden_layer_bias_gradients = [np.zeros_like(b) for b in self.biases] if self.biases is not None else None

class TaskHandler:
    @staticmethod
    def run_Task1(epoch: int = 1, task: str = '1-1'):
        config = {
            'bias': True,
            'weights': [
                np.array([[0.5, 0.2], [0.6, -0.6]]),
                np.array([[0.8, -0.5]]),
                np.array([[0.6], [-0.3]])
            ],
            'biases': [
                np.array([[0.3], [0.25]]),
                np.array([[0.6]]),
                np.array([[0.4], [0.75]])
            ]
        }
        nn = Network(
            input_size=config['weights'][0].shape[0],
            hidden_layer_sizes=[2, 1],
            output_size=config['biases'][-1].shape[0],
            hidden_activations=["relu",'linear'],
            output_activation="linear",**config
        )
        inputs = np.array([[1.5],
                           [0.5]])
        groundtruths = np.array([[0.8], 
                                 [1]])
        loss_fn = MSE()
        learning_rate = 0.01

        print(f"------ Task {task} ------")
        for i in range(epoch):
            outputs = nn.forward(inputs)
            loss = loss_fn.get_total_loss(outputs=outputs, groundtruths=groundtruths)
            losses = loss_fn.get_output_losses(outputs=outputs, groundtruths=groundtruths)
            nn.backward(losses)
            nn.zero_grad(learning_rate)
        
        if epoch == 1:
            print(f"{nn.weights=}\n")
        else:
            print(f"Total Loss: {loss}\n")


    @staticmethod
    def run_Task2(epoch: int = 1, task: str = '2-1'):
        config = {
            'bias': True,
            'weights': [
                np.array([[0.5, 0.2], [0.6, -0.6]]),
                np.array([[0.8, 0.4]]),
            ],
            'biases': [
                np.array([[0.3], [0.25]]),
                np.array([[-0.5]]),
            ]
        }
        nn = Network(
            input_size=config['weights'][0].shape[0],
            hidden_layer_sizes=[2],
            output_size=config['biases'][-1].shape[0],
            hidden_activations=["relu"],
            output_activation="sigmoid",**config
        )
        inputs = np.array([[0.75],
                           [1.25]])
        
        groundtruths = np.array([[1]])
        loss_fn = BinaryCrossEntropy()
        learning_rate = 0.1
        
        print(f"------ Task {task} ------")
        for i in range(epoch):
            outputs = nn.forward(inputs)
            loss = loss_fn.get_total_loss(outputs=outputs, groundtruths=groundtruths)
            losses = loss_fn.get_output_losses(outputs=outputs, groundtruths=groundtruths)
            nn.backward(losses)
            nn.zero_grad(learning_rate)
        
        if epoch == 1:
            print(f"{nn.weights=}\n")
        else:
            print(f"Total Loss: {loss}\n")



if __name__ == "__main__":
    TaskHandler.run_Task1(1, '1-1')
    TaskHandler.run_Task1(1000, '1-2')
    TaskHandler.run_Task2(1, '2-1')
    TaskHandler.run_Task2(1000, '2-2')
