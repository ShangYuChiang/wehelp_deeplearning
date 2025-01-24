import os

class Network:
    """
        Initialize the neural network with input size, hidden layer sizes, output size, and other parameters.

        Args:
        **kwargs: Optional keyword arguments for bias, weights, and biases.

        Parameters:
        input_size (int): The number of input features.
        hidden_layer_sizes (list): The number of neurons in each hidden layer.
        output_size (int): The number of output neurons.
        bias (bool): Whether to use bias for each layer. Default is True.
        weights (list): Custom initial weights. If not provided, defaults to random initialization.
        biases (list): Custom initial biases. If not provided, defaults to 1.
    """
    def __init__(self, input_size: int, hidden_layer_sizes: list, output_size: int, kwargs: dict):
        self.input_size = input_size  
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size 
        
        self.bias = kwargs.get('bias', True)
        self.bias_weight = [1]
        self.weights = kwargs.get('weights', None) 
        self.biases = kwargs.get('biases', None) 

        self.weights = self.weights if self.weights else self.initialize_weights()
        self.biases = self.biases if self.biases else self.initialize_biases()
    
    def initialize_weights(self):
        """
        Initialize the weights to 1.0 for each connection in the network.
        The shape of weights is determined by the number of neurons in each layer.
        """
        all_weights = []
        layer_sizes = [self.input_size] + self.hidden_layer_sizes + [self.output_size]
        for i in range(len(layer_sizes) - 1):
            all_weights.append([[1.0 for _ in range(layer_sizes[i + 1])] for _ in range(layer_sizes[i])])
        return all_weights

    def initialize_biases(self):
        """
        Initialize the biases to 1.0 for each neuron in the hidden and output layers.
        """
        if not self.bias:
            return []
        
        all_biases = []
        layer_sizes =  [1] + self.hidden_layer_sizes + [self.output_size]
        for size in layer_sizes[1:]:
            all_biases.append([1.0] * size)  
        return all_biases
    
    def apply_layer(self, inputs, weights, biases):
        """
        Apply weights and biases to the input and compute the output for a single layer.
        """
        output = []
        for i in range(len(biases)):
            layer_input = sum(inputs[j] * weights[j][i] for j in range(len(inputs))) + self.bias_weight*biases[i]
            output.append(layer_input)
        return output
    
    def forward(self, inputs: list):
        layer_output = inputs
        for i in range(len(self.hidden_layer_sizes) + 1):
            if self.bias:
                self.bias_weight = 1  
            else:
                self.bias_weight = 0
            weights = self.weights[i]
            biases = self.biases[i] if self.bias else []  
            layer_output = self.apply_layer(layer_output, weights, biases)
        
        return layer_output
        

class TaskHandler():
    @staticmethod
    def run_task_1():
        config = {
            'bias': True,
            'weights': [[[0.5, 0.6], [0.2, -0.6]], [[0.8], [0.4]]], 
            'biases': [[0.3, 0.25], [-0.5]]  
        }

        nn = Network(input_size= 2, hidden_layer_sizes= [2], output_size= 1, kwargs = config)
        print("------ Model 1 ------")
        outputs = nn.forward([1.5, 0.5])
        print(outputs)
        outputs = nn.forward([0, 1])
        print(outputs)
        

    @staticmethod
    def run_task_2():
        config = {
            'bias': True,
            'weights': [[[0.5, 0.6], [1.5, -0.8]], [[0.6, 0.0],[-0.8,0.0]],[[0.5,-0.4]]], 
            'biases': [[0.3, 1.25], [0.3], [0.2,0.5]]  
        }

        network = Network(input_size= 2, hidden_layer_sizes= [2,1], output_size= 2, kwargs = config)
        print("------ Model 2 ------")
        outputs = network.forward([0.75, 1.25])
        print(outputs)
        outputs = network.forward([-1, 0.5])
        print(outputs)
        
if __name__ == "__main__":
    TaskHandler.run_task_1()
    TaskHandler.run_task_2()