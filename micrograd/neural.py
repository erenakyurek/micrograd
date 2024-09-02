import random
from micrograd.core import Value

class Neuron:

    def __init__(self, num_of_inputs):
        self.w = [Value(random.uniform(-1,1)) for _ in range(num_of_inputs)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:

    def __init__(self, num_of_inputs, num_of_output):
        self.neurons = [Neuron(num_of_inputs) for _ in range(num_of_output)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [parameter for individual_neuron in self.neurons for parameter in individual_neuron.parameters()]

class MLP:

    def __init__(self, num_of_inputs, num_of_outputs):
        size = [num_of_inputs] + num_of_outputs
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(num_of_outputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [parameter for individual_layer in self.layers for parameter in individual_layer.parameters()]