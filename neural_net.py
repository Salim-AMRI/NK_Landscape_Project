import torch.nn as nn

### Policy net
class Net(nn.Module):

    def __init__(self, layers_size):
        super(Net, self).__init__()
        self.layers_size = layers_size
        layers = []

        for i in range(len(layers_size) - 1):
            layers.append(nn.Linear(layers_size[i], layers_size[i+1]))
            if(i != len(layers_size) - 2):
                layers.append(nn.Tanh())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        return self.layers(x)


    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()