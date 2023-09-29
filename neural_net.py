import torch.nn as nn
import torch as th

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



def InvariantModule(module_class, N, remix):
    """Takes a module class as input and return a module class that mixes batch elements
    :args module_class: class of a th.nn.Module
    :return: batched module class
    """

    class _InvariantModuleClass(th.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self._ind_module = module_class(
                *args,
                **kwargs,
            )
            if remix:
                self._mean_module = module_class(*args, **kwargs)

        def forward(self, *inputs):
            if remix:
                bs = N
                red_inputs = [th.mean(x, dim=1, keepdim=True) for x in inputs]
                return  self._ind_module(*inputs) + self._mean_module(*red_inputs)
                #return bs * self._ind_module(*inputs) / (bs + 1) + self._mean_module(
                #    *red_inputs
                #) / (bs + 1)
            return self._ind_module(*inputs)

        def reset_parameters(self):
            self._ind_module.reset_parameters()
            if remix:
                self._mean_module.reset_parameters()

    return _InvariantModuleClass


class InvariantNNet(nn.Module):
    def __init__(self,  N, remix, layers_size):
        super().__init__()
        layers = []

        self.N = N
        self.remix = remix

        InvariantLinear = InvariantModule(th.nn.Linear, self.N, self.remix)

        for i in range(len(layers_size) - 1):
            layers.append(InvariantLinear(layers_size[i], layers_size[i + 1]))
            if i != len(layers_size) - 2:
                layers.append(nn.Tanh())

        self.layers = nn.Sequential(*layers)

    def forward(self, s):
        #s = th.transpose(s, 1, 2)
        s = s.float()
        return self.layers(s)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
