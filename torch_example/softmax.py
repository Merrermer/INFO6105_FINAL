import torch
from torch import nn
import math

if __name__ == '__main__':
    softmax = nn.Softmax(dim=-1)
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 10.0, 6.0]])
    output = softmax(input)
    print(output)
    
    print(math.exp(1.0)/sum([math.exp(1.0), math.exp(2.0), math.exp(3.0)]))
    print(math.exp(2.0)/sum([math.exp(1.0), math.exp(2.0), math.exp(3.0)]))
    print(math.exp(3.0)/sum([math.exp(1.0), math.exp(2.0), math.exp(3.0)]))

    print(math.exp(4.0)/sum([math.exp(4.0), math.exp(10.0), math.exp(6.0)]))
    print(math.exp(10.0)/sum([math.exp(4.0), math.exp(10.0), math.exp(6.0)]))
    print(math.exp(6.0)/sum([math.exp(4.0), math.exp(10.0), math.exp(6.0)]))