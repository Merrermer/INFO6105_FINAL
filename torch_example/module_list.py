import torch
import torch.nn as nn

class LinearBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        y = self.linear(x)
        print(f"{x} -> {y}, delta: {y - x}")
        return self.linear(x)

class MultiLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([LinearBlock() for _ in range(10)])

    def forward(self, x):
        for linear in self.linears:
            x = linear(x)
        return x

if __name__ == '__main__':
    model = MultiLinear()
    input = torch.randn(2)
    output = model(input)
    print(output)
    
    # ModuleList 比起 List
    # 会自动将子模块注册到父模块当中