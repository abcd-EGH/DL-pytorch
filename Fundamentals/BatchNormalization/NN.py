import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_batch_norm=True):
        super(NeuralNetwork, self).__init__()
        layers = []
        
        # First Layer
        layers.append(nn.Linear(input_size, hidden_size))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.Sigmoid())
        
        # Second Layer
        layers.append(nn.Linear(hidden_size, hidden_size))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.Sigmoid())
        
        # Third Layer
        layers.append(nn.Linear(hidden_size, hidden_size))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.Sigmoid())
        
        # Output Layer (Softmax 제거)
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten the input tensor
        x = x.view(x.size(0), -1)
        return self.model(x)