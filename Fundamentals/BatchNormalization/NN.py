import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_batch_norm=True):
        super(NeuralNetwork, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        
        # 첫 번째 레이어
        self.fc1 = nn.Linear(input_size, hidden_size)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_size)
        
        # 두 번째 레이어
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        if use_batch_norm:
            self.bn2 = nn.BatchNorm1d(hidden_size)
        
        # 세 번째 레이어
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        if use_batch_norm:
            self.bn3 = nn.BatchNorm1d(hidden_size)
        
        # 출력 레이어
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, return_activations=False):
        x = x.view(x.size(0), -1)
        activations = {}
        
        # FC1
        fc1_out = self.fc1(x)
        if return_activations:
            activations['fc1_out'] = fc1_out.detach().cpu()

        if self.use_batch_norm:
            bn1_out = self.bn1(fc1_out)
            if return_activations:
                activations['bn1_out'] = bn1_out.detach().cpu()
            x1_act = torch.sigmoid(bn1_out)
        else:
            x1_act = torch.sigmoid(fc1_out)
        
        # FC2
        fc2_out = self.fc2(x1_act)
        if return_activations:
            activations['fc2_out'] = fc2_out.detach().cpu()

        if self.use_batch_norm:
            bn2_out = self.bn2(fc2_out)
            if return_activations:
                activations['bn2_out'] = bn2_out.detach().cpu()
            x2_act = torch.sigmoid(bn2_out)
        else:
            x2_act = torch.sigmoid(fc2_out)
        
        # FC3
        fc3_out = self.fc3(x2_act)
        if return_activations:
            activations['fc3_out'] = fc3_out.detach().cpu()

        if self.use_batch_norm:
            bn3_out = self.bn3(fc3_out)
            if return_activations:
                activations['bn3_out'] = bn3_out.detach().cpu()
            x3_act = torch.sigmoid(bn3_out)
        else:
            x3_act = torch.sigmoid(fc3_out)
        
        # Output
        out = self.fc_out(x3_act)
        if return_activations:
            activations['out'] = out.detach().cpu()
        
        if return_activations:
            return out, activations
        else:
            return out
