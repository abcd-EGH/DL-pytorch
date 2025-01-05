import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from NN import NeuralNetwork
from utils import set_seed, train_model

# 시드 설정
set_seed(seed=777)

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 전체 학습 과정을 함수로 묶기
def train_and_compare_models(learning_rate=0.001, epochs=10, batch_size=64):
    # 데이터 로드
    result_dir = "results/"
    os.makedirs(result_dir, exist_ok=True)  # 결과 디렉토리 생성

    # MNIST 데이터셋 로드
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)

    # DataLoader 생성
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 모델 설정
    input_size = 28 * 28
    hidden_size = 100
    output_size = 10

    # 모델: Batch Normalization 적용
    model_bn = NeuralNetwork(input_size, hidden_size, output_size, use_batch_norm=True)
    criterion_bn = nn.CrossEntropyLoss()
    optimizer_bn = optim.Adam(model_bn.parameters(), lr=learning_rate)

    # 모델: Batch Normalization 미적용
    model_no_bn = NeuralNetwork(input_size, hidden_size, output_size, use_batch_norm=False)
    criterion_no_bn = nn.CrossEntropyLoss()
    optimizer_no_bn = optim.Adam(model_no_bn.parameters(), lr=learning_rate)

    print("\nTraining model WITH Batch Normalization:")
    metrics_bn = train_model(model_bn, train_loader, test_loader, criterion_bn, optimizer_bn, epochs, device=device)

    print("\nTraining model WITHOUT Batch Normalization:")
    metrics_no_bn = train_model(model_no_bn, train_loader, test_loader, criterion_no_bn, optimizer_no_bn, epochs, device=device)

    # 결과 시각화
    metrics_bn = np.array(metrics_bn)
    metrics_no_bn = np.array(metrics_no_bn)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), metrics_bn[:, 1], label="With Batch Norm")
    plt.plot(range(1, epochs + 1), metrics_no_bn[:, 1], label="Without Batch Norm")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"MNIST Classification with PyTorch\nLearning Rate: {learning_rate}, Batch Size: {batch_size}, Epochs: {epochs}")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(result_dir, f"mnist_accuracy_comparison_lr{learning_rate}_bs{batch_size}.png"))
    plt.close()

if __name__ == "__main__":
    # 하이퍼파라미터 설정
    learning_rates = [0.01, 0.001, 0.0001]
    batch_sizes = [16, 32, 64, 128]
    epochs = 15

    # 모든 learning rate와 batch size 조합에 대해 실험
    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"\n실험 - Learning Rate: {lr}, Batch Size: {bs}")
            train_and_compare_models(lr, epochs, bs)