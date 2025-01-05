import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import random

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs, device='cpu'):
    model.to(device)
    metrics = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)

        # 테스트 정확도
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(x_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        accuracy = correct / total
        metrics.append([avg_train_loss, accuracy])
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, Accuracy: {accuracy:.4f}")

    # --- 활성값(activation) 수집 + 시각화 ---
    model.eval()
    with torch.no_grad():
        x_test_batch, y_test_batch = next(iter(test_loader))
        x_test_batch = x_test_batch.to(device)
        _, activations = model(x_test_batch, return_activations=True)
    
    # (1) 모델별 하위 디렉토리 결정
    model_type = "with_bn" if model.use_batch_norm else "no_bn"
    batch_size = x_test_batch.size(0)
    model_result_dir = os.path.join("results", f"bs{batch_size}", model_type)
    os.makedirs(model_result_dir, exist_ok=True)

    print(f"\n===== Activation Distribution Analysis ({model_type}) =====")
    for key in activations:
        plt.figure(figsize=(5, 3))
        data = activations[key].flatten().cpu().numpy()
        
        # 평균, 표준편차
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        # 콘솔 출력
        print(f"[{model_type}] {key} - Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        
        # 히스토그램 저장 (model_result_dir 아래)
        plt.hist(data, bins=50, alpha=0.7, color='blue')
        plt.title(f"{model_type} - {key}\nMean={mean_val:.4f}, Std={std_val:.4f}")
        plt.xlabel("Activation value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(model_result_dir, f"activation_hist_{key}.png"))
        plt.close()

    return metrics

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)