import random
import numpy as np
import torch

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 학습 및 평가 함수
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs, device="cpu"):
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

        # 테스트 정확도 측정
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(x_batch)
                
                # 예측 확률 계산 시 Softmax 적용
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probabilities, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        accuracy = correct / total
        metrics.append([avg_train_loss, accuracy])

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, Accuracy: {accuracy:.4f}")

    return metrics