import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 데이터 로드
import os
# motor_current_data.npz 파일이 src/ 폴더에 있다고 가정하고, 실행 위치에 상관없이 파일을 찾도록 경로 지정
data = np.load(os.path.join(os.path.dirname(__file__), "motor_current_data.npz"))
sequences = data["sequences"]     # (1200, 128), float32
labels = data["labels"]           # (1200,), int64 in {0, 1, 2}
class_names = list(data["class_names"])

# 시각화: 각 클래스별 예시 1개씩
plt.figure(figsize=(12, 4))
for i, cname in enumerate(class_names):
    idx = np.where(labels == i)[0][0]
    plt.plot(sequences[idx], label=cname)
plt.legend()
plt.title("Motor Current Waveforms (Example per Class)")
plt.xlabel("Time step")
plt.ylabel("Current")
plt.show()

# train/val/test 분할 (7:1.5:1.5)
X_train, X_temp, y_train, y_temp = train_test_split(sequences, labels, test_size=0.3, stratify=labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

def to_tensor(x, y):
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# LSTM/Transformer: (batch, seq_len, 1)
X_train_seq, y_train_t = to_tensor(X_train, y_train)
X_val_seq, y_val_t = to_tensor(X_val, y_val)
X_test_seq, y_test_t = to_tensor(X_test, y_test)
X_train_seq = X_train_seq.unsqueeze(-1)
X_val_seq = X_val_seq.unsqueeze(-1)
X_test_seq = X_test_seq.unsqueeze(-1)

# CNN: (batch, 1, seq_len)
X_train_cnn = X_train_seq.permute(0, 2, 1)  # (B, 1, 128)
X_val_cnn = X_val_seq.permute(0, 2, 1)
X_test_cnn = X_test_seq.permute(0, 2, 1)

train_seq_loader = DataLoader(TensorDataset(X_train_seq, y_train_t), batch_size=32, shuffle=True)
val_seq_loader = DataLoader(TensorDataset(X_val_seq, y_val_t), batch_size=32)
test_seq_loader = DataLoader(TensorDataset(X_test_seq, y_test_t), batch_size=32)

train_cnn_loader = DataLoader(TensorDataset(X_train_cnn, y_train_t), batch_size=32, shuffle=True)
val_cnn_loader = DataLoader(TensorDataset(X_val_cnn, y_val_t), batch_size=32)
test_cnn_loader = DataLoader(TensorDataset(X_test_cnn, y_test_t), batch_size=32)

# LSTM 모델
def get_lstm(input_dim=1, hidden_dim=32, num_layers=1, num_classes=3):
    class LSTMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, num_classes)
        def forward(self, x):
            # 입력 x는 이미 (B, 128, 1) 형태임
            _, (h_n, _) = self.lstm(x)
            out = self.fc(h_n[-1])
            return out
    return LSTMClassifier()

# 1D CNN 모델
def get_cnn(num_classes=3):
    class CNN1DClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
            self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(64, num_classes)
        def forward(self, x):
            # 입력 x는 이미 (B, 1, 128) 형태임
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = self.pool(x).squeeze(-1)
            out = self.fc(x)
            return out
    return CNN1DClassifier()

# Transformer 모델
def get_transformer(input_dim=1, d_model=32, nhead=4, num_layers=2, num_classes=3):
    class TransformerClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_encoding = nn.Parameter(torch.randn(1, 128, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64, batch_first=True, dropout=0.1)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.fc = nn.Linear(d_model, num_classes)
        def forward(self, x):
            # 입력 x는 이미 (B, 128, 1) 형태임
            x = self.input_proj(x)  # (B, 128, d_model)
            x = x + self.pos_encoding[:, :x.size(1), :]
            x = self.transformer(x)  # (B, 128, d_model)
            x = x.mean(dim=1)  # (B, d_model)
            out = self.fc(x)
            return out
    return TransformerClassifier()

# 학습 및 평가 루프
def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        print(f"Epoch {epoch+1}, Val Acc: {val_acc:.4f}")
    return model

def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            pred = out.argmax(dim=1)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(yb.cpu().numpy())
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")
    return np.concatenate(all_preds), np.concatenate(all_labels)

if __name__ == "__main__":
    print("\n--- LSTM ---")
    lstm_model = get_lstm()
    lstm_model = train_model(lstm_model, train_seq_loader, val_seq_loader)
    test_model(lstm_model, test_seq_loader)

    print("\n--- 1D CNN ---")
    cnn_model = get_cnn()
    cnn_model = train_model(cnn_model, train_cnn_loader, val_cnn_loader)
    test_model(cnn_model, test_cnn_loader)

    print("\n--- Transformer ---")
    transformer_model = get_transformer()
    transformer_model = train_model(transformer_model, train_seq_loader, val_seq_loader)
    test_model(transformer_model, test_seq_loader)
