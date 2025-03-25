import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# 하이퍼파라미터 설정
LOOKBACK = 24         # 과거 24시간 데이터를 사용해 다음 값을 예측
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
MODEL_TYPE = "LSTM"   # "LSTM", "GRU", "CNN" 등으로 변경 가능

# 체크포인트 폴더 생성 (없으면)
CHECKPOINT_DIR = "checkpoints"
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# 결과 이미지 저장 폴더 생성
RESULT_IMAGE_DIR = "result_image"
if not os.path.exists(RESULT_IMAGE_DIR):
    os.makedirs(RESULT_IMAGE_DIR)

# 데이터 시퀀스 생성 함수
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

# PyTorch Dataset 정의
class TimeSeriesDataset(Dataset):
    def __init__(self, data, lookback):
        self.X, self.y = create_sequences(data, lookback)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        # X: (lookback, features), y: (features,)
        return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.y[index], dtype=torch.float32)

# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        # 마지막 시점의 출력을 사용
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# GRU 모델 정의
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=50):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# CNN 모델 정의 (1D Convolution)
class CNNModel(nn.Module):
    def __init__(self, input_size, num_filters=64, kernel_size=3):
        super(CNNModel, self).__init__()
        # Conv1d 입력: (batch_size, channels, seq_len) -> channels: input_size
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        # 출력 시퀀스 길이 계산: LOOKBACK - kernel_size + 1
        conv_output_length = LOOKBACK - kernel_size + 1
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_filters * conv_output_length, 1)
    
    def forward(self, x):
        # x: (batch_size, seq_len, features)
        x = x.permute(0, 2, 1)  # (batch_size, features, seq_len)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 모델 빌드 함수 (모델 타입에 따라)
def build_model(model_type, input_shape):
    # input_shape: (LOOKBACK, features)
    input_size = input_shape[1]  # features 차원 (여기서는 1)
    if model_type == "LSTM":
        return LSTMModel(input_size)
    elif model_type == "GRU":
        return GRUModel(input_size)
    elif model_type == "CNN":
        return CNNModel(input_size)
    else:
        raise ValueError("정의되지 않은 모델 타입입니다: " + model_type)

# CSV 파일 로드 및 전처리 함수
def load_and_preprocess(train_file, test_file, lookback):
    # CSV 파일 읽기 (date, close 컬럼 존재)
    train_df = pd.read_csv(train_file, parse_dates=["date"])
    test_df = pd.read_csv(test_file, parse_dates=["date"])
    
    # 날짜 순으로 정렬
    train_df = train_df.sort_values("date").reset_index(drop=True)
    test_df = test_df.sort_values("date").reset_index(drop=True)
    
    # close 값 추출
    train_close = train_df["close"].values.reshape(-1, 1)
    test_close = test_df["close"].values.reshape(-1, 1)
    
    # 학습 데이터 기준으로 MinMaxScaler 피팅 후 변환
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_close)
    test_scaled = scaler.transform(test_close)
    
    # PyTorch Dataset 생성
    train_dataset = TimeSeriesDataset(train_scaled, lookback)
    test_dataset = TimeSeriesDataset(test_scaled, lookback)
    
    # 시퀀스 생성으로 인해 offset된 테스트 날짜 (시각화를 위해)
    test_dates = test_df["date"].values[lookback:]
    
    return train_dataset, test_dataset, scaler, test_dates

# 예측 결과 시각화 함수 (이미지를 result_image 폴더에 저장)
def plot_predictions(dates, true_values, predictions, metric_str="", save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, true_values, label="True Price")
    plt.plot(dates, predictions, label="Predicted Price")
    plt.xlabel("Date")
    plt.ylabel("BTC Close Price")
    plt.title(f"BTC Price Prediction\n{metric_str}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()

def main():
    # 파일 경로 지정
    train_file = "/Users/jwmun/Desktop/최종_프로젝트/data/XRP_train.csv"
    test_file = "/Users/jwmun/Desktop/최종_프로젝트/data/XRP_test.csv"
    
    # 데이터 로드 및 전처리
    train_dataset, test_dataset, scaler, test_dates = load_and_preprocess(train_file, test_file, LOOKBACK)
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 모델 구성 (입력 shape: (LOOKBACK, 1))
    input_shape = (LOOKBACK, 1)
    model = build_model(MODEL_TYPE, input_shape)
    
    # GPU 사용 가능하면 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 손실함수 및 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(CHECKPOINT_DIR, f"best_{MODEL_TYPE}.pt")
    
    # 학습 루프
    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # 검증 (테스트 데이터)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # 검증 손실이 개선되면 모델 가중치 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model at epoch {epoch+1} with val loss {avg_val_loss:.4f}")
    
    # 가장 좋은 모델 가중치 로드
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    # 테스트 데이터에 대해 예측 수행
    predictions = []
    true_vals = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.append(outputs.cpu().numpy())
            true_vals.append(y_batch.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    
    # 예측 결과와 실제 값을 원래 스케일로 복원
    pred_inversed = scaler.inverse_transform(predictions)
    true_inversed = scaler.inverse_transform(true_vals)
    
    # 최종 테스트 성능 지표 계산
    final_mse = np.mean((pred_inversed - true_inversed)**2)
    final_mae = np.mean(np.abs(pred_inversed - true_inversed))
    metric_str = f"Test MSE: {final_mse:.4f}, Test MAE: {final_mae:.4f}"
    print(metric_str)
    
    # 마지막 테스트 기간에 대해 예측 결과 시각화 및 결과 이미지 저장
    save_path = os.path.join(RESULT_IMAGE_DIR, f"prediction_result_{MODEL_TYPE}.png")
    plot_predictions(test_dates, true_inversed.flatten(), pred_inversed.flatten(), metric_str, save_path)
    
    # 학습에 사용한 scaler도 함께 저장 (모델 예측 시 동일한 정규화 적용을 위해)
    scaler_path = os.path.join(CHECKPOINT_DIR, f"scaler_{MODEL_TYPE}.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print("Scaler saved to", scaler_path)

if __name__ == "__main__":
    main()
