import os
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 하이퍼파라미터 및 경로 설정
LOOKBACK = 24         # 과거 24시간 데이터를 사용해 다음 값을 예측
BATCH_SIZE = 32
MODEL_TYPE = "LSTM"   # "LSTM", "GRU", "CNN" 중 선택

CHECKPOINT_DIR = "checkpoints"
RESULT_DIR = "result"
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

# 시퀀스 생성 함수
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
        return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.y[index], dtype=torch.float32)

# 모델 빌드 함수: model/ 폴더 내의 모듈에서 해당 모델 클래스를 import
def build_model(model_type, input_shape):
    input_size = input_shape[1]
    if model_type == "LSTM":
        from model.LSTM import LSTMModel
        model = LSTMModel(input_size)
    elif model_type == "GRU":
        from model.GRU import GRUModel
        model = GRUModel(input_size)
    elif model_type == "CNN":
        from model.CNN import CNNModel
        model = CNNModel(input_size)
    else:
        raise ValueError("정의되지 않은 모델 타입입니다: " + model_type)
    return model

def generate_predictions():
    # 테스트 데이터 파일 로드 (CSV 파일은 "date"와 "close" 컬럼을 포함)
    test_file = "/Users/jwmun/Desktop/최종_프로젝트/data/XRP_test.csv"
    test_df = pd.read_csv(test_file, parse_dates=["date"])
    test_df = test_df.sort_values("date").reset_index(drop=True)
    
    # close 값 추출 및 정규화 (학습 시 저장한 scaler 불러오기)
    test_close = test_df["close"].values.reshape(-1, 1)
    scaler_path = os.path.join(CHECKPOINT_DIR, f"scaler_{MODEL_TYPE}.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    test_scaled = scaler.transform(test_close)
    
    # 시퀀스 데이터 생성
    X_test, y_test = create_sequences(test_scaled, LOOKBACK)
    test_dataset = TimeSeriesDataset(test_scaled, LOOKBACK)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 모델 구성 및 가중치 로드
    input_shape = (LOOKBACK, 1)
    model = build_model(MODEL_TYPE, input_shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_model_path = os.path.join(CHECKPOINT_DIR, f"best_{MODEL_TYPE}.pt")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
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
    
    # 정규화 복원: inverse_transform
    pred_inversed = scaler.inverse_transform(predictions)
    true_inversed = scaler.inverse_transform(true_vals)
    
    # 예측 결과 DataFrame 생성 및 CSV 파일 저장
    result_df = pd.DataFrame({
        "Prediction": pred_inversed.flatten(),
        "Actual": true_inversed.flatten()
    })
    result_csv_path = os.path.join(RESULT_DIR, f"prediction_vs_actual_{MODEL_TYPE}.csv")
    result_df.to_csv(result_csv_path, index=False)
    print("Prediction results saved to", result_csv_path)
    return result_csv_path

def run_backtest(prediction_csv_path):
    # 예측 CSV 파일 로드
    df = pd.read_csv(prediction_csv_path)
    if df.empty:
        raise ValueError("CSV 파일이 비어있습니다: " + prediction_csv_path)
    
    # 백테스트 초기 설정
    initial_capital = 100.0  # 초기 자본
    capital = initial_capital
    position = 0.0           # 보유 수량
    state = "cash"           # 현재 상태: "cash" 또는 "long"
    results = []
    
    # 백테스트 진행: 전일 실제값이 필요하므로 index 1부터 시작
    for i in range(1, len(df)):
        prev_actual = df.loc[i - 1, "Actual"]
        curr_actual = df.loc[i, "Actual"]
        prediction = df.loc[i, "Prediction"]
        
        # 시그널 결정: 예측값이 전일 실제값보다 높으면 매수, 그렇지 않으면 매도
        signal = "buy" if prediction > prev_actual else "sell"
        action = "hold"
        
        if state == "cash" and signal == "buy":
            # 전량 매수: 현재 자본으로 매수
            position = capital / curr_actual
            capital = 0.0
            state = "long"
            action = "buy"
        elif state == "long" and signal == "sell":
            # 전량 매도: 보유 수량 전량 매도
            capital = position * curr_actual
            position = 0.0
            state = "cash"
            action = "sell"
        # 이미 long 상태에서 buy 시그널이면 그대로 보유 (hold), cash 상태에서 sell이면 아무 행동도 없음
        
        portfolio_value = position * curr_actual if state == "long" else capital
        
        results.append({
            "Day": i,
            "Prediction": prediction,
            "Actual": curr_actual,
            "Prev_Actual": prev_actual,
            "Signal": signal,
            "Action": action,
            "Portfolio Value": portfolio_value
        })
    
    # 마지막 날에도 보유 중이면 마지막 가격으로 매도 처리
    if state == "long":
        last_price = df.loc[len(df) - 1, "Actual"]
        capital = position * last_price
        position = 0.0
        state = "cash"
        portfolio_value = capital
        results.append({
            "Day": len(df) - 1,
            "Prediction": df.loc[len(df) - 1, "Prediction"],
            "Actual": last_price,
            "Prev_Actual": df.loc[len(df) - 2, "Actual"],
            "Signal": "sell_final",
            "Action": "sell",
            "Portfolio Value": portfolio_value
        })
    
    final_return = (capital - initial_capital) / initial_capital * 100
    print(f"최종 포트폴리오 가치: {capital:.2f}")
    print(f"수익률: {final_return:.2f}%")
    
    backtest_result_df = pd.DataFrame(results)
    backtest_result_csv = os.path.join(RESULT_DIR, "backtest_result.csv")
    backtest_result_df.to_csv(backtest_result_csv, index=False)
    print("백테스트 결과가 저장되었습니다:", backtest_result_csv)

def main():
    # 예측 결과 CSV 파일 생성
    prediction_csv_path = generate_predictions()
    # 생성된 CSV 파일을 바탕으로 백테스트 진행
    run_backtest(prediction_csv_path)

if __name__ == "__main__":
    main()
