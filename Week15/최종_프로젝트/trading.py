import os
import time
import numpy as np
import pandas as pd
import pickle
import torch
import pyupbit
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler

# ===============================
# 환경변수(.env) 로드 및 API 키 설정
# ===============================
load_dotenv()
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

# ===============================
# 설정 및 초기화
# ===============================
LOOKBACK = 24         # 최근 24시간 데이터를 사용 (1시간봉 기준)
MODEL_TYPE = "LSTM"   # "LSTM", "GRU", "CNN" 중 선택

CHECKPOINT_DIR = "checkpoints"
RESULT_DIR = "result"
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

# 거래 내역 로그 파일 (실제 거래 내역 기록)
LIVE_TRADE_LOG = os.path.join(RESULT_DIR, "real_trade_log.csv")
if not os.path.exists(LIVE_TRADE_LOG):
    pd.DataFrame(columns=["Timestamp", "Prediction", "Actual", "Signal", "Action", "Order Details"])\
      .to_csv(LIVE_TRADE_LOG, index=False)

# ===============================
# 모델 및 scaler 로드
# ===============================
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

# scaler 객체 로드
scaler_path = os.path.join(CHECKPOINT_DIR, f"scaler_{MODEL_TYPE}.pkl")
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

input_shape = (LOOKBACK, 1)
model = build_model(MODEL_TYPE, input_shape)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
best_model_path = os.path.join(CHECKPOINT_DIR, f"best_{MODEL_TYPE}.pt")
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

# ===============================
# 업비트 API 설정 (실제 거래)
# ===============================
upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)

# ===============================
# 업비트에서 최근 24시간 데이터 가져오기
# ===============================
def get_recent_24h_data():
    """
    pyupbit를 사용하여 KRW-BTC의 최근 24시간(1시간봉) 데이터를 가져옵니다.
    반환되는 DataFrame은 시간순(오름차순)으로 정렬됩니다.
    """
    df = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=LOOKBACK)
    return df

# ===============================
# 실제 거래 실행 루프 (1시간마다 실행)
# ===============================
print("실제 거래를 시작합니다. (주의: 실제 자금이 거래됩니다!)")
while True:
    try:
        # 1. 최근 24시간 데이터 가져오기
        df_ohlcv = get_recent_24h_data()
        if df_ohlcv is None or df_ohlcv.empty:
            print("데이터를 불러오지 못했습니다. 1분 후 재시도합니다.")
            time.sleep(60)
            continue

        close_prices = df_ohlcv["close"].values  # (24,) 배열
        # 현재 실제 가격: 가장 최신 1시간봉의 종가 사용
        current_actual = close_prices[-1]

        # 2. 데이터 정규화 및 모델 입력 형태 변환
        data_scaled = scaler.transform(close_prices.reshape(-1, 1))
        input_seq = data_scaled.reshape(1, LOOKBACK, 1)
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).to(device)
        
        # 3. 다음 1시간 가격 예측
        with torch.no_grad():
            pred_scaled = model(input_tensor).cpu().numpy()
        predicted_price = scaler.inverse_transform(pred_scaled)[0, 0]

        # 4. 거래 시그널 결정: 예측 가격이 현재 가격보다 높으면 "buy", 낮으면 "sell"
        signal = "buy" if predicted_price > current_actual else "sell"
        action = "hold"

        # 5. 현재 계좌 잔고 확인
        krw_balance = upbit.get_balance("KRW")
        btc_balance = upbit.get_balance("KRW-BTC")
        
        # 6. 실제 거래 실행 (시장가 주문)
        order_details = None
        krw_balance=10000
        # 현금 보유 중이고 "buy" 시그널이면 전량 매수 (최소 주문 금액: 보통 5,000원 이상)
        if signal == "buy" and krw_balance is not None and krw_balance > 5000:
            order_details = upbit.buy_market_order("KRW-BTC", krw_balance)
            action = "buy"
        # BTC 보유 중이고 "sell" 시그널이면, 현재 보유한 BTC 잔고(실제 잔고 확인 후)를 전량 매도
        elif signal == "sell":
            btc_balance = upbit.get_balance("KRW-BTC")  # 최신 BTC 잔고 재확인
            if btc_balance is not None and btc_balance > 0.0001:
                order_details = upbit.sell_market_order("KRW-BTC", btc_balance)
                action = "sell"
            else:
                action = "hold"
        
        # 7. 거래 내역 로그 기록
        current_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "Timestamp": current_time,
            "Prediction": predicted_price,
            "Actual": current_actual,
            "Signal": signal,
            "Action": action,
            "Order Details": order_details
        }
        print(log_entry)
        df_log = pd.DataFrame([log_entry])
        df_log.to_csv(LIVE_TRADE_LOG, mode="a", header=False, index=False)
        
        # 8. 다음 1시간(3600초) 대기
        time.sleep(3600)
    except Exception as e:
        print("오류 발생:", e)
        # 오류 발생 시 1분 후 재시도
        time.sleep(60)
