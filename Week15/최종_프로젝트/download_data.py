import yfinance as yf
import pandas as pd
import datetime

# 티커 설정
ticker = "BTC-USD"

# 날짜 범위 설정: 오늘 기준 1년 전부터 오늘까지
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=365)

# 한 번에 최대 59일치씩 다운로드 (60일 제한에 안전하게 맞추기 위해 59일)
chunk_days = 59
dfs = []
current_start = start_date

while current_start < end_date:
    # chunk의 끝 날짜 계산 (end 파라미터는 제외되므로 +1일)
    current_end = current_start + datetime.timedelta(days=chunk_days)
    if current_end > end_date:
        current_end = end_date
    # yfinance의 download() 함수는 end 날짜를 포함하지 않으므로, current_end+1일로 설정
    df_chunk = yf.download(
        ticker,
        start=current_start.strftime("%Y-%m-%d"),
        end=(current_end + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1h"
    )
    if not df_chunk.empty:
        dfs.append(df_chunk)
    current_start = current_end + datetime.timedelta(days=1)

# 여러 구간의 데이터를 하나의 DataFrame으로 합치고 중복 인덱스 제거
data = pd.concat(dfs)
data = data[~data.index.duplicated(keep='first')]

# CSV 파일로 저장
data.to_csv("BTC-USD_1year_hourly.csv")
print("다운로드 완료: BTC-USD_1year_hourly.csv")
