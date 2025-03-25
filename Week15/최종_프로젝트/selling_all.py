import os
import time
import requests
import jwt
import uuid
import hashlib
from urllib.parse import urlencode
from dotenv import load_dotenv

# .env 파일에서 API 키 로드
load_dotenv()
UPBIT_ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY")
UPBIT_SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")

BASE_URL = "https://api.upbit.com/v1"

def generate_headers(query=None):
    """업비트 API 요청에 필요한 JWT 인증 헤더 생성"""
    payload = {
        "access_key": UPBIT_ACCESS_KEY,
        "nonce": str(uuid.uuid4()),
    }
    if query:
        query_string = urlencode(query).encode()
        m = hashlib.sha512()
        m.update(query_string)
        payload["query_hash"] = m.hexdigest()
        payload["query_hash_alg"] = "SHA512"
    token = jwt.encode(payload, UPBIT_SECRET_KEY, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}

def get_accounts():
    """계좌 정보를 조회하여 각 코인의 잔고 반환 (KRW 제외)"""
    url = f"{BASE_URL}/accounts"
    headers = generate_headers()
    response = requests.get(url, headers=headers)
    accounts = response.json()
    return accounts

def place_market_order(market, side, volume=None, price=None):
    """시장가 주문 실행 (매도는 volume, 매수는 price 기준)"""
    url = f"{BASE_URL}/orders"
    data = {
        "market": market,
        "side": side,
        "ord_type": "market",  # 시장가 주문
    }
    if side == "ask" and volume is not None:
        data["volume"] = str(volume)
    elif side == "bid" and price is not None:
        data["price"] = str(price)
        data["ord_type"] = "price"
    headers = generate_headers(data)
    response = requests.post(url, json=data, headers=headers)
    return response.json()

def sell_all_coins():
    """계좌에 보유한 모든 코인을 전량 매도하는 함수 (KRW 제외)"""
    accounts = get_accounts()
    if not accounts:
        print("계좌 정보를 불러오지 못했습니다.")
        return

    for account in accounts:
        currency = account.get("currency")
        # 원화(KRW)는 제외
        if currency == "KRW":
            continue

        balance = float(account.get("balance", 0))
        if balance > 0:
            market = f"KRW-{currency}"
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {currency} 보유량 {balance} 전량 매도 시도 (마켓: {market})")
            order_result = place_market_order(market, "ask", volume=balance)
            print(f"주문 응답: {order_result}")
            # 주문 간 잠시 대기 (API rate limit 고려)
            time.sleep(1)
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {currency} 잔고 0, 매도 건너뜀.")

if __name__ == "__main__":
    # print(get_accounts())
    sell_all_coins()