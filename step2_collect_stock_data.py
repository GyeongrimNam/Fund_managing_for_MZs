# ============================================================
# 2단계: pykrx로 종목 데이터 수집하기 (Google Colab에서 실행)
# ============================================================
# KOSPI + KOSDAQ 시가총액 상위 50개 종목의 기본 정보, 최근 1년 주가,
# PER/PBR/배당수익률, 업종(섹터) 분류를 수집해 stock_data.csv로 저장합니다.
#
# !pip install pykrx python-dotenv 를 먼저 실행하세요 (Colab 셀 1개).
#
# [Colab에서 KRX_ID/KRX_PW 설정하기]
# 왼쪽 사이드바의 열쇠(Secrets) 아이콘 -> KRX_ID, KRX_PW 추가 -> 각 항목의
# "Notebook access" 토글을 켠다. 노트북 내용에 비밀번호가 남지 않고, 공유해도
# 값이 함께 넘어가지 않아 .env 파일을 올리는 것보다 안전하다.

import os
import json
import time
import datetime

import pandas as pd
from dotenv import load_dotenv

# Colab Secrets가 있으면 그 값을 쓰고, 없으면 (로컬 실행 등) .env를 사용한다.
try:
    from google.colab import userdata

    for _key in ("KRX_ID", "KRX_PW"):
        try:
            os.environ[_key] = userdata.get(_key)
        except Exception:
            pass  # 해당 Secret이 없으면 무시하고 .env 로 폴백
except ImportError:
    pass

load_dotenv()  # Colab Secrets로 이미 설정됐다면 기존 값을 덮어쓰지 않음 (override=False)

from pykrx import stock
from pykrx.website.comm import auth as _pykrx_auth

# ------------------------------------------------------------
# 0. 로그인 시도 횟수 안전장치
# ------------------------------------------------------------
# pykrx는 인증되지 않은 상태에서 요청할 때마다 내부적으로 재로그인을 시도한다.
# 잘못된 계정 정보로 반복 요청하면 짧은 시간에 로그인 시도가 폭증해 KRX 계정이
# 잠길 수 있으므로, 프로세스 전체 기준 실제 로그인 시도를 3회로 제한한다.
MAX_LOGIN_ATTEMPTS = 3
_login_post_count = 1 if (os.getenv("KRX_ID") and os.getenv("KRX_PW")) else 0
_real_login_krx = _pykrx_auth.login_krx


def _guarded_login_krx(login_id, login_pw, session=None):
    global _login_post_count
    if _login_post_count >= MAX_LOGIN_ATTEMPTS:
        print(f"KRX 로그인 시도 상한({MAX_LOGIN_ATTEMPTS}회) 도달 - 추가 로그인 시도를 건너뜁니다.")
        return False
    _login_post_count += 1
    return _real_login_krx(login_id, login_pw, session)


_pykrx_auth.login_krx = _guarded_login_krx

TOP_N = 50
REQUEST_DELAY_SEC = 0.4  # 과도한 요청 방지를 위한 짧은 대기


# ------------------------------------------------------------
# 1. 최근 거래일 자동 탐색 (휴장일이면 하루씩 뒤로)
# ------------------------------------------------------------
def find_latest_trading_day(max_back_days=10):
    d = datetime.date.today()
    for _ in range(max_back_days):
        date_str = d.strftime("%Y%m%d")
        df = stock.get_market_cap_by_ticker(date_str, market="KOSPI")
        if not df.empty and (df["종가"] != 0).any():
            return date_str
        d -= datetime.timedelta(days=1)
    raise RuntimeError("최근 거래일을 찾지 못했습니다 (최대 탐색 일수 초과).")


# ------------------------------------------------------------
# 2. 시가총액 상위 50개 종목 선정 (KOSPI + KOSDAQ)
# ------------------------------------------------------------
def get_top_n_by_market_cap(date_str, top_n=TOP_N):
    frames = []
    for market in ["KOSPI", "KOSDAQ"]:
        df = stock.get_market_cap_by_ticker(date_str, market=market)
        df = df.copy()
        df["시장"] = market
        frames.append(df)
    combined = pd.concat(frames)
    combined.index.name = "종목코드"
    return combined.sort_values("시가총액", ascending=False).head(top_n)


# ------------------------------------------------------------
# 3. PER/PBR/배당수익률 조회
# ------------------------------------------------------------
def get_fundamentals(date_str):
    frames = []
    for market in ["KOSPI", "KOSDAQ"]:
        df = stock.get_market_fundamental_by_ticker(date_str, market=market)
        frames.append(df)
    combined = pd.concat(frames)
    combined.index.name = "종목코드"
    return combined


# ------------------------------------------------------------
# 4. 업종(섹터) 분류 조회
# ------------------------------------------------------------
# get_market_cap_by_ticker는 종목명을 주지 않으므로, 종목명도 여기서 함께 가져온다.
def get_sector_classifications(date_str):
    frames = []
    for market in ["KOSPI", "KOSDAQ"]:
        df = stock.get_market_sector_classifications(date_str, market=market)
        frames.append(df[["종목명", "업종명"]])
    combined = pd.concat(frames)
    combined.index.name = "종목코드"
    return combined


# ------------------------------------------------------------
# 5. 종목별 최근 1년 주가 수집
# ------------------------------------------------------------
def get_1y_price_history(ticker, end_date_str):
    end_date = datetime.datetime.strptime(end_date_str, "%Y%m%d").date()
    start_date = end_date - datetime.timedelta(days=365)
    df = stock.get_market_ohlcv(start_date.strftime("%Y%m%d"), end_date_str, ticker)
    if df is None or df.empty:
        raise ValueError("주가 데이터 없음")
    # 날짜/종가만 JSON으로 직렬화해 stock_data.csv 한 셀에 저장 (3단계에서 파싱해 사용)
    return json.dumps(
        [[idx.strftime("%Y-%m-%d"), int(close)] for idx, close in df["종가"].items()],
        ensure_ascii=False,
    )


# ------------------------------------------------------------
# 6. 전체 수집 실행
# ------------------------------------------------------------
def main():
    date_str = find_latest_trading_day()
    print(f"기준 거래일: {date_str}")

    top50 = get_top_n_by_market_cap(date_str)
    fundamentals = get_fundamentals(date_str)
    sectors = get_sector_classifications(date_str)

    rows = []
    total = len(top50)
    for i, (ticker, cap_row) in enumerate(top50.iterrows(), start=1):
        name = sectors.loc[ticker, "종목명"] if ticker in sectors.index else ""
        market = cap_row["시장"]
        market_cap = int(cap_row["시가총액"])
        volume = int(cap_row["거래량"])

        print(f"[{i}/{total}] {ticker} {name} 처리 중...")

        try:
            price_history = get_1y_price_history(ticker, date_str)
        except Exception as exc:
            print(f"  -> 실패, 건너뜀: {exc}")
            time.sleep(REQUEST_DELAY_SEC)
            continue

        per = fundamentals.loc[ticker, "PER"] if ticker in fundamentals.index else None
        pbr = fundamentals.loc[ticker, "PBR"] if ticker in fundamentals.index else None
        bps = fundamentals.loc[ticker, "BPS"] if ticker in fundamentals.index else None
        eps = fundamentals.loc[ticker, "EPS"] if ticker in fundamentals.index else None
        div_yield = fundamentals.loc[ticker, "DIV"] if ticker in fundamentals.index else None
        dps = fundamentals.loc[ticker, "DPS"] if ticker in fundamentals.index else None
        sector_name = sectors.loc[ticker, "업종명"] if ticker in sectors.index else "미분류"

        rows.append(
            {
                "종목코드": ticker,
                "종목명": name,
                "시장": market,
                "업종명": sector_name,
                "시가총액": market_cap,
                "거래량": volume,
                "최근1년종가": price_history,
                "PER": per,
                "PBR": pbr,
                "BPS": bps,
                "EPS": eps,
                "배당수익률": div_yield,
                "DPS": dps,
            }
        )

        time.sleep(REQUEST_DELAY_SEC)

    result_df = pd.DataFrame(rows)
    result_df.to_csv("stock_data.csv", index=False, encoding="utf-8-sig")
    print(f"\nstock_data.csv 저장 완료 (수집 성공: {len(result_df)} / {total})")
    print(result_df[["종목코드", "종목명", "시장", "업종명", "시가총액", "PER", "PBR", "BPS", "EPS", "배당수익률", "DPS"]].head(10))


if __name__ == "__main__":
    main()
