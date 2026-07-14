import datetime
import os
import random

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

# Must run before `pykrx` is imported anywhere in the process: pykrx reads
# KRX_ID/KRX_PW at import time to establish its data.krx.co.kr login session.
load_dotenv()

KRX_ID = os.getenv("KRX_ID")
KRX_PW = os.getenv("KRX_PW")
KRX_CREDENTIALS_CONFIGURED = bool(KRX_ID and KRX_PW)

try:
    from pykrx import stock as pykrx_stock
    from pykrx.website.comm import auth as _pykrx_auth
except ImportError:
    pykrx_stock = None
    _pykrx_auth = None

FAILURE_MESSAGES = {
    "no_credentials": "KRX 계정 정보(KRX_ID/KRX_PW) 미설정",
    "network_error": "네트워크 연결 오류",
    "login_or_no_data": "KRX 로그인 실패 또는 해당일 시세 데이터 없음(휴장일일 수 있음)",
    "login_attempts_exceeded": "로그인 시도 3회 초과로 자동 중단됨(계정 잠금 방지)",
    "unknown_error": "알 수 없는 오류",
}

MAX_LOGIN_ATTEMPTS = 3

# Importing pykrx already performs one real login POST if KRX_ID/KRX_PW are
# set (pykrx/website/comm/webio.py runs `build_krx_session()` at module load).
# We can't intercept that first attempt, but pykrx retries login on every
# single unauthenticated request afterward, so without a hard cap here one
# button click can fire off dozens of login POSTs and lock the KRX account.
# We patch the actual login function so every attempt past MAX_LOGIN_ATTEMPTS
# is skipped without touching the network.
_login_post_count = 1 if (KRX_CREDENTIALS_CONFIGURED and _pykrx_auth is not None) else 0

if _pykrx_auth is not None:
    _real_login_krx = _pykrx_auth.login_krx

    def _guarded_login_krx(login_id, login_pw, session=None):
        global _login_post_count
        if _login_post_count >= MAX_LOGIN_ATTEMPTS:
            print(f"KRX 로그인 시도 상한({MAX_LOGIN_ATTEMPTS}회) 도달 - 추가 로그인 시도를 건너뜁니다.")
            return False
        _login_post_count += 1
        return _real_login_krx(login_id, login_pw, session)

    _pykrx_auth.login_krx = _guarded_login_krx


def login_attempts_exhausted():
    return _login_post_count >= MAX_LOGIN_ATTEMPTS


def _recent_business_day(base=None):
    d = base or datetime.date.today()
    while d.weekday() >= 5:
        d -= datetime.timedelta(days=1)
    return d.strftime("%Y%m%d")


def _mock_row(ticker):
    rng = random.Random(ticker)
    price = rng.randint(10, 500) * 1000
    market_cap = price * rng.randint(1_000_000, 500_000_000)
    momentum_pct = round(rng.uniform(-15, 25), 2)
    return {"price": price, "market_cap": market_cap, "momentum_pct": momentum_pct}


def _classify_failure(exc):
    if isinstance(exc, requests.exceptions.RequestException):
        return "network_error"
    if isinstance(exc, (KeyError, ValueError)):
        return "login_or_no_data"
    return "unknown_error"


def _mock_all(tickers, reason):
    fallback_reasons = {reason: len(tickers)}
    rows = []
    for ticker in tickers:
        mock = _mock_row(ticker)
        mock["data_source"] = "mock"
        mock["fallback_reason"] = reason
        mock["ticker"] = ticker
        rows.append(mock)
    return rows, fallback_reasons


@st.cache_data(ttl=600, show_spinner=False)
def fetch_price_data(tickers):
    """Fetch price/market-cap/momentum data for the given tickers.

    Returns (DataFrame, fallback_reason_counts). Any ticker pykrx can't
    resolve (no credentials, login failure, network error, holiday with no
    data) falls back to deterministic mock data tagged with the reason. Once
    MAX_LOGIN_ATTEMPTS pykrx calls have failed, pykrx is skipped entirely for
    the rest of the process to avoid hammering KRX login and locking the
    account further.
    """
    columns = ["ticker", "price", "market_cap", "momentum_pct", "data_source", "fallback_reason"]
    today_str = _recent_business_day()
    past_str = _recent_business_day(datetime.date.today() - datetime.timedelta(days=28))

    if pykrx_stock is None:
        rows, fallback_reasons = _mock_all(tickers, "unknown_error")
        return pd.DataFrame(rows)[columns], fallback_reasons

    if not KRX_CREDENTIALS_CONFIGURED:
        rows, fallback_reasons = _mock_all(tickers, "no_credentials")
        return pd.DataFrame(rows)[columns], fallback_reasons

    if login_attempts_exhausted():
        rows, fallback_reasons = _mock_all(tickers, "login_attempts_exceeded")
        return pd.DataFrame(rows)[columns], fallback_reasons

    # Fetch the whole market's cap/price table once, not per ticker: pykrx
    # retries login on every request, so calling this inside the loop below
    # would multiply login attempts by len(tickers).
    try:
        cap_df = pykrx_stock.get_market_cap_by_ticker(today_str)
    except Exception as exc:
        rows, fallback_reasons = _mock_all(tickers, _classify_failure(exc))
        return pd.DataFrame(rows)[columns], fallback_reasons

    rows = []
    fallback_reasons = {}

    for ticker in tickers:
        if login_attempts_exhausted():
            mock = _mock_row(ticker)
            mock["data_source"] = "mock"
            mock["fallback_reason"] = "login_attempts_exceeded"
            mock["ticker"] = ticker
            rows.append(mock)
            fallback_reasons["login_attempts_exceeded"] = fallback_reasons.get("login_attempts_exceeded", 0) + 1
            continue

        row = None
        reason = None
        try:
            ohlcv_now = pykrx_stock.get_market_ohlcv(past_str, today_str, ticker)
            if ticker in cap_df.index and not ohlcv_now.empty:
                price = int(cap_df.loc[ticker, "종가"])
                market_cap = int(cap_df.loc[ticker, "시가총액"])
                start_price = float(ohlcv_now["종가"].iloc[0])
                end_price = float(ohlcv_now["종가"].iloc[-1])
                momentum_pct = round((end_price - start_price) / start_price * 100, 2)
                row = {
                    "price": price,
                    "market_cap": market_cap,
                    "momentum_pct": momentum_pct,
                    "data_source": "pykrx",
                    "fallback_reason": None,
                }
            else:
                reason = "login_or_no_data"
        except Exception as exc:
            reason = _classify_failure(exc)

        if row is None:
            mock = _mock_row(ticker)
            mock["data_source"] = "mock"
            mock["fallback_reason"] = reason
            row = mock
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1

        row["ticker"] = ticker
        rows.append(row)

    return pd.DataFrame(rows)[columns], fallback_reasons
