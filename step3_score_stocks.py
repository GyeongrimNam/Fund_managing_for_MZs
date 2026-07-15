# ============================================================
# 3단계: 종목 평가와 추천 점수 계산하기 (Google Colab에서 실행)
# ============================================================
# stock_data.csv를 읽어 수익률/변동성/MDD를 계산하고, 수익성·안정성·가치·배당
# 4개 점수(0~100)로 정규화한 뒤 투자 성향별 가중치로 추천 점수를 계산합니다.
#
# !pip install pandas 는 Colab에 기본 설치되어 있어 별도 설치가 필요 없습니다.

import json
import datetime

import numpy as np
import pandas as pd

INPUT_FILE = "stock_data.csv"
OUTPUT_FILE = "scored_stocks.csv"

# 투자 성향별 가중치 (수익성, 안정성, 가치, 배당)
PROFILE_WEIGHTS = {
    "안정형": {"수익성": 0.15, "안정성": 0.45, "가치": 0.15, "배당": 0.25},
    "중립형": {"수익성": 0.30, "안정성": 0.30, "가치": 0.25, "배당": 0.15},
    "공격형": {"수익성": 0.50, "안정성": 0.10, "가치": 0.30, "배당": 0.10},
}


# ------------------------------------------------------------
# 1. 가격 시계열 파싱 + 수익률/변동성/MDD 계산
# ------------------------------------------------------------
def parse_price_history(json_str):
    records = json.loads(json_str)
    dates = [datetime.date.fromisoformat(d) for d, _ in records]
    prices = [float(p) for _, p in records]
    return pd.Series(prices, index=pd.DatetimeIndex(dates)).sort_index()


def price_n_months_ago(series, end_date, months):
    target = end_date - pd.DateOffset(months=months)
    eligible = series[series.index <= target]
    if not eligible.empty:
        return eligible.iloc[-1]
    return series.iloc[0]  # 데이터가 그만큼 길지 않으면 가장 오래된 값으로 대체


def compute_return_pct(start_price, end_price):
    if start_price is None or start_price == 0 or pd.isna(start_price):
        return np.nan
    return (end_price - start_price) / start_price * 100


def compute_annualized_volatility_pct(series):
    daily_returns = series.pct_change().dropna()
    if len(daily_returns) < 2:
        return np.nan
    return daily_returns.std() * np.sqrt(252) * 100


def compute_mdd_pct(series):
    if series.empty:
        return np.nan
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    return abs(drawdown.min()) * 100  # 양수(%) 로 표현, 클수록 낙폭이 큼


def compute_metrics_for_row(price_history_json):
    series = parse_price_history(price_history_json)
    if series.empty:
        return pd.Series(
            {"현재가": np.nan, "수익률1년": np.nan, "수익률6개월": np.nan, "수익률3개월": np.nan,
             "연환산변동성": np.nan, "최대낙폭": np.nan}
        )

    end_date = series.index[-1]
    end_price = series.iloc[-1]

    r1y = compute_return_pct(series.iloc[0], end_price)
    r6m = compute_return_pct(price_n_months_ago(series, end_date, 6), end_price)
    r3m = compute_return_pct(price_n_months_ago(series, end_date, 3), end_price)
    vol = compute_annualized_volatility_pct(series)
    mdd = compute_mdd_pct(series)

    return pd.Series(
        {"현재가": end_price, "수익률1년": r1y, "수익률6개월": r6m, "수익률3개월": r3m,
         "연환산변동성": vol, "최대낙폭": mdd}
    )


# ------------------------------------------------------------
# 2. 퍼센타일 랭크 정규화 (0~100). higher_is_better=False면 방향을 뒤집는다.
# min-max는 극단치(예: 1년 수익률 900%대 종목) 하나 때문에 나머지 종목들의 점수가
# 실제 순위와 무관하게 다같이 낮게 눌리는 문제가 있어서, "전체 종목 중 몇 번째로
# 좋은가"라는 상대 순위(percentile) 기준으로 바꿨다. 극단치의 영향을 크게 줄여준다.
# ------------------------------------------------------------
def percentile_score(series, higher_is_better=True):
    s = series.astype(float)
    valid = s.dropna()
    if valid.empty or valid.nunique() <= 1:
        return pd.Series(50.0, index=s.index)  # 변별력이 없으면 중간값 부여

    pct_rank = s.rank(pct=True, ascending=higher_is_better)
    return (pct_rank * 100).clip(0, 100)


# ------------------------------------------------------------
# 3. 4대 점수 계산 (수익성/안정성/가치/배당)
# ------------------------------------------------------------
def compute_scores(df):
    # 결측값/비정상값 처리
    # - PER/PBR가 0 이하면(적자 등으로 의미 없는 값) 정규화에서 제외하고 최저점(0) 처리
    per = df["PER"].where(df["PER"] > 0)
    pbr = df["PBR"].where(df["PBR"] > 0)
    # - 배당수익률 0은 "배당 없음"이라는 유효한 정보이므로 결측 처리하지 않음
    div_yield = df["배당수익률"].fillna(0)

    # 수익성 점수: 최근 1년/6개월/3개월 수익률을 5:3:2로 합성한 뒤 정규화
    composite_return = (
        df["수익률1년"].fillna(0) * 0.5
        + df["수익률6개월"].fillna(0) * 0.3
        + df["수익률3개월"].fillna(0) * 0.2
    )
    profitability = percentile_score(composite_return, higher_is_better=True)

    # 안정성 점수: 변동성과 최대낙폭이 낮을수록 높은 점수, 두 지표를 평균
    vol_score = percentile_score(df["연환산변동성"], higher_is_better=False)
    mdd_score = percentile_score(df["최대낙폭"], higher_is_better=False)
    stability = (vol_score.fillna(0) + mdd_score.fillna(0)) / 2

    # 가치 점수: PER, PBR이 낮을수록 높은 점수, 두 지표를 평균 (0 이하 값은 최저점 0)
    per_score = percentile_score(per, higher_is_better=False).fillna(0)
    pbr_score = percentile_score(pbr, higher_is_better=False).fillna(0)
    value = (per_score + pbr_score) / 2

    # 배당 점수: 배당수익률이 높을수록 높은 점수
    dividend = percentile_score(div_yield, higher_is_better=True)

    df["수익성점수"] = profitability.round(1)
    df["안정성점수"] = stability.round(1)
    df["가치점수"] = value.round(1)
    df["배당점수"] = dividend.round(1)
    return df


# ------------------------------------------------------------
# 4. 투자 성향별 추천 점수 + 추천 이유
# ------------------------------------------------------------
def compute_profile_scores(df):
    for profile, w in PROFILE_WEIGHTS.items():
        df[f"추천점수_{profile}"] = (
            df["수익성점수"] * w["수익성"]
            + df["안정성점수"] * w["안정성"]
            + df["가치점수"] * w["가치"]
            + df["배당점수"] * w["배당"]
        ).round(1)
    return df


# 추천 이유를 "가치 점수 91점" 같은 추상적인 숫자가 아니라, 실제 지표값으로 풀어서
# 설명한다 (투자 초보자가 봤을 때 바로 와닿도록).
def _describe_profitability(row):
    r1y = row.get("수익률1년")
    if r1y is None or pd.isna(r1y):
        return "최근 수익률 흐름이 좋은 종목이에요."
    if r1y >= 0:
        return f"최근 1년간 주가가 {r1y:.0f}% 올랐어요."
    return f"최근 1년 수익률은 {r1y:.0f}%지만, 다른 지표들이 이를 상쇄할 만큼 우수해요."


# 변동성/PER/PBR은 퍼센타일 랭크라서 "데이터셋 안에서 몇 번째로 좋은가"만 반영한다
# (예: 전체 종목의 PER이 대체로 높으면 PER 150배도 상위권 점수를 받을 수 있다).
# 그래서 절대적으로도 괜찮은 수준인지 별도 기준으로 한 번 더 확인한 뒤 문장을
# 다르게 쓴다 - 그래야 "PER 150배인데 저평가"라는 식의 어색한 문장이 안 나온다.
STABLE_VOL_THRESHOLD = 30   # 연환산 변동성 30% 이하면 절대적으로도 안정적이라고 봄
STABLE_MDD_THRESHOLD = 25   # 최대낙폭 25% 이하면 절대적으로도 안정적이라고 봄
CHEAP_PER_THRESHOLD = 20    # PER 20배 이하면 절대적으로도 저평가라고 봄
CHEAP_PBR_THRESHOLD = 2     # PBR 2배 이하면 절대적으로도 저평가라고 봄


def _describe_stability(row):
    vol = row.get("연환산변동성")
    mdd = row.get("최대낙폭")
    parts = []
    if vol is not None and not pd.isna(vol):
        parts.append(f"연환산 변동성 {vol:.0f}%")
    if mdd is not None and not pd.isna(mdd):
        parts.append(f"최대낙폭 {mdd:.0f}%")
    if not parts:
        return "가격 흐름이 상대적으로 안정적인 종목이에요."

    label = ", ".join(parts)
    is_stable = (
        (vol is None or pd.isna(vol) or vol <= STABLE_VOL_THRESHOLD)
        and (mdd is None or pd.isna(mdd) or mdd <= STABLE_MDD_THRESHOLD)
    )
    if is_stable:
        return f"{label}로 가격 흐름이 안정적인 편이에요."
    return f"{label}인 종목이에요. (비슷한 종목들 중에서는 상대적으로 안정적인 편이에요)"


def _describe_value(row):
    per = row.get("PER")
    pbr = row.get("PBR")
    valid_per = per is not None and not pd.isna(per) and per > 0
    valid_pbr = pbr is not None and not pd.isna(pbr) and pbr > 0
    parts = []
    if valid_per:
        parts.append(f"PER {per:.1f}배")
    if valid_pbr:
        parts.append(f"PBR {pbr:.1f}배")
    if not parts:
        return "밸류에이션 매력이 있는 종목이에요."

    label = ", ".join(parts)
    is_cheap = (
        (not valid_per or per <= CHEAP_PER_THRESHOLD)
        and (not valid_pbr or pbr <= CHEAP_PBR_THRESHOLD)
    )
    if is_cheap:
        return f"{label}로 저평가된 편이에요."
    return f"{label}인 종목이에요. (비슷한 종목들 중에서는 상대적으로 저평가된 편이에요)"


def _describe_dividend(row):
    div_yield = row.get("배당수익률")
    if div_yield is not None and not pd.isna(div_yield) and div_yield > 0:
        return f"배당수익률 {div_yield:.2f}%로 꾸준한 배당을 기대할 수 있어요."
    return "배당 매력이 있는 종목이에요."


FACTOR_DESCRIBERS = {
    "수익성": _describe_profitability,
    "안정성": _describe_stability,
    "가치": _describe_value,
    "배당": _describe_dividend,
}


def build_reason(row, profile):
    scores = {"수익성": row["수익성점수"], "안정성": row["안정성점수"],
              "가치": row["가치점수"], "배당": row["배당점수"]}
    weights = PROFILE_WEIGHTS[profile]
    # 해당 성향 가중치가 반영된 기여도 기준으로 강점 지표 2개 선정
    contribution = {k: scores[k] * weights[k] for k in scores}
    top_factors = sorted(contribution, key=contribution.get, reverse=True)[:2]
    descriptions = [FACTOR_DESCRIBERS[f](row) for f in top_factors]
    return f"{profile} 투자자에게 잘 맞는 종목이에요. " + " ".join(descriptions)


# ------------------------------------------------------------
# 5. 실행
# ------------------------------------------------------------
def main():
    df = pd.read_csv(INPUT_FILE, dtype={"종목코드": str})

    metrics = df["최근1년종가"].apply(compute_metrics_for_row)
    df = pd.concat([df, metrics], axis=1)

    df = compute_scores(df)
    df = compute_profile_scores(df)

    for profile in PROFILE_WEIGHTS:
        df[f"추천이유_{profile}"] = df.apply(lambda r: build_reason(r, profile), axis=1)

    output_cols = [
        "종목코드", "종목명", "시장", "업종명", "현재가",
        "수익률1년", "수익률6개월", "수익률3개월", "연환산변동성", "최대낙폭",
        "PER", "PBR", "BPS", "EPS", "배당수익률", "DPS",
        "수익성점수", "안정성점수", "가치점수", "배당점수",
    ]
    for profile in PROFILE_WEIGHTS:
        output_cols += [f"추천점수_{profile}", f"추천이유_{profile}"]

    result = df[output_cols].round(2)
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"{OUTPUT_FILE} 저장 완료 (종목 수: {len(result)})")

    for profile in PROFILE_WEIGHTS:
        print(f"\n=== {profile} 추천 상위 10개 ===")
        top10 = result.sort_values(f"추천점수_{profile}", ascending=False).head(10)
        print(top10[["종목코드", "종목명", "업종명", f"추천점수_{profile}", f"추천이유_{profile}"]].to_string(index=False))


if __name__ == "__main__":
    main()
