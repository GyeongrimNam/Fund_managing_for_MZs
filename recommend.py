import math
import pandas as pd
from sectors import SECTORS


def _rank_score(series, ascending):
    ranks = series.rank(ascending=ascending, method="min")
    n = len(series)
    if n <= 1:
        return pd.Series([1.0] * n, index=series.index)
    return (n - ranks) / (n - 1)


def score_sector_stocks(df):
    df = df.copy()
    cap_score = _rank_score(df["market_cap"], ascending=False)
    momentum_score = _rank_score(df["momentum_pct"], ascending=False)
    df["score"] = (cap_score + momentum_score) / 2
    return df.sort_values("score", ascending=False)


def allocate_sector(sector_df, sector_budget, top_n=2):
    picked = sector_df.head(top_n).copy()
    if picked.empty or sector_budget <= 0:
        picked["alloc_budget"] = []
        picked["shares"] = []
        picked["invested"] = []
        return picked

    total_score = picked["score"].sum()
    if total_score <= 0:
        picked["weight"] = 1 / len(picked)
    else:
        picked["weight"] = picked["score"] / total_score

    picked["alloc_budget"] = picked["weight"] * sector_budget
    picked["shares"] = picked.apply(
        lambda r: math.floor(r["alloc_budget"] / r["price"]) if r["price"] > 0 else 0, axis=1
    )
    picked["invested"] = picked["shares"] * picked["price"]
    return picked[picked["shares"] > 0]


def build_recommendation(total_budget, sector_weights, price_data, top_n_per_sector=2):
    weight_sum = sum(sector_weights.values()) or 1
    normalized_weights = {k: v / weight_sum for k, v in sector_weights.items()}

    sector_results = {}
    total_invested = 0.0

    for sector_name, weight in normalized_weights.items():
        if weight <= 0:
            continue
        tickers = [item["ticker"] for item in SECTORS[sector_name]]
        names = {item["ticker"]: item["name"] for item in SECTORS[sector_name]}

        sector_df = price_data[price_data["ticker"].isin(tickers)].copy()
        sector_df["name"] = sector_df["ticker"].map(names)
        sector_df = score_sector_stocks(sector_df)

        sector_budget = total_budget * weight
        allocated = allocate_sector(sector_df, sector_budget, top_n=top_n_per_sector)

        sector_results[sector_name] = allocated
        total_invested += allocated["invested"].sum() if not allocated.empty else 0

    leftover_cash = total_budget - total_invested

    return {
        "sector_results": sector_results,
        "total_invested": total_invested,
        "leftover_cash": leftover_cash,
    }
