import pandas as pd
import streamlit as st

from data import FAILURE_MESSAGES, KRX_CREDENTIALS_CONFIGURED, fetch_price_data
from recommend import build_recommendation
from sectors import SECTORS

st.set_page_config(page_title="MZ 예산 맞춤 투자 추천", page_icon="📈", layout="wide")

st.title("📈 예산 맞춤 섹터별 종목 추천")
st.caption(
    "예산과 관심 섹터 비중을 입력하면, 섹터별로 시가총액(안정성) + 최근 등락률(모멘텀)을 "
    "함께 고려한 추천 종목과 매수 수량을 계산해줍니다."
)

st.warning(
    "⚠️ 이 앱은 투자 자문이 아닌 참고용 데모입니다. 실제 투자 판단과 책임은 본인에게 있습니다. "
    "시세 데이터는 지연 데이터일 수 있습니다.",
    icon="⚠️",
)

if not KRX_CREDENTIALS_CONFIGURED:
    st.info(
        "KRX 계정 정보가 설정되지 않아 실 시세 대신 예시(mock) 데이터를 보여주고 있습니다. "
        "[data.krx.co.kr](https://data.krx.co.kr)에서 무료 회원가입 후 `.env` 파일에 "
        "`KRX_ID`, `KRX_PW`를 입력하면 실 시세를 사용할 수 있습니다.",
        icon="ℹ️",
    )

st.header("1️⃣ 투자 예산 입력")
budget = st.number_input(
    "총 투자 예산 (원)", min_value=10_000, max_value=1_000_000_000, value=1_000_000, step=10_000
)

st.header("2️⃣ 관심 섹터 및 비중 설정")
all_sectors = list(SECTORS.keys())
selected_sectors = st.multiselect("관심 섹터를 선택하세요", all_sectors, default=all_sectors[:4])

sector_weights = {}
if selected_sectors:
    st.write("섹터별 비중(%)을 조절하세요. 합계는 자동으로 정규화됩니다.")
    default_weight = round(100 / len(selected_sectors))
    cols = st.columns(len(selected_sectors))
    for col, sector_name in zip(cols, selected_sectors):
        with col:
            sector_weights[sector_name] = st.slider(
                sector_name, min_value=0, max_value=100, value=default_weight, step=5
            )
    weight_sum = sum(sector_weights.values())
    if weight_sum == 0:
        st.error("최소 한 섹터의 비중은 0보다 커야 합니다.")
    else:
        st.info(f"입력된 비중 합계: {weight_sum}% → 자동으로 100%에 맞춰 정규화됩니다.")
else:
    st.info("섹터를 1개 이상 선택해주세요.")

top_n = st.slider("섹터당 추천 종목 수", min_value=1, max_value=3, value=2)

st.header("3️⃣ 추천 결과")

if st.button("🔍 추천 받기", type="primary", disabled=not selected_sectors):
    with st.spinner("최신 시세를 불러오는 중입니다..."):
        tickers = tuple(item["ticker"] for sector in selected_sectors for item in SECTORS[sector])
        price_data, fallback_reasons = fetch_price_data(tickers)

    if fallback_reasons:
        reason_lines = "\n".join(
            f"- {FAILURE_MESSAGES.get(reason, reason)}: {count}개 종목"
            for reason, count in fallback_reasons.items()
        )
        st.info(
            f"일부 종목({sum(fallback_reasons.values())}개)의 실 시세를 불러오지 못해 모의(mock) 데이터로 "
            f"대체했습니다. 사유:\n{reason_lines}",
            icon="ℹ️",
        )

    if "login_attempts_exceeded" in fallback_reasons:
        st.error(
            "KRX 로그인 실패가 반복되어 계정 잠금을 막기 위해 자동 로그인 시도를 중단했습니다. "
            "지금 다시 눌러도 재시도하지 않습니다. data.krx.co.kr에서 계정 잠금 상태와 비밀번호를 "
            "확인한 뒤, `.env`를 수정하고 앱을 재시작(streamlit 프로세스 재실행)해야 다시 로그인을 시도합니다.",
            icon="🔒",
        )

    result = build_recommendation(
        total_budget=budget,
        sector_weights=sector_weights,
        price_data=price_data,
        top_n_per_sector=top_n,
    )

    summary_rows = []
    for sector_name, df in result["sector_results"].items():
        st.subheader(f"📊 {sector_name}")
        if df.empty:
            st.write("예산이 부족하거나 1주도 매수할 수 없는 종목만 있어 추천 결과가 없습니다.")
            continue

        display_df = df[["name", "ticker", "price", "momentum_pct", "shares", "invested"]].copy()
        display_df.columns = ["종목명", "종목코드", "현재가(원)", "최근등락률(%)", "매수수량(주)", "투자금액(원)"]
        display_df["현재가(원)"] = display_df["현재가(원)"].map(lambda x: f"{x:,.0f}")
        display_df["투자금액(원)"] = display_df["투자금액(원)"].map(lambda x: f"{x:,.0f}")
        st.dataframe(display_df, hide_index=True, use_container_width=True)

        for _, row in df.iterrows():
            summary_rows.append({"섹터": sector_name, "종목명": row["name"], "투자금액": row["invested"]})

    st.divider()
    st.subheader("💰 전체 요약")
    c1, c2, c3 = st.columns(3)
    c1.metric("총 예산", f"{budget:,.0f}원")
    c2.metric("총 투자금액", f"{result['total_invested']:,.0f}원")
    c3.metric("미투자 잔액", f"{result['leftover_cash']:,.0f}원")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        chart_df = summary_df.groupby("섹터")["투자금액"].sum()
        st.bar_chart(chart_df)
else:
    st.write("예산과 섹터 비중을 설정한 뒤 '추천 받기' 버튼을 눌러주세요.")
