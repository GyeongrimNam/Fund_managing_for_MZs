# ============================================================
# 4단계: Streamlit 웹앱 통합 (Google Colab에서 %%writefile app.py로 저장)
# ============================================================
# 3페이지 구성: 소개 페이지 -> 설문(+관심 섹터) 페이지 -> 결과 페이지
# 입력 파일: risk_model.pkl (1단계), scored_stocks.csv (3단계)

import math
import os
import re

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="투자성향 맞춤 종목 추천", page_icon="📊", layout="wide")


def inject_custom_css():
    st.markdown(
        """
        <style>
        :root {
            --accent: #00c896;
            --accent-dark: #00a67d;
            --card-bg: rgba(255, 255, 255, 0.03);
            --card-border: rgba(255, 255, 255, 0.12);
        }

        h1 { font-weight: 800 !important; letter-spacing: -0.5px; }
        h2, h3 { font-weight: 700 !important; }

        /* st.container(border=True)로 감싼 섹션을 카드처럼 표시 */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: var(--card-bg);
            border: 1px solid var(--card-border) !important;
            border-radius: 16px;
            padding: 0.4rem 0.4rem;
        }

        /* 기본 버튼 */
        .stButton > button {
            border-radius: 10px;
            border: 1px solid var(--card-border);
            transition: all 0.15s ease;
            font-weight: 600;
        }
        .stButton > button:hover {
            border-color: var(--accent);
            color: var(--accent);
        }

        /* primary 버튼(추천받기, 선택된 섹터 등) */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--accent), var(--accent-dark));
            border: none;
            color: #04241c;
            font-weight: 700;
        }
        .stButton > button[kind="primary"]:hover {
            filter: brightness(1.08);
            color: #04241c;
        }

        /* 메트릭 카드 */
        div[data-testid="stMetric"] {
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 0.9rem 1rem;
        }

        /* 경고/정보 배너, 데이터프레임 모서리 둥글게 */
        div[data-testid="stAlert"] { border-radius: 12px; }
        div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

        /* 라디오 문항을 pill 느낌으로 */
        div[role="radiogroup"] label {
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 2px 14px;
            margin-right: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


MODEL_FILE = "risk_model.pkl"
DATA_FILE = "scored_stocks.csv"
FEATURE_COLS = ["손실감수수준", "기대수익률", "투자기간", "배당선호도", "변동성감수수준"]

PROFILE_DESCRIPTIONS = {
    "안정형": "원금 손실을 최소화하고 배당·안정성을 중시하는 투자자입니다. 변동성이 낮고 꾸준한 배당을 주는 종목을 선호합니다.",
    "중립형": "수익과 안정성의 균형을 추구하는 투자자입니다. 성장성과 안정성을 함께 고려한 종목을 선호합니다.",
    "공격형": "높은 수익을 위해 변동성을 감수할 수 있는 투자자입니다. 성장성이 높은 종목을 선호합니다.",
}

# 투자 초보자를 위한 용어 설명 (지표 옆 ? 아이콘에 마우스를 올리면 표시됨)
TERM_HELP = {
    "PER": "주가를 주당순이익으로 나눈 값으로 PER이 낮을수록 기업이 내는 이익에 비해 주가가 저평가 되어 있다는 의미에요.",
    "PBR": "주가를 주당순자산으로 나눈 값으로, PBR이 낮을수록 기업의 실제 자산가치 대비 주가가 저평가 되어 있다는 의미에요.",
    "EPS": "1주당 회사가 벌어들인 순이익을 의미해요. 숫자가 클수록 회사의 기업 가치가 크고, 배당 줄 수 있는 여유가 늘어났다고 볼 수 있어요.",
    "BPS": "회사가 경영을 멈추고 현재 시점의 순자산을 주주들에게 나누어줄 경우, 한 주당 얼마씩 줄 수 있는지를 의미해요. 숫자가 커질수록 회사의 기업가치가 높다고 볼 수 있어요.",
    "ROE": "회사가 자기자본(주주지분)으로 1년간 얼마를 벌어들였는지 보여주는 지표예요. 부채를 통해 벌어들인 수익은 포함되지 않아요.",
    "배당수익률": "주가 대비 1년간 받을 수 있는 배당금의 비율이에요. 예금 이자율과 비슷하게, 주가는 그대로여도 매년 받을 수 있는 현금 수익의 비율을 뜻해요.",
    "주당배당금": "주식 1주를 가지고 있을 때 실제로 받는 배당금(현금)이에요.",
}


# ------------------------------------------------------------
# 캐시된 리소스/데이터 로딩
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data(show_spinner=False)
def load_scored_stocks():
    return pd.read_csv(DATA_FILE, dtype={"종목코드": str})


def files_missing():
    missing = [f for f in (MODEL_FILE, DATA_FILE) if not os.path.exists(f)]
    return missing


# ------------------------------------------------------------
# 페이지 전환 헬퍼 (세션 상태로 단일 스크립트 안에서 페이지 관리)
# ------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "intro"


def go_to(page_name):
    st.session_state.page = page_name


# ------------------------------------------------------------
# 공통: 투자 참고용 경고 문구
# ------------------------------------------------------------
def render_disclaimer():
    st.warning(
        "⚠️ 이 서비스는 투자 자문이 아닌 참고용 데모입니다. 실제 투자 판단과 책임은 본인에게 있습니다.",
        icon="⚠️",
    )


# ------------------------------------------------------------
# 예산 배분: 주어진 종목들끼리 추천점수 비례로 예산을 나누고 매수 수량 계산
# ------------------------------------------------------------
def allocate_budget(candidates_df, budget, score_col):
    df = candidates_df.copy()
    if df.empty:
        return df

    total_score = df[score_col].sum()
    if total_score <= 0:
        df["배분금액"] = budget / len(df)
    else:
        df["배분금액"] = df[score_col] / total_score * budget

    df["매수수량"] = df.apply(
        lambda r: math.floor(r["배분금액"] / r["현재가"]) if r["현재가"] > 0 else 0, axis=1
    )
    df["투자금액"] = df["매수수량"] * df["현재가"]
    return df


# 종목 1개에 대한 "투자 지표" 카드를 그린다 (가치평가/수익/배당, 용어 설명 툴팁 포함)
def render_investment_metrics_card(row):
    per = row.get("PER")
    pbr = row.get("PBR")
    bps = row.get("BPS")
    eps = row.get("EPS")
    div_yield = row.get("배당수익률")
    dps = row.get("DPS")
    roe = (eps / bps * 100) if bps and bps > 0 else None

    def fmt(value, suffix="", decimals=1):
        if value is None or pd.isna(value):
            return "-"
        return f"{value:,.{decimals}f}{suffix}"

    st.caption("투자 지표")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**가치평가**")
        st.metric("PER", fmt(per, "배"), help=TERM_HELP["PER"])
        st.metric("PBR", fmt(pbr, "배"), help=TERM_HELP["PBR"])
    with c2:
        st.markdown("**수익**")
        st.metric("EPS", fmt(eps, "원", 0), help=TERM_HELP["EPS"])
        st.metric("BPS", fmt(bps, "원", 0), help=TERM_HELP["BPS"])
        st.metric("ROE", fmt(roe, "%"), help=TERM_HELP["ROE"])
    with c3:
        st.markdown("**배당 (최근 12개월)**")
        st.metric("배당수익률", fmt(div_yield, "%", 2), help=TERM_HELP["배당수익률"])
        st.metric("주당배당금", fmt(dps, "원", 0), help=TERM_HELP["주당배당금"])


# 예산으로 1주도 못 사는 종목은 건너뛰고, 같은 후보군에서 순위가 낮은 다음 종목으로
# 채워 넣어 (가능하다면) 상위 target_n개를 채운다. 점수 순으로 하나씩 넣어보면서,
# 이미 뽑힌 종목 중 누구라도 0주가 되게 만드는 후보는 넣지 않고 다음 후보로 넘어간다.
# 이렇게 하면 뽑힌 종목은 항상 전부 1주 이상 매수 가능한 상태로 유지된다.
def select_top_n_within_budget(candidates_df, budget, score_col, target_n=10):
    ranked = candidates_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    selected = ranked.iloc[0:0]

    for i in range(len(ranked)):
        if len(selected) >= target_n:
            break
        tentative = pd.concat([selected, ranked.iloc[[i]]], ignore_index=True)
        tentative = allocate_budget(tentative, budget, score_col)
        if (tentative["매수수량"] > 0).all():
            selected = tentative

    return selected.sort_values(score_col, ascending=False).reset_index(drop=True)


# ------------------------------------------------------------
# 페이지 1: 소개 페이지
# ------------------------------------------------------------
def render_intro_page():
    st.title("📊 투자성향 맞춤 종목 추천 서비스")
    with st.container(border=True):
        st.markdown(
            """
            간단한 설문에 답하면 AI가 나의 투자 성향(안정형/중립형/공격형)을 예측하고,
            관심 있는 섹터 안에서 성향에 맞는 추천 종목 상위 10개를 보여드립니다.

            - 5개 문항으로 구성된 짧은 설문
            - 관심 섹터를 선택하면 해당 섹터 안에서만 추천
            - 수익성 / 안정성 / 가치 / 배당 점수를 한눈에 비교
            """
        )
    render_disclaimer()
    st.write("")
    if st.button("📝 설문조사 하러가기", type="primary"):
        go_to("survey")
        st.rerun()


# ------------------------------------------------------------
# 페이지 2: 설문 + 관심 섹터 선택
# ------------------------------------------------------------
MIN_BUDGET = 10_000
MAX_BUDGET = 1_000_000_000


def _format_budget_input():
    digits = re.sub(r"[^\d]", "", st.session_state.get("budget_text", ""))
    st.session_state.budget_text = f"{int(digits):,}" if digits else ""


def render_survey_page():
    st.title("📝 투자 성향 설문")

    with st.container(border=True):
        st.subheader("💰 투자 예산")
        if "budget_text" not in st.session_state:
            st.session_state.budget_text = f"{1_000_000:,}"

        st.text_input(
            "총 투자 예산 (원)", key="budget_text", on_change=_format_budget_input,
        )
        budget_digits = re.sub(r"[^\d]", "", st.session_state.budget_text)
        budget = int(budget_digits) if budget_digits else 0
        budget = max(MIN_BUDGET, min(MAX_BUDGET, budget))
        st.caption(f"입력된 예산: {budget:,}원 (최소 {MIN_BUDGET:,}원 ~ 최대 {MAX_BUDGET:,}원)")

    st.write("")
    with st.container(border=True):
        st.subheader("📋 설문 문항")
        st.caption("각 문항에 0(매우 낮음) ~ 5(매우 높음) 중 하나를 선택해주세요.")

        scale = [0, 1, 2, 3, 4, 5]
        loss_tolerance = st.radio("손실 감수 수준", scale, index=3, horizontal=True)
        expected_return = st.radio("기대 수익률", scale, index=3, horizontal=True)
        investment_period = st.radio("투자 기간", scale, index=3, horizontal=True)
        dividend_preference = st.radio("배당 선호도", scale, index=3, horizontal=True)
        volatility_tolerance = st.radio("변동성 감수 수준", scale, index=3, horizontal=True)

    st.write("")
    with st.container(border=True):
        st.subheader("🏭 관심 섹터 선택")
        st.caption("섹터 버튼을 눌러 선택/해제하세요. 하나도 선택하지 않으면 전체 종목을 대상으로 추천합니다.")
        scored = load_scored_stocks()
        sector_options = sorted(scored["업종명"].dropna().unique().tolist())

        if "sector_selection" not in st.session_state:
            st.session_state.sector_selection = set()

        SECTORS_PER_ROW = 4
        for row_start in range(0, len(sector_options), SECTORS_PER_ROW):
            row_sectors = sector_options[row_start:row_start + SECTORS_PER_ROW]
            cols = st.columns(SECTORS_PER_ROW)
            for col, sector in zip(cols, row_sectors):
                is_selected = sector in st.session_state.sector_selection
                if col.button(
                    sector,
                    key=f"sector_btn_{sector}",
                    type="primary" if is_selected else "secondary",
                    use_container_width=True,
                ):
                    if is_selected:
                        st.session_state.sector_selection.discard(sector)
                    else:
                        st.session_state.sector_selection.add(sector)
                    st.rerun()

        selected_sectors = sorted(st.session_state.sector_selection)
        if selected_sectors:
            st.caption(f"선택된 섹터: {', '.join(selected_sectors)}")
        else:
            st.caption("선택된 섹터 없음 (전체 종목 대상)")

    st.write("")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← 처음으로"):
            go_to("intro")
            st.rerun()
    with col2:
        if st.button("🔍 추천받기", type="primary"):
            st.session_state.budget = budget
            st.session_state.survey_answers = {
                "손실감수수준": loss_tolerance,
                "기대수익률": expected_return,
                "투자기간": investment_period,
                "배당선호도": dividend_preference,
                "변동성감수수준": volatility_tolerance,
            }
            st.session_state.selected_sectors = selected_sectors
            go_to("result")
            st.rerun()


# ------------------------------------------------------------
# 페이지 3: 결과
# ------------------------------------------------------------
def render_result_page():
    st.title("🎯 추천 결과")

    answers = st.session_state.get("survey_answers")
    budget = st.session_state.get("budget")
    if not answers or not budget:
        st.error("설문 응답이 없습니다. 설문 페이지로 돌아가주세요.")
        if st.button("← 설문으로 돌아가기"):
            go_to("survey")
            st.rerun()
        return

    model = load_model()
    scored = load_scored_stocks()

    X = pd.DataFrame([answers])[FEATURE_COLS]
    predicted_profile = model.predict(X)[0]

    with st.container(border=True):
        st.subheader(f"🧭 예측된 투자 성향: **{predicted_profile}**")
        st.info(PROFILE_DESCRIPTIONS.get(predicted_profile, ""), icon="ℹ️")

    selected_sectors = st.session_state.get("selected_sectors") or []
    candidates = scored.copy()
    if selected_sectors:
        candidates = candidates[candidates["업종명"].isin(selected_sectors)]
        st.caption(f"관심 섹터({', '.join(selected_sectors)}) 안에서 추천합니다.")
    else:
        st.caption("전체 종목을 대상으로 추천합니다.")

    score_col = f"추천점수_{predicted_profile}"
    reason_col = f"추천이유_{predicted_profile}"

    if candidates.empty:
        st.warning("선택한 섹터에 해당하는 종목이 없습니다. 섹터 선택을 조정해주세요.")
    else:
        allocated = select_top_n_within_budget(candidates, budget, score_col, target_n=10)

        st.write("")
        with st.container(border=True):
            st.subheader("🏆 추천 종목 상위 10개")
            if allocated.empty:
                st.warning("예산이 부족해 1주도 매수할 수 없습니다. 예산을 늘려주세요.")
            else:
                if len(allocated) < 10:
                    st.caption(f"예산으로 매수 가능한 종목이 {len(allocated)}개뿐이라 그만큼만 추천합니다. 예산을 늘리면 더 많이 추천됩니다.")
                display_cols = {
                    "종목코드": "종목코드",
                    "종목명": "종목명",
                    "업종명": "업종명",
                    "현재가": "현재가(원)",
                    "매수수량": "매수수량(주)",
                    "투자금액": "투자금액(원)",
                    score_col: "추천점수",
                }
                display_df = allocated[list(display_cols.keys())].rename(columns=display_cols)
                for col in ["현재가(원)", "매수수량(주)", "투자금액(원)"]:
                    display_df[col] = display_df[col].map(lambda x: f"{x:,.0f}")
                st.dataframe(display_df, hide_index=True, use_container_width=True)

        if not allocated.empty:
            st.write("")
            with st.container(border=True):
                st.subheader("💰 예산 요약")
                total_invested = allocated["투자금액"].sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("총 예산", f"{budget:,.0f}원")
                c2.metric("총 투자금액", f"{total_invested:,.0f}원")
                c3.metric("미투자 잔액", f"{budget - total_invested:,.0f}원")

            st.write("")
            with st.container(border=True):
                st.subheader("📈 추천 점수 그래프")
                chart_df = allocated.set_index("종목명")[score_col].rename("추천점수")
                st.bar_chart(chart_df)

            st.write("")
            with st.container(border=True):
                st.subheader("🗒️ 종목별 상세 정보")
                for _, row in allocated.iterrows():
                    with st.expander(f"{row['종목명']} ({row['종목코드']}) - {row[reason_col]}"):
                        render_investment_metrics_card(row)

    st.divider()
    render_disclaimer()

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← 설문 다시하기"):
            go_to("survey")
            st.rerun()
    with col2:
        if st.button("처음으로"):
            go_to("intro")
            st.rerun()


# ------------------------------------------------------------
# 메인 라우팅
# ------------------------------------------------------------
def main():
    inject_custom_css()
    missing = files_missing()
    if missing:
        st.error(
            f"다음 파일이 없어 앱을 실행할 수 없습니다: {', '.join(missing)}. "
            "1단계(risk_model.pkl)와 3단계(scored_stocks.csv)를 먼저 실행해주세요."
        )
        return

    if st.session_state.page == "intro":
        render_intro_page()
    elif st.session_state.page == "survey":
        render_survey_page()
    elif st.session_state.page == "result":
        render_result_page()


main()
