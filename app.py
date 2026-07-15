# ============================================================
# 4단계: Streamlit 웹앱 통합 (Google Colab에서 %%writefile app.py로 저장)
# ============================================================
# 3페이지 구성: 소개 페이지 -> 설문(+관심 섹터) 페이지 -> 결과 페이지
# 입력 파일: risk_model.pkl (1단계), scored_stocks.csv (3단계)

import json
import math
import os
import re

import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel

# Colab Secrets에 GEMINI_API_KEY가 있으면 사용, 없으면 (로컬 실행 등) .env로 폴백
# (step2의 KRX_ID/KRX_PW와 동일한 패턴)
try:
    from google.colab import userdata

    try:
        os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
    except Exception:
        pass  # 해당 Secret이 없으면 무시하고 .env 로 폴백
except ImportError:
    pass

load_dotenv()  # override=False 이므로 이미 설정된 값은 덮어쓰지 않음

st.set_page_config(page_title="투자성향 맞춤 종목 추천", page_icon="📊", layout="wide")

GEMINI_MODEL = "gemini-3.1-flash-lite"  # 짧은 요약용 - 속도 빠르고 비용 저렴 (gemini-2.5-flash-lite는 조기 404로 중단됨)


def inject_custom_css():
    st.markdown(
        """
        <style>
        :root {
            --accent: #2F88FF;
            --accent-dark: #1666D9;
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
            color: #ffffff;
            font-weight: 700;
        }
        .stButton > button[kind="primary"]:hover {
            filter: brightness(1.08);
            color: #ffffff;
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

        /* 네비게이션 바: 로고와 오른쪽 메뉴 버튼을 같은 높이로 정렬 */
        div[data-testid="stHorizontalBlock"]:has(button[kind="primary"], button[kind="secondary"]) {
            align-items: center;
        }

        /* 로고 이미지가 흐릿하면 채도/명암을 살짝 올려 버튼 색과 더 잘 어울리게 함 */
        div[data-testid="stImage"] img {
            filter: saturate(1.6) contrast(1.15);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


SITE_NAME = "FOMO"  # 로고 이미지(logo.png)가 없을 때 대신 보여줄 임시 텍스트 마크

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(APP_DIR, "risk_model.pkl")
DATA_FILE = os.path.join(APP_DIR, "scored_stocks.csv")
LOGO_FILE = os.path.join(APP_DIR, "FOMO_Logo.png")  # app.py와 같은 폴더에 두면 자동으로 로고가 표시됨
FEATURE_COLS = ["손실감수수준", "기대수익률", "투자기간", "배당선호도", "변동성감수수준"]

PROFILE_DESCRIPTIONS = {
    "안정형": "원금 손실을 최소화하고 배당·안정성을 중시하는 투자자입니다. 변동성이 낮고 꾸준한 배당을 주는 종목을 선호합니다.",
    "중립형": "수익과 안정성의 균형을 추구하는 투자자입니다. 성장성과 안정성을 함께 고려한 종목을 선호합니다.",
    "공격형": "높은 수익을 위해 변동성을 감수할 수 있는 투자자입니다. 성장성이 높은 종목을 선호합니다.",
}

# 페이지 3(내 투자성향 확인하기)에서 보여줄 성향별 추천 투자전략
STRATEGY_TIPS = {
    "안정형": [
        "배당수익률이 높고 변동성이 낮은 대형주 위주로 담아보세요.",
        "여러 섹터에 나눠 담아 한 업종에 쏠리지 않도록 하는 게 좋아요.",
    ],
    "중립형": [
        "성장성과 안정성을 함께 갖춘 종목을 중심으로 포트폴리오를 구성해보세요.",
        "일부는 안정적인 배당주, 일부는 성장주로 나눠 담는 것도 좋은 방법이에요.",
    ],
    "공격형": [
        "성장성이 높은 종목 위주로 담되, 변동성이 큰 만큼 분산투자를 추천 드려요.",
        "단기 등락에 일희일비하지 않고 장기적인 관점으로 접근하는 게 중요해요.",
    ],
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


SURVEY_STATE_KEYS = (
    "user_name",
    "budget_text",
    "sector_selection",
    "survey_step",
    "survey_wip",
    "loss_tolerance",
    "expected_return",
    "investment_period",
    "dividend_preference",
    "volatility_tolerance",
)


# 결과 페이지에서 설문/처음으로 돌아갈 때 이전 예산/섹터/설문 응답을 초기화한다.
# (키를 지우면 다음 렌더링에서 위젯의 index=3, 빈 예산 등 기본값으로 새로 생성된다.)
def reset_survey_state():
    for key in SURVEY_STATE_KEYS:
        st.session_state.pop(key, None)


# ------------------------------------------------------------
# 상단 네비게이션 바: 왼쪽 사이트 마크 + 오른쪽 4개 메뉴
# ------------------------------------------------------------
NAV_ITEMS = [
    ("HOME", "intro"),
    ("설문조사 하러가기", "survey"),
    ("내 투자성향 확인하기", "profile"),
    ("종목 추천받기", "recommend"),
]

# 아직 설문(이름/예산/문항)을 안 끝낸 상태로 이동하면 설문부터 하도록 안내한다.
PAGES_REQUIRE_SURVEY = {"profile", "recommend"}


def render_navbar():
    current_page = st.session_state.page
    logo_col, *nav_cols = st.columns([2, 1, 1, 1, 1])
    with logo_col:
        if os.path.exists(LOGO_FILE):
            st.image(LOGO_FILE, width=220)
        else:
            st.markdown(f"### {SITE_NAME}")
    for col, (label, target_page) in zip(nav_cols, NAV_ITEMS):
        with col:
            is_active = current_page == target_page
            if st.button(
                label,
                key=f"nav_{target_page}_{label}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                if target_page in PAGES_REQUIRE_SURVEY and not st.session_state.get("survey_answers"):
                    st.session_state.nav_notice = "먼저 설문을 완료해주세요."
                    go_to("survey")
                else:
                    go_to(target_page)
                st.rerun()
    st.divider()

    notice = st.session_state.pop("nav_notice", None)
    if notice:
        st.info(notice)


# ------------------------------------------------------------
# 공통: 투자 참고용 경고 문구
# ------------------------------------------------------------
def render_disclaimer():
    st.warning(
        "⚠️ 이 서비스는 투자 자문이 아닌 참고용 데모입니다. 실제 투자 판단과 책임은 본인에게 있습니다.",
        icon="⚠️",
    )


# ------------------------------------------------------------
# AI 요약 (Gemini): 전체 요약 2~3문장 + 종목별 한줄평을 한 번의 API 호출로 생성
# GEMINI_API_KEY가 없으면 조용히 비활성화되고, 있으면 결과 페이지에 카드로 표시된다.
# (종목 수만큼 API를 나눠 부르면 비용/시간이 N배가 되므로, 구조화된 출력(JSON) 하나로
#  전체 요약과 종목별 한줄평을 동시에 받는다.)
# ------------------------------------------------------------
class _StockInsight(BaseModel):
    종목코드: str
    한줄평: str


class _AIInsights(BaseModel):
    전체요약: str
    종목별한줄평: list[_StockInsight]


@st.cache_resource(show_spinner=False)
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
    except ImportError:
        return None
    return genai.Client(api_key=api_key)


# 동일한 추천 조합(prompt)에 대해서는 재실행 시 API를 다시 호출하지 않도록 캐시
@st.cache_data(show_spinner="🤖 AI 요약 생성 중...")
def _call_gemini(prompt):
    client = get_gemini_client()
    if client is None:
        return None
    try:
        from google.genai import types

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_AIInsights,
            ),
        )
        parsed = _AIInsights.model_validate(json.loads(response.text))
        return {
            "전체요약": parsed.전체요약,
            "종목별": {item.종목코드: item.한줄평 for item in parsed.종목별한줄평},
        }
    except Exception as exc:
        return f"__ERROR__:{exc}"


def generate_ai_insights(predicted_profile, allocated, score_col, reason_col):
    if get_gemini_client() is None or allocated.empty:
        return None

    stock_lines = "\n".join(
        f"- 종목코드 {r['종목코드']} / {r['종목명']} ({r['업종명']}): PER {r.get('PER', '-')}, "
        f"PBR {r.get('PBR', '-')}, 배당수익률 {r.get('배당수익률', '-')}%, "
        f"추천점수 {r[score_col]:.1f}, 참고 설명: {r[reason_col]}"
        for _, r in allocated.iterrows()
    )
    prompt = (
        "당신은 주식 초보자에게 친절하게 설명하는 투자 어시스턴트입니다.\n"
        f"아래는 '{predicted_profile}' 투자 성향으로 예측된 사용자에게 추천된 종목들입니다. "
        "'참고 설명'은 이미 계산된 사실 기반 설명이니, 새로운 수치를 지어내지 말고 "
        "이 내용에 근거해서만 자연스럽게 다듬어 주세요.\n\n"
        f"{stock_lines}\n\n"
        "다음 두 가지를 JSON으로 응답하세요:\n"
        f"1. 전체요약: 이 추천 결과 전체를 2~3문장으로 설명 (어떤 업종/특징이 많은지, "
        f"'{predicted_profile}' 투자자에게 왜 적합한지)\n"
        "2. 종목별한줄평: 각 종목코드별로 한 문장짜리 한줄평 (모든 종목 빠짐없이 포함)\n\n"
        "과장된 확신이나 매수/매도 지시는 하지 말고, 참고용 설명으로만 작성하세요."
    )

    result = _call_gemini(prompt)
    if isinstance(result, str) and result.startswith("__ERROR__:"):
        st.caption(f"AI 요약을 불러오지 못했습니다: {result[len('__ERROR__:'):]}")
        return None
    return result


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

    st.caption("투자 지표 (KRX 공식 통계 기준 · 별도 재무제표) — 증권사 앱은 연결 재무제표 기준을 써서 수치가 다를 수 있어요")
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
            **FOMO**는 주식 투자에 대한 기초 지식이 부족하고 투자 가능한 금액이 많지 않은
            사회초년생과 초보 투자자를 위한 사용자 맞춤형 투자 추천 서비스에요.

            사용자는 간단한 설문을 통해 월 투자 가능 금액, 투자 기간, 손실 감수 수준, 관심 산업 등의
            정보를 입력하면 투자 성향 테스트 결과와 종목을 추천해드려요.
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

# 자연스러운 문장형 질문 + 자연어 선택지 (내부적으로는 1단계 모델이 학습한 1~5 스케일로 매핑)
SURVEY_QUESTIONS = [
    {
        "key": "loss_tolerance",
        "feature": "손실감수수준",
        "question": "투자한 돈에서 20%가 갑자기 사라진다면, 나는...",
        "options": [
            ("바로 다 판다", 1),
            ("불안해서 일부는 정리한다", 2),
            ("불안하지만 지켜본다", 3),
            ("그대로 둔다", 4),
            ("오히려 더 사고 싶다", 5),
        ],
    },
    {
        "key": "expected_return",
        "feature": "기대수익률",
        "question": "내가 바라는 연평균 수익률은?",
        "options": [
            ("은행 이자보다 조금 높으면 충분하다", 1),
            ("물가상승률 정도만 되면 좋겠다", 2),
            ("코스피 지수 평균 정도면 만족한다", 3),
            ("두 자릿수 수익을 원한다", 4),
            ("가능하면 훨씬 큰 수익을 원한다", 5),
        ],
    },
    {
        "key": "investment_period",
        "feature": "투자기간",
        "question": "이 돈, 언제까지 안 건드려도 될까요?",
        "options": [
            ("1년 이내", 1),
            ("1~2년", 2),
            ("2~3년", 3),
            ("3~5년", 4),
            ("5년 이상", 5),
        ],
    },
    {
        "key": "dividend_preference",
        "feature": "배당선호도",
        "question": "매년 돈이 정기적으로 들어오는 게 중요한가요?",
        "options": [
            ("전혀 안 중요하다", 1),
            ("있으면 좋다", 2),
            ("어느 정도는 신경 쓰인다", 3),
            ("꾸준히 받고 싶다", 4),
            ("배당이 핵심이다", 5),
        ],
    },
    {
        "key": "volatility_tolerance",
        "feature": "변동성감수수준",
        "question": "주가가 오르락내리락해도 괜찮나요?",
        "options": [
            ("스트레스 받는다", 1),
            ("조금 불안하다", 2),
            ("그런가보다 한다", 3),
            ("별로 신경 쓰지 않는다", 4),
            ("오히려 흥미롭다", 5),
        ],
    },
]


def _format_budget_input():
    digits = re.sub(r"[^\d]", "", st.session_state.get("budget_text", ""))
    st.session_state.budget_text = f"{int(digits):,}" if digits else ""


def render_survey_page():
    st.title("📝 투자 성향 설문")

    with st.container(border=True):
        st.subheader("🙋 이름")
        user_name = st.text_input("이름 (또는 별명)", key="user_name", placeholder="예: 홍길동")

    st.write("")
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

    total_q = len(SURVEY_QUESTIONS)
    if "survey_step" not in st.session_state:
        st.session_state.survey_step = 0
    step = min(st.session_state.survey_step, total_q - 1)

    # 페이지를 하나씩 넘기다 보면 현재 문항이 아닌 라디오는 렌더링되지 않는 구간이 생기는데,
    # 그 사이에 Streamlit이 위젯 key의 세션 상태를 되살리지 못해 답이 사라지는 문제가 있었다.
    # 그래서 답변을 위젯 key가 아니라 별도의 survey_wip 딕셔너리에 직접 보관하고,
    # 라디오의 index도 그 딕셔너리 값을 기준으로 매번 명시적으로 계산해서 넣어준다.
    if "survey_wip" not in st.session_state:
        st.session_state.survey_wip = {sq["feature"]: None for sq in SURVEY_QUESTIONS}

    st.write("")
    with st.container(border=True):
        st.subheader("📋 몇 가지만 편하게 답해주세요")
        st.progress((step + 1) / total_q)
        st.caption(f"질문 {step + 1} / {total_q}")

        q = SURVEY_QUESTIONS[step]
        labels = [label for label, _ in q["options"]]
        value_map = dict(q["options"])
        label_by_value = {v: k for k, v in q["options"]}

        current_value = st.session_state.survey_wip.get(q["feature"])
        current_label = label_by_value.get(current_value)
        default_index = labels.index(current_label) if current_label in labels else None

        selected_label = st.radio(q["question"], labels, index=default_index, key=q["key"])
        st.session_state.survey_wip[q["feature"]] = value_map.get(selected_label)

        nav1, nav2 = st.columns([1, 1])
        with nav1:
            if step > 0 and st.button("← 이전 질문"):
                st.session_state.survey_step = step - 1
                st.rerun()
        with nav2:
            if step < total_q - 1:
                if st.button("다음 질문 →", type="primary"):
                    if selected_label is None:
                        st.error("답을 선택해주세요.")
                    else:
                        st.session_state.survey_step = step + 1
                        st.rerun()

    survey_values = dict(st.session_state.survey_wip)
    all_answered = all(v is not None for v in survey_values.values())

    if not all_answered:
        st.write("")
        st.info("설문을 먼저 완료해주세요.")
        st.write("")
        if st.button("← 처음으로"):
            go_to("intro")
            st.rerun()
        return

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
        if st.button("🧭 내 투자성향 확인하러가기", type="primary"):
            if not user_name.strip():
                st.error("이름을 입력해주세요.")
            else:
                # user_name은 이미 key="user_name" 위젯이 세션 상태에 저장해뒀으므로
                # 여기서 다시 대입하면 안 된다 (위젯 인스턴스화 후 재대입 시 에러 발생).
                st.session_state.budget = budget
                st.session_state.survey_answers = survey_values
                go_to("profile")
                st.rerun()


# ------------------------------------------------------------
# 페이지 3: 내 투자성향 확인하기
# ------------------------------------------------------------
def predict_profile():
    """survey_answers를 기반으로 투자 성향을 예측한다 (profile/recommend 페이지 공용)."""
    answers = st.session_state.get("survey_answers")
    if not answers:
        return None
    model = load_model()
    X = pd.DataFrame([answers])[FEATURE_COLS]
    return model.predict(X)[0]


def render_profile_page():
    st.title("🧭 내 투자성향")

    user_name = (st.session_state.get("user_name") or "").strip()
    predicted_profile = predict_profile()
    if not user_name or predicted_profile is None:
        st.error("설문 응답이 없습니다. 설문 페이지로 돌아가주세요.")
        if st.button("← 설문으로 돌아가기"):
            go_to("survey")
            st.rerun()
        return

    with st.container(border=True):
        st.subheader(f"{user_name}님은 **{predicted_profile}** 성향이에요")
        st.info(PROFILE_DESCRIPTIONS.get(predicted_profile, ""), icon="ℹ️")

        st.markdown("**추천 투자전략**")
        for tip in STRATEGY_TIPS.get(predicted_profile, []):
            st.markdown(f"- {tip}")

    st.write("")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← 설문 다시하기"):
            reset_survey_state()
            go_to("survey")
            st.rerun()
    with col2:
        if st.button("📈 종목 추천받으러가기", type="primary"):
            go_to("recommend")
            st.rerun()


# ------------------------------------------------------------
# 페이지 4: 종목 추천받기
# ------------------------------------------------------------
def render_recommend_page():
    st.title("🎯 종목 추천")

    budget = st.session_state.get("budget")
    predicted_profile = predict_profile()
    if not budget or predicted_profile is None:
        st.error("설문 응답이 없습니다. 설문 페이지로 돌아가주세요.")
        if st.button("← 설문으로 돌아가기"):
            go_to("survey")
            st.rerun()
        return

    scored = load_scored_stocks()

    # 관심 섹터는 2페이지(설문) 마지막 단계에서 이미 선택했으므로 여기서는 그 값을 그대로 사용한다.
    selected_sectors = sorted(st.session_state.get("sector_selection", set()))
    if selected_sectors:
        st.caption(f"관심 섹터({', '.join(selected_sectors)}) 안에서 추천합니다.")
    else:
        st.caption("전체 종목을 대상으로 추천합니다.")

    candidates = scored.copy()
    if selected_sectors:
        candidates = candidates[candidates["업종명"].isin(selected_sectors)]

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
            insights = generate_ai_insights(predicted_profile, allocated, score_col, reason_col)
            if insights:
                st.write("")
                with st.container(border=True):
                    st.subheader("🤖 AI 요약")
                    st.write(insights["전체요약"])
            elif get_gemini_client() is None:
                st.caption(
                    "💡 Colab Secrets에 GEMINI_API_KEY를 등록하면 추천 결과를 AI가 자연어로 요약해줍니다."
                )

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
                ai_per_stock = insights["종목별"] if insights else {}
                for _, row in allocated.iterrows():
                    # AI가 해당 종목코드에 대한 한줄평을 못 준 경우 규칙 기반 설명으로 대체
                    reason_text = ai_per_stock.get(row["종목코드"], row[reason_col])
                    with st.expander(f"{row['종목명']} ({row['종목코드']}) - {reason_text}"):
                        render_investment_metrics_card(row)

    st.divider()
    render_disclaimer()

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← 설문 다시하기"):
            reset_survey_state()
            go_to("survey")
            st.rerun()
    with col2:
        if st.button("처음으로"):
            reset_survey_state()
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

    render_navbar()

    if st.session_state.page == "intro":
        render_intro_page()
    elif st.session_state.page == "survey":
        render_survey_page()
    elif st.session_state.page == "profile":
        render_profile_page()
    elif st.session_state.page == "recommend":
        render_recommend_page()


main()
