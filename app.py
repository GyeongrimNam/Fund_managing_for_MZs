# ============================================================
# 4단계: Streamlit 웹앱 통합 (Google Colab에서 %%writefile app.py로 저장)
# ============================================================
# 3페이지 구성: 소개 페이지 -> 설문(+관심 섹터) 페이지 -> 결과 페이지
# 입력 파일: risk_model.pkl (1단계), scored_stocks.csv (3단계)

import base64
import json
import math
import os
import re
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel

from ui import load_css

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


SITE_NAME = "FOMO"  # 로고 이미지(logo.png)가 없을 때 대신 보여줄 임시 텍스트 마크

APP_DIR = Path(__file__).resolve().parent
MODEL_FILE = APP_DIR / "risk_model.pkl"
DATA_FILE = APP_DIR / "scored_stocks.csv"
LOGO_FILE = APP_DIR / "FOMO_Logo.svg"
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

# 페이지 4(종목 추천받기)에서 성향별로 함께 보여줄 ETF 추천 목록.
# 개별 종목과 달리 ETF는 PER/PBR 같은 4축 점수가 잘 맞지 않아서(가치점수 계산이 어색해짐),
# 점수로 경쟁시켜 순위를 매기지 않고 카테고리(성향) 매칭 + 거래량 순으로 큐레이션한
# 고정 목록을 그대로 보여준다. 실제 거래량은 pykrx로 조회해 확인한 값 기준(2025-07 상순).
ETF_RECOMMENDATIONS = {
    "안정형": [
        {"종목명": "SOL 미국배당다우존스", "종목코드": "446720", "설명": "다우존스 US Dividend 100 지수를 추종하는, 미국 배당성장주 ETF에요.", "구성종목": "코카콜라, 셰브런, IBM 등 미국의 대표적인 배당주들"},
        {"종목명": "TIGER 배당성장", "종목코드": "211560", "설명": "국내 배당성장주 지수를 추종하는 ETF에요.", "구성종목": "KT&G, 삼성전자, 현대차 등 꾸준히 배당을 늘려온 국내 대표 기업들"},
        {"종목명": "KODEX 종합채권(AA-이상)액티브", "종목코드": "273130", "설명": "국채와 AA등급 이상 우량 회사채를 섞어 담는 채권형 ETF에요.", "구성종목": "개별 주식이 아니라 국고채·우량회사채 위주"},
        {"종목명": "KODEX 국고채3년", "종목코드": "114260", "설명": "만기 3년 안팎의 국고채에 투자하는 ETF예요, 채권형 중에서도 변동성이 낮은 편이에요.", "구성종목": "국가가 발행하는 국고채 위주"},
        {"종목명": "KODEX 배당가치", "종목코드": "325020", "설명": "배당과 가치주 성격을 함께 담은 지수를 추종하는 ETF에요.", "구성종목": "배당수익률이 높으면서 저평가된 국내 대형주들"},
    ],
    "중립형": [
        {"종목명": "KODEX 200", "종목코드": "069500", "설명": "코스피200 지수를 그대로 추종하는, 국내에서 제일 규모가 큰 ETF에요.", "구성종목": "삼성전자, SK하이닉스 등 코스피 시가총액 상위 200개 대형주"},
        {"종목명": "KODEX 200미국채혼합50", "종목코드": "284430", "설명": "코스피200에 50%, 미국 국채에 50%를 나눠 담아서 리스크를 낮춘 혼합형 ETF에요.", "구성종목": "코스피200 대형주 절반 + 미국 국채 절반"},
    ],
}

# 공격형 전용: 사용자가 2페이지에서 고른 관심 섹터(업종명)에 매칭되는 테마 ETF.
# 매칭되는 섹터가 없거나 섹터를 아예 안 골랐으면 ETF_AGGRESSIVE_FALLBACK을 대신 보여준다.
ETF_SECTOR_MAP = {
    "증권": {"종목명": "KODEX 증권", "종목코드": "102970", "설명": "국내 증권업종 지수를 추종하는 ETF에요.", "구성종목": "미래에셋증권, 삼성증권, NH투자증권, 키움증권 등 국내 주요 증권사들"},
    "은행": {"종목명": "KODEX 은행", "종목코드": "091170", "설명": "국내 은행업종 지수를 추종하는 ETF에요.", "구성종목": "KB금융, 신한지주, 하나금융지주, 우리금융지주 등 국내 주요 금융지주사들"},
    "IT 서비스": {"종목명": "TIGER 인터넷TOP10", "종목코드": "365000", "설명": "국내 인터넷 대표기업 10종목을 추종하는 ETF에요.", "구성종목": "네이버, 카카오 등 국내 대표 인터넷 기업들"},
    "전기·전자": {"종목명": "KODEX 반도체", "종목코드": "091160", "설명": "국내 반도체 업종 지수를 추종하는 ETF에요.", "구성종목": "삼성전자, SK하이닉스 등 국내 대표 반도체 기업들"},
    "화학": {"종목명": "TIGER 2차전지테마", "종목코드": "305540", "설명": "2차전지 관련 기업들을 모아 담은 테마형 ETF에요.", "구성종목": "LG에너지솔루션, 삼성SDI, 에코프로 등 2차전지 밸류체인 기업들"},
    "운송장비·부품": {"종목명": "KODEX 자동차", "종목코드": "091180", "설명": "국내 자동차 업종 지수를 추종하는 ETF에요.", "구성종목": "현대차, 기아, 현대모비스 등 국내 대표 자동차 기업들"},
    "제약": {"종목명": "TIGER 헬스케어", "종목코드": "143860", "설명": "국내 헬스케어 업종 지수를 추종하는 ETF에요.", "구성종목": "삼성바이오로직스, 셀트리온 등 국내 대표 제약·바이오 기업들"},
    "금속": {"종목명": "KODEX 철강", "종목코드": "117680", "설명": "국내 철강 업종 지수를 추종하는 ETF에요.", "구성종목": "POSCO홀딩스, 현대제철 등 국내 대표 철강 기업들"},
    "음식료·담배": {"종목명": "KODEX 필수소비재", "종목코드": "266410", "설명": "국내 필수소비재 업종 지수를 추종하는 ETF에요.", "구성종목": "KT&G, 오리온, CJ제일제당 등 국내 대표 음식료·생활필수품 기업들"},
    "유통": {"종목명": "KODEX 경기소비재", "종목코드": "266390", "설명": "국내 경기소비재 업종 지수를 추종하는 ETF에요.", "구성종목": "현대차, 기아 등 경기 흐름에 민감한 소비재 관련 기업들"},
}

ETF_AGGRESSIVE_FALLBACK = [
    {"종목명": "KODEX 미국S&P500", "종목코드": "379800", "설명": "미국 S&P500 지수를 추종하는 ETF에요.", "구성종목": "애플, 마이크로소프트, 엔비디아 등 미국 대표 대형주들"},
    {"종목명": "TIGER 미국나스닥100", "종목코드": "133690", "설명": "미국 나스닥100 지수를 추종하는 ETF에요.", "구성종목": "애플, 마이크로소프트, 엔비디아 등 미국 대표 기술주들"},
]


def get_etf_recommendations(predicted_profile, selected_sectors):
    """성향(및 공격형의 경우 관심 섹터)에 맞는 ETF 추천 목록을 반환한다."""
    if predicted_profile == "공격형":
        matched = [ETF_SECTOR_MAP[s] for s in selected_sectors if s in ETF_SECTOR_MAP]
        return matched if matched else ETF_AGGRESSIVE_FALLBACK
    return ETF_RECOMMENDATIONS.get(predicted_profile, [])


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
    if page_name in {page for _, page in NAV_ITEMS}:
        st.query_params["page"] = page_name


def sync_page_from_query_params():
    requested_page = st.query_params.get("page")
    valid_pages = {page for _, page in NAV_ITEMS}
    if requested_page not in valid_pages:
        return

    if requested_page in PAGES_REQUIRE_SURVEY and not st.session_state.get("survey_answers"):
        st.session_state.nav_notice = "먼저 설문을 완료해주세요."
        st.session_state.page = "survey"
        st.query_params["page"] = "survey"
        return

    st.session_state.page = requested_page


SURVEY_STATE_KEYS = (
    "user_name",
    "user_name_widget",
    "budget_text",
    "budget_text_widget",
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


# 로고는 왼쪽에, 메뉴 4개는 spacer 뒤 오른쪽 영역에 모아 배치한다.
# (로고, spacer, HOME, 설문조사 하러가기, 내 투자성향 확인하기, 종목 추천받기)
NAV_COLUMN_RATIOS = [2.4, 4.2, 0.9, 1.45, 1.75, 1.35]


def render_navbar():
    current_page = st.session_state.page
    logo_html = get_logo_html()

    with st.container(key="main_navigation"):
        logo_col, spacer_col, *nav_cols = st.columns(NAV_COLUMN_RATIOS)
        with logo_col:
            st.markdown(
                f'<a class="fomo-navbar-logo" href="?page=intro" aria-label="{SITE_NAME} 홈">{logo_html}</a>',
                unsafe_allow_html=True,
            )
        for col, (label, target_page) in zip(nav_cols, NAV_ITEMS):
            with col:
                is_active = current_page == target_page
                if st.button(
                    label,
                    key=f"nav_{target_page}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True,
                ):
                    go_to(target_page)
                    st.rerun()

    notice = st.session_state.pop("nav_notice", None)
    if notice:
        st.info(notice)


def get_logo_html():
    if LOGO_FILE.exists():
        svg_content = LOGO_FILE.read_text(encoding="utf-8")
        encoded_logo = base64.b64encode(svg_content.encode("utf-8")).decode("utf-8")
        return f'<img src="data:image/svg+xml;base64,{encoded_logo}" alt="{SITE_NAME}" />'
    return f'<span class="fomo-navbar-text-logo">{SITE_NAME}</span>'


# ------------------------------------------------------------
# 공통: 투자 참고용 경고 문구
# ------------------------------------------------------------
def render_disclaimer():
    st.markdown(
        """
        <div style="margin-top:48px; padding-top:16px; border-top:1px solid rgba(128,128,128,0.25);
                    text-align:center; font-size:0.8rem; color:var(--text-secondary, #888); line-height:1.6;">
            본 서비스는 「자본시장과 금융투자업에 관한 법률」상 투자자문업으로 등록되지 않은
            참고용 서비스이며, 제공되는 정보는 투자 판단을 돕기 위한 참고 자료일 뿐
            특정 종목에 대한 투자 권유가 아닙니다.<br>
            투자에 따른 손익 등 모든 책임은 이용자 본인에게 귀속됩니다.
        </div>
        """,
        unsafe_allow_html=True,
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
    st.markdown(
        """
        <div class="fomo-hero fomo-animate fomo-delay-0">
            <span class="fomo-hero-badge">초보 투자자를 위한 맞춤 서비스 🎯</span>
            <div class="fomo-hero-title">투자, 이제<br><span class="fomo-hero-title-accent">나에게 맞게</span> 시작해요</div>
            <p class="fomo-hero-desc">어떤 주식을 사야 할지 막막하셨나요?<br>FOMO가 내 상황에 딱 맞는 종목을 쉽게 찾아드려요.</p>
            <div class="fomo-hero-stats">
                <div class="fomo-hero-stat">
                    <div class="fomo-hero-stat-value">5분</div>
                    <div class="fomo-hero-stat-label">설문 소요 시간</div>
                </div>
                <div class="fomo-hero-stat">
                    <div class="fomo-hero-stat-value">3가지</div>
                    <div class="fomo-hero-stat-label">투자 성향 유형</div>
                </div>
                <div class="fomo-hero-stat">
                    <div class="fomo-hero-stat-value">무료</div>
                    <div class="fomo-hero-stat-label">완전 무료 서비스</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="fomo-section fomo-animate fomo-delay-1">
            <p class="fomo-section-label">이런 분들을 위해 만들었어요</p>
            <div class="fomo-section-title">혹시 이런 고민 해본 적 있으세요?</div>
            <div class="fomo-problem-grid">
                <div class="fomo-problem-card">
                    <div class="fomo-problem-icon">🌱</div>
                    <div class="fomo-problem-title">투자 처음이에요</div>
                    <div class="fomo-problem-desc">주식이 뭔지는 알지만 막상 시작하려니 막막한 분</div>
                </div>
                <div class="fomo-problem-card">
                    <div class="fomo-problem-icon">💵</div>
                    <div class="fomo-problem-title">소액으로 시작할게요</div>
                    <div class="fomo-problem-desc">적은 금액이라도 꾸준히 모아보고 싶은 사회초년생</div>
                </div>
                <div class="fomo-problem-card">
                    <div class="fomo-problem-icon">🤔</div>
                    <div class="fomo-problem-title">뭘 사야 할지 모르겠어요</div>
                    <div class="fomo-problem-desc">종목은 많은데 어디서부터 시작해야 할지 모르는 분</div>
                </div>
                <div class="fomo-problem-card">
                    <div class="fomo-problem-icon">📖</div>
                    <div class="fomo-problem-title">용어가 너무 어려워요</div>
                    <div class="fomo-problem-desc">PER, ETF, 분산투자... 쉽게 설명해줬으면 하는 분</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="fomo-section fomo-section-alt fomo-animate fomo-delay-2">
            <p class="fomo-section-label">이렇게 이용하면 돼요</p>
            <div class="fomo-section-title">딱 3단계면 끝이에요</div>
            <div class="fomo-steps-grid">
                <div class="fomo-step-card">
                    <div class="fomo-step-number">01</div>
                    <div class="fomo-step-title">간단한 설문 답하기</div>
                    <div class="fomo-step-desc">투자 예산, 손실 감수 수준, 투자 기간 등을 물어봐요. 5분이면 충분해요.</div>
                </div>
                <div class="fomo-step-card">
                    <div class="fomo-step-number">02</div>
                    <div class="fomo-step-title">투자 성향 파악하기</div>
                    <div class="fomo-step-desc">안정형, 중립형, 공격형 중 나에게 맞는 유형을 알려드려요.</div>
                </div>
                <div class="fomo-step-card">
                    <div class="fomo-step-number">03</div>
                    <div class="fomo-step-title">맞춤 종목 추천받기</div>
                    <div class="fomo-step-desc">내 성향에 딱 맞는 주식과 ETF를 이유와 함께 보여드려요.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="fomo-section fomo-animate fomo-delay-3">
            <p class="fomo-section-label">추천 결과에서 이런 걸 볼 수 있어요</p>
            <div class="fomo-section-title">단순 추천이 아니에요.<br><span class="fomo-hero-title-accent">이유까지</span> 알려드려요.</div>
            <div class="fomo-feature-grid">
                <div class="fomo-feature-card">
                    <div class="fomo-feature-icon">✅</div>
                    <div class="fomo-feature-title">추천 이유를 알 수 있어요</div>
                    <div class="fomo-feature-desc">"왜 이 종목을 추천하는지" 쉬운 말로 설명해드려요.</div>
                </div>
                <div class="fomo-feature-card">
                    <div class="fomo-feature-icon">⚠️</div>
                    <div class="fomo-feature-title">위험도를 미리 알 수 있어요</div>
                    <div class="fomo-feature-desc">변동성, 최대낙폭 같은 위험 지표를 솔직하게 보여드려요.</div>
                </div>
                <div class="fomo-feature-card">
                    <div class="fomo-feature-icon">📊</div>
                    <div class="fomo-feature-title">투자 지표도 함께 봐요</div>
                    <div class="fomo-feature-desc">PER, PBR 같은 어려운 용어도 쉬운 설명과 함께 보여드려요.</div>
                </div>
                <div class="fomo-feature-card">
                    <div class="fomo-feature-icon">📦</div>
                    <div class="fomo-feature-title">분산투자용 ETF도 추천해요</div>
                    <div class="fomo-feature-desc">개별 종목과 함께 참고할 만한 ETF도 알려드려요.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <section class="fomo-survey-cta" aria-label="설문조사 이동">
            <div class="fomo-survey-cta-icon">🚀</div>
            <h2 class="fomo-survey-cta-title">나의 투자 성향,<br>지금 바로 알아볼까요?</h2>
            <p class="fomo-survey-cta-desc">5분 설문으로 나에게 맞는 종목을 추천받아 보세요.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    _, cta_col, _ = st.columns([1, 1, 1])
    with cta_col:
        if st.button("설문조사 하러가기", type="primary", use_container_width=True):
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
    digits = re.sub(r"[^\d]", "", st.session_state.get("budget_text_widget", ""))
    formatted = f"{int(digits):,}" if digits else ""
    st.session_state.budget_text_widget = formatted
    st.session_state.budget_text = formatted


# 2페이지 전체를 "페이지 안의 페이지"처럼 한 단계씩 진행한다: 이름 -> 예산 -> 문항 5개 -> 관심 섹터
NAME_STEP = 0
BUDGET_STEP = 1
QUESTION_STEPS = list(range(2, 2 + len(SURVEY_QUESTIONS)))
SECTOR_STEP = 2 + len(SURVEY_QUESTIONS)
TOTAL_SURVEY_STEPS = SECTOR_STEP + 1


@st.fragment
def render_sector_selector(sector_options):
    """실제 업종 목록의 선택 UI만 fragment 범위에서 다시 실행한다."""
    sectors_per_row = 4
    for row_start in range(0, len(sector_options), sectors_per_row):
        row_sectors = sector_options[row_start:row_start + sectors_per_row]
        cols = st.columns(sectors_per_row)
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
                st.rerun(scope="fragment")

    selected_sectors = sorted(st.session_state.sector_selection)
    if selected_sectors:
        st.caption(f"선택된 섹터: {', '.join(selected_sectors)}")
    else:
        st.caption("선택 없이 넘어가면 전체 종목 대상으로 추천합니다")


def render_survey_page():
    if "survey_step" not in st.session_state:
        st.session_state.survey_step = 0
    step = min(st.session_state.survey_step, TOTAL_SURVEY_STEPS - 1)

    # 페이지를 하나씩 넘기다 보면 현재 스텝이 아닌 위젯은 렌더링되지 않는 구간이 생기는데,
    # 그 사이에 Streamlit이 위젯 key의 세션 상태를 되살리지 못해 값이 사라지는 문제가 있었다.
    # 그래서 답변은 위젯 key가 아니라 별도의 딕셔너리(survey_wip)에 직접 보관하고,
    # 위젯의 index/value도 그 딕셔너리 값을 기준으로 매번 명시적으로 계산해서 넣어준다.
    if "survey_wip" not in st.session_state:
        st.session_state.survey_wip = {sq["feature"]: None for sq in SURVEY_QUESTIONS}
    if "sector_selection" not in st.session_state:
        st.session_state.sector_selection = set()
    if "budget_text" not in st.session_state:
        st.session_state.budget_text = f"{1_000_000:,}"
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""

    budget_digits = re.sub(r"[^\d]", "", st.session_state.budget_text)
    budget = int(budget_digits) if budget_digits else 0
    budget = max(MIN_BUDGET, min(MAX_BUDGET, budget))

    can_advance = True
    error_message = None

    st.markdown(
        f"""
        <div class="fomo-survey-page"></div>
        <header class="fomo-survey-header">
            <a class="fomo-survey-logo" href="?page=intro" aria-label="{SITE_NAME} 홈">
                {get_logo_html()}
            </a>
            <div class="fomo-survey-count">{step + 1} / {TOTAL_SURVEY_STEPS}</div>
        </header>
        """,
        unsafe_allow_html=True,
    )
    st.progress((step + 1) / TOTAL_SURVEY_STEPS)

    if step == NAME_STEP:
        st.markdown(
            """
            <section class="fomo-survey-copy">
                <span class="fomo-survey-badge is-blue">시작하기</span>
                <h1>안녕하세요 👋<br>이름을 알려주세요</h1>
                <p>닉네임도 괜찮아요.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )
        # 이름 위젯(key="user_name_widget")은 이 스텝에서만 렌더링되는데, Streamlit은
        # 어떤 스텝에서도 렌더링되지 않은 위젯의 세션 상태를 다음 렌더링에서 지워버린다.
        # 그래서 실제 값은 스텝과 무관한 별도 키(user_name)에 보관하고, 위젯은 그 값을
        # 초기값으로 보여준 뒤 매번 다시 동기화한다.
        user_name = st.text_input(
            "이름 (또는 별명)",
            value=st.session_state.user_name,
            key="user_name_widget",
            placeholder="예: 홍길동",
            label_visibility="collapsed",
        )
        st.session_state.user_name = user_name
        can_advance = bool(user_name.strip())
        error_message = "이름을 입력해주세요."

    elif step == BUDGET_STEP:
        st.markdown(
            f"""
            <section class="fomo-survey-copy">
                <span class="fomo-survey-badge is-green">투자 예산</span>
                <h1>총 투자 예산은<br>얼마인가요? 💰</h1>
                <p>최소 {MIN_BUDGET:,}원 ~ 최대 {MAX_BUDGET:,}원</p>
            </section>
            """,
            unsafe_allow_html=True,
        )
        st.text_input(
            "총 투자 예산 (원)",
            value=st.session_state.budget_text,
            key="budget_text_widget",
            on_change=_format_budget_input,
            label_visibility="collapsed",
        )

    elif step in QUESTION_STEPS:
        q_idx = step - QUESTION_STEPS[0]
        q = SURVEY_QUESTIONS[q_idx]
        with st.container(key="survey_question_card"):
            st.markdown(
                f"""
                <section class="fomo-survey-copy">
                    <span class="fomo-survey-badge is-purple">성향 분석 · {q_idx + 1} / {len(SURVEY_QUESTIONS)}</span>
                    <h1>{q["question"]}</h1>
                </section>
                """,
                unsafe_allow_html=True,
            )

            labels = [label for label, _ in q["options"]]
            value_map = dict(q["options"])
            label_by_value = {v: k for k, v in q["options"]}

            current_value = st.session_state.survey_wip.get(q["feature"])
            current_label = label_by_value.get(current_value)
            default_index = labels.index(current_label) if current_label in labels else None

            selected_label = st.radio(
                q["question"],
                labels,
                index=default_index,
                key=q["key"],
                label_visibility="collapsed",
            )
        st.session_state.survey_wip[q["feature"]] = value_map.get(selected_label)
        can_advance = selected_label is not None
        error_message = "답을 선택해주세요."

    else:  # SECTOR_STEP
        st.markdown(
            """
            <section class="fomo-survey-copy">
                <span class="fomo-survey-badge is-orange">거의 다 왔어요!</span>
                <h1>🏭 관심 있는 산업을 선택해주세요.</h1>
                <p>여러 개 선택 가능합니다.<br>선택하지 않으면 전체 종목을 추천합니다.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )
        scored = load_scored_stocks()
        sector_options = sorted(scored["업종명"].dropna().unique().tolist())
        render_sector_selector(sector_options)

    with st.container(key="survey_bottom_nav"):
        nav1, _, nav2 = st.columns([2, 6, 2])
        with nav1:
            if step > 0:
                if st.button("← 이전"):
                    st.session_state.survey_step = step - 1
                    st.rerun()
            else:
                if st.button("← 처음으로"):
                    go_to("intro")
                    st.rerun()
        with nav2:
            if step < TOTAL_SURVEY_STEPS - 1:
                if st.button("다음 →", type="primary"):
                    if not can_advance:
                        st.error(error_message)
                    else:
                        st.session_state.survey_step = step + 1
                        st.rerun()
            else:
                if st.button("내 투자성향 확인하기 →", type="primary"):
                    st.session_state.budget = budget
                    st.session_state.survey_answers = dict(st.session_state.survey_wip)
                    go_to("profile")
                    st.rerun()


# ------------------------------------------------------------
# 페이지 3: 내 투자성향 확인하기
# ------------------------------------------------------------
def _get_profile_proba():
    """survey_answers 기반으로 성향별 확률(dict[성향, 확률])을 반환한다. 답변이 없으면 None."""
    answers = st.session_state.get("survey_answers")
    if not answers:
        return None
    model = load_model()
    X = pd.DataFrame([answers])[FEATURE_COLS]
    proba = model.predict_proba(X)[0]
    return dict(zip(model.classes_, proba))


def predict_profile():
    """survey_answers를 기반으로 투자 성향을 예측한다 (profile/recommend 페이지 공용)."""
    proba = _get_profile_proba()
    if proba is None:
        return None
    return max(proba, key=proba.get)


# 게이지에서 성향별 위치 점수 (안정형=0 ~ 공격형=100)
PROFILE_SCORE_MAP = {"안정형": 0, "중립형": 50, "공격형": 100}

# 설문 문항의 feature 키 -> 근거 요약에 쓸 짧은 항목명
FEATURE_LABELS = {
    "손실감수수준": "손실 감수도",
    "기대수익률": "기대 수익률",
    "투자기간": "투자 기간",
    "배당선호도": "배당 선호도",
    "변동성감수수준": "변동성 감수도",
}


def render_profile_page():
    from html import escape

    user_name = (st.session_state.get("user_name") or "").strip()
    proba = _get_profile_proba()
    predicted_profile = max(proba, key=proba.get) if proba else None
    if not user_name or predicted_profile is None:
        st.error("설문 응답이 없습니다. 설문 페이지로 돌아가주세요.")
        if st.button("← 설문으로 돌아가기", key="profile_missing_go_survey"):
            go_to("survey")
            st.rerun()
        return

    gauge_score = sum(
        proba.get(profile, 0) * score
        for profile, score in PROFILE_SCORE_MAP.items()
    )
    gauge_pct = max(0.0, min(100.0, gauge_score))

    answers = st.session_state.get("survey_answers", {})
    answer_rows = []
    for q in SURVEY_QUESTIONS:
        feature = q["feature"]
        value = answers.get(feature)
        label_by_value = {v: k for k, v in q["options"]}
        answer_label = label_by_value.get(value, "-")
        answer_rows.append(
            '<div class="profile-answer-row">'
            f'<span class="profile-answer-label">{escape(FEATURE_LABELS.get(feature, feature))}</span>'
            f'<span class="profile-answer-value">{escape(str(answer_label))}</span>'
            '</div>'
        )

    strategy_rows = "".join(
        '<div class="profile-strategy-row">'
        f'<span class="profile-strategy-number">{index}</span>'
        f'<span>{escape(tip)}</span>'
        '</div>'
        for index, tip in enumerate(STRATEGY_TIPS.get(predicted_profile, []), start=1)
    )

    with st.container(key="profile_result_page"):
        st.markdown(
            f"""
            <section class="profile-card profile-summary-card">
                <div class="profile-badge">🎯 {escape(predicted_profile)}</div>
                <h1>{escape(user_name)}님은 <strong>{escape(predicted_profile)}</strong>이에요</h1>
                <div class="profile-description">
                    <span class="profile-info-icon">i</span>
                    <span>{escape(PROFILE_DESCRIPTIONS.get(predicted_profile, ""))}</span>
                </div>
                <h2>추천 투자전략</h2>
                <div class="profile-strategy-list">{strategy_rows}</div>
            </section>

            <section class="profile-card profile-spectrum-card">
                <h2>투자성향 스펙트럼</h2>
                <div class="profile-gauge" role="img" aria-label="투자성향 스펙트럼 {gauge_pct:.1f}%">
                    <div class="profile-gauge-fill" style="width:{gauge_pct:.4f}%"></div>
                    <div class="profile-gauge-marker" style="left:{gauge_pct:.4f}%"></div>
                </div>
                <div class="profile-gauge-labels">
                    <span>안정형</span><span>중립형</span><span>공격형</span>
                </div>
            </section>

            <section class="profile-card profile-answers-card">
                <h2>이렇게 답변했어요</h2>
                <div class="profile-answer-list">{''.join(answer_rows)}</div>
            </section>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← 설문 다시하기", key="profile_restart_survey", use_container_width=True):
                reset_survey_state()
                go_to("survey")
                st.rerun()
        with col2:
            if st.button(
                "📈 종목 추천받으러가기",
                type="primary",
                key="profile_go_recommend",
                use_container_width=True,
            ):
                go_to("recommend")
                st.rerun()


# ------------------------------------------------------------
# 페이지 4: 종목 추천받기
# ------------------------------------------------------------
def render_recommend_page():
    from html import escape

    budget = st.session_state.get("budget")
    predicted_profile = predict_profile()
    if not budget or predicted_profile is None:
        st.error("설문 응답이 없습니다. 설문 페이지로 돌아가주세요.")
        if st.button("← 설문으로 돌아가기", key="recommend_missing_go_survey"):
            go_to("survey")
            st.rerun()
        return

    scored = load_scored_stocks()

    # 관심 섹터는 2페이지(설문) 마지막 단계에서 이미 선택했으므로 여기서는 그 값을 그대로 사용한다.
    selected_sectors = sorted(st.session_state.get("sector_selection", set()))
    if selected_sectors:
        sector_message = f"관심 섹터({', '.join(selected_sectors)}) 안에서 추천합니다."
    else:
        sector_message = "전체 종목을 대상으로 추천합니다."

    candidates = scored.copy()
    if selected_sectors:
        candidates = candidates[candidates["업종명"].isin(selected_sectors)]

    score_col = f"추천점수_{predicted_profile}"
    reason_col = f"추천이유_{predicted_profile}"

    with st.container(key="recommend_result_page"):
        st.markdown(
            '<header class="recommend-header">'
            '<h1>🎯 종목 추천</h1>'
            f'<p>{escape(sector_message)}</p>'
            '</header>',
            unsafe_allow_html=True,
        )

        if candidates.empty:
            st.warning("선택한 섹터에 해당하는 종목이 없습니다. 섹터 선택을 조정해주세요.")
        else:
            allocated = select_top_n_within_budget(candidates, budget, score_col, target_n=10)

            if allocated.empty:
                with st.container(key="recommend_stocks_card"):
                    st.markdown("### 🏆 추천 종목 상위 10개")
                    st.warning("예산이 부족해 1주도 매수할 수 없습니다. 예산을 늘려주세요.")
            else:
                stock_rows = []
                for _, row in allocated.iterrows():
                    stock_rows.append(
                        '<tr>'
                        f'<td class="stock-code">{escape(str(row["종목코드"]))}</td>'
                        f'<td class="stock-name">{escape(str(row["종목명"]))}</td>'
                        f'<td><span class="sector-pill">{escape(str(row["업종명"]))}</span></td>'
                        f'<td class="current-price">{row["현재가"]:,.0f}</td>'
                        f'<td>{row["매수수량"]:,.0f}</td>'
                        f'<td>{row["투자금액"]:,.0f}</td>'
                        f'<td><span class="score-pill">{row[score_col]:,.1f}</span></td>'
                        '</tr>'
                    )
                shortage_text = (
                    f'예산으로 매수 가능한 종목이 {len(allocated)}개뿐이라 '
                    '그만큼만 추천합니다. 예산을 늘리면 더 많이 추천됩니다.'
                    if len(allocated) < 10
                    else '예산 내에서 매수 가능한 상위 추천 종목입니다.'
                )
                st.markdown(
                    '<section class="recommend-card recommend-stocks-card">'
                    '<h2>🏆 추천 종목 상위 10개</h2>'
                    f'<p class="card-description">{escape(shortage_text)}</p>'
                    '<div class="recommend-table-wrap"><table class="recommend-table">'
                    '<thead><tr><th>종목코드</th><th>종목명</th><th>업종명</th>'
                    '<th>현재가(원)</th><th>매수수량(주)</th><th>투자금액(원)</th><th>추천점수</th></tr></thead>'
                    f'<tbody>{"".join(stock_rows)}</tbody></table></div></section>',
                    unsafe_allow_html=True,
                )

                etf_list = get_etf_recommendations(predicted_profile, selected_sectors)
                if etf_list:
                    etf_rows = "".join(
                        '<div class="recommend-etf-item">'
                        f'<div><strong>{escape(str(etf["종목명"]))}</strong> '
                        f'<span>({escape(str(etf["종목코드"]))})</span> - {escape(str(etf["설명"]))}</div>'
                        f'<small>주요 구성종목: {escape(str(etf["구성종목"]))}</small>'
                        '</div>'
                        for etf in etf_list
                    )
                    st.markdown(
                        '<section class="recommend-card recommend-etf-card">'
                        '<h2>📦 추천 ETF</h2>'
                        '<p class="card-description">개별 종목과 별개로, 분산투자용으로 참고할 만한 ETF예요.</p>'
                        f'<div class="recommend-etf-list">{etf_rows}</div>'
                        '<p class="recommend-note">※ 구성종목은 대표 예시이며, 정확한 비중은 운용사 홈페이지에서 확인하세요.</p>'
                        '</section>',
                        unsafe_allow_html=True,
                    )

                insights = generate_ai_insights(predicted_profile, allocated, score_col, reason_col)
                if insights:
                    with st.container(key="recommend_ai_card"):
                        st.markdown("### 😊 AI 요약")
                        st.write(insights["전체요약"])
                elif get_gemini_client() is None:
                    with st.container(key="recommend_ai_card"):
                        st.markdown("### 😊 AI 요약")
                        st.caption("💡 Colab Secrets에 GEMINI_API_KEY를 등록하면 추천 결과를 AI가 자연어로 요약해줍니다.")

                total_invested = allocated["투자금액"].sum()
                remaining = budget - total_invested
                investment_ratio = total_invested / budget * 100 if budget > 0 else 0
                st.markdown(
                    '<section class="recommend-card recommend-budget-card">'
                    '<h2>💰 예산 요약</h2>'
                    '<div class="budget-summary-grid">'
                    f'<div><span>총 예산</span><strong>{budget:,.0f}원</strong></div>'
                    f'<div><span>총 투자금액</span><strong class="invested">{total_invested:,.0f}원</strong></div>'
                    f'<div><span>미투자 잔액</span><strong class="remaining">{remaining:,.0f}원</strong></div>'
                    '</div>'
                    '<div class="investment-ratio-label">'
                    f'<span>투자 비율</span><span>{investment_ratio:.0f}%</span></div>'
                    '<div class="investment-ratio-track">'
                    f'<div style="width:{max(0.0, min(100.0, investment_ratio)):.4f}%"></div></div>'
                    '</section>',
                    unsafe_allow_html=True,
                )

                sub_score_cols = ["수익성점수", "안정성점수", "가치점수", "배당점수"]
                stock_order = allocated["종목명"].tolist()
                bar_df = allocated[["종목명"] + sub_score_cols].melt(
                    id_vars="종목명", var_name="구성 요소", value_name="점수"
                )
                total_df = allocated[["종목명", score_col]].rename(columns={score_col: "추천점수"})
                bars = alt.Chart(bar_df).mark_bar(cornerRadiusEnd=4).encode(
                    y=alt.Y("종목명:N", sort=stock_order, title=None, axis=alt.Axis(labelLimit=110)),
                    x=alt.X("점수:Q", stack="zero", title=None, axis=None),
                    color=alt.Color(
                        "구성 요소:N",
                        sort=sub_score_cols,
                        scale=alt.Scale(domain=sub_score_cols, range=["#203f91", "#13bce8", "#ff6974", "#ef91ca"]),
                        legend=alt.Legend(title=None, orient="bottom", direction="horizontal"),
                    ),
                    tooltip=["종목명", "구성 요소", alt.Tooltip("점수:Q", format=".1f")],
                )
                labels = alt.Chart(total_df).mark_text(align="left", dx=8, color="#526071").encode(
                    y=alt.Y("종목명:N", sort=stock_order),
                    x=alt.X("추천점수:Q"),
                    text=alt.Text("추천점수:Q", format=".0f"),
                )
                chart = (bars + labels).properties(height=max(220, 34 * len(stock_order) + 55)).configure_view(stroke=None)
                with st.container(key="recommend_chart_card"):
                    st.markdown("### 📊 추천 점수 그래프")
                    st.caption("종목별 4개 점수(수익성/안정성/가치/배당)를 가로 막대로 쌓아서 보여줍니다.")
                    st.altair_chart(chart, use_container_width=True)

                with st.container(key="recommend_details_card"):
                    st.markdown("### 📋 종목별 상세 정보")
                    ai_per_stock = insights["종목별"] if insights else {}
                    for index, (_, row) in enumerate(allocated.iterrows()):
                        reason_text = ai_per_stock.get(row["종목코드"], row[reason_col])
                        with st.expander(
                            f"{row['종목명']} ({row['종목코드']}) - {reason_text}",
                            expanded=False,
                        ):
                            render_investment_metrics_card(row)

        button_col1, button_spacer, button_col2 = st.columns([1.2, 2, 1.2])
        with button_col1:
            if st.button("← 설문 다시하기", key="recommend_restart_survey", use_container_width=True):
                reset_survey_state()
                go_to("survey")
                st.rerun()
        with button_col2:
            if st.button("처음으로 🏠", key="recommend_go_home", type="primary", use_container_width=True):
                reset_survey_state()
                go_to("intro")
                st.rerun()


# ------------------------------------------------------------
# 메인 라우팅
# ------------------------------------------------------------
def main():
    sync_page_from_query_params()
    current_page = st.session_state.get("page", "intro")
    load_css("common.css")
    if current_page == "intro":
        load_css("intro.css")
    elif current_page == "survey":
        load_css("survey.css")
    elif current_page == "profile":
        load_css("profile.css")
    elif current_page == "recommend":
        load_css("recommend.css")

    missing = files_missing()
    if missing:
        st.error(
            f"다음 파일이 없어 앱을 실행할 수 없습니다: {', '.join(missing)}. "
            "1단계(risk_model.pkl)와 3단계(scored_stocks.csv)를 먼저 실행해주세요."
        )
        return

    if st.session_state.page != "survey":
        render_navbar()

    if st.session_state.page == "intro":
        render_intro_page()
    elif st.session_state.page == "survey":
        render_survey_page()
    elif st.session_state.page == "profile":
        render_profile_page()
    elif st.session_state.page == "recommend":
        render_recommend_page()

    if st.session_state.page != "survey":
        render_disclaimer()


main()
