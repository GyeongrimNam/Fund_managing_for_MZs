# FOMO — 투자성향 맞춤 종목 추천 서비스

주식 투자 기초 지식이 부족하고 투자 가능한 금액이 많지 않은 사회초년생/초보 투자자를 위한
사용자 맞춤형 투자 추천 서비스입니다. 간단한 설문으로 투자 성향(안정형/중립형/공격형)을
예측하고, 관심 섹터 안에서 예산에 맞는 추천 종목을 보여줍니다.

> ⚠️ **이 서비스는 투자 자문이 아닙니다.** 참고용 데모이며, 실제 투자 판단과 결과에 대한
> 책임은 전적으로 사용자 본인에게 있습니다.

## 구성 (5단계 파이프라인)

| 파일 | 역할 |
|---|---|
| `step1_train_model.py` | 설문 응답(1~5 스케일) → 투자 성향(안정형/중립형/공격형) 분류 모델 학습, `risk_model.pkl` 생성 |
| `step2_collect_stock_data.py` | pykrx로 KOSPI+KOSDAQ 시가총액 상위 50개 종목 데이터 수집, `stock_data.csv` 생성 |
| `step3_score_stocks.py` | 수익성/안정성/가치/배당 점수 및 성향별 추천점수·추천이유 계산, `scored_stocks.csv` 생성 |
| `app.py` | Streamlit 4페이지 웹앱 (소개 → 설문 → 내 투자성향 확인 → 종목 추천) |
| `step5_run_on_colab.md` | Google Colab에서 cloudflared 터널로 앱을 외부에 공개하는 방법 |

## 1. 설치

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. KRX 계정 준비 (종목 데이터 수집용)

`step2_collect_stock_data.py`는 시세/재무 데이터를 [pykrx](https://github.com/sharebook-kr/pykrx)로
가져오며, data.krx.co.kr 로그인 세션이 필요합니다.

1. https://data.krx.co.kr 접속 후 무료 회원가입 (증권 계좌가 아닌 사이트 로그인 ID/PW입니다)
2. 발급받은 아이디/비밀번호를 `.env`에 입력

```bash
cp .env.example .env
```

```
KRX_ID=발급받은_아이디
KRX_PW=발급받은_비밀번호
```

`.env`는 `.gitignore`에 포함되어 커밋되지 않습니다.

> ⚠️ pykrx는 인증되지 않은 상태에서 반복 요청 시 내부적으로 재로그인을 시도합니다. 잘못된
> 계정 정보로 반복 실행하면 KRX 계정이 잠길 수 있으니, 로그인 실패가 이어지면 바로 재시도하지
> 말고 원인을 먼저 확인하세요 (`step2_collect_stock_data.py`에 로그인 시도 상한 안전장치가 있습니다).

## 3. (선택) AI 요약용 GEMINI_API_KEY

종목 추천 결과를 AI가 자연어로 요약해주는 기능(선택)을 쓰려면 `.env`에 `GEMINI_API_KEY`를
추가하세요. [Google AI Studio](https://aistudio.google.com/apikey)에서 무료로 발급받을 수
있습니다. 키가 없어도 나머지 기능은 규칙 기반 설명으로 정상 동작합니다.

## 4. 파이프라인 실행 (순서대로)

```bash
python step1_train_model.py          # risk_model.pkl, investment_users.csv 생성
python step2_collect_stock_data.py   # stock_data.csv 생성
python step3_score_stocks.py         # scored_stocks.csv 생성
streamlit run app.py                 # 웹앱 실행
```

브라우저가 자동으로 열리지 않으면 터미널에 표시된 `http://localhost:8501` 주소로 접속하세요.

## Google Colab에서 실행하기

로컬 환경 없이 Colab에서 전체 파이프라인을 실행하고 cloudflared 터널로 외부에 공개하려면
`step5_run_on_colab.md`를 참고하세요.

## 동작 방식 요약

- 설문 5문항(손실감수수준/기대수익률/투자기간/배당선호도/변동성감수수준)을 1~5 스케일로 받아
  RandomForestClassifier로 투자 성향을 예측합니다.
- 성향별 가중치로 수익성/안정성/가치/배당 점수를 합산해 추천점수를 매기고, 예산 안에서
  1주도 못 사는 종목은 건너뛰고 다음 순위 종목으로 채워 상위 10개를 추천합니다.
- (선택) Gemini API로 전체 요약 + 종목별 한줄평을 한 번의 호출로 생성합니다.
