# ============================================================
# 1단계: 투자 성향 분류 모델 만들기 (Google Colab에서 실행)
# ============================================================
# 손실 감수 수준, 기대 수익률, 투자 기간, 배당 선호도, 변동성 감수 수준(1~5)을
# 입력값으로 사용해 투자 성향(안정형/중립형/공격형)을 예측하는
# RandomForestClassifier를 학습합니다.

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ------------------------------------------------------------
# 1. 가상 학습 데이터 300개 생성
# ------------------------------------------------------------
np.random.seed(42)
N = 300

# 설문 응답(1~5 정수)을 균등 분포로 생성
loss_tolerance = np.random.randint(1, 6, N)       # 손실 감수 수준
expected_return = np.random.randint(1, 6, N)      # 기대 수익률
investment_period = np.random.randint(1, 6, N)    # 투자 기간
dividend_preference = np.random.randint(1, 6, N)  # 배당 선호도
volatility_tolerance = np.random.randint(1, 6, N) # 변동성 감수 수준

# 성향 점수 = 공격적인 성향일수록 커지는 가중합
# (배당 선호도는 반대로 작용: 배당을 선호할수록 안정 지향)
risk_score = (
    loss_tolerance * 1.0
    + expected_return * 1.0
    + investment_period * 0.6
    + volatility_tolerance * 1.0
    - dividend_preference * 0.6
)

# 약간의 노이즈를 더해 현실적인 데이터로 만듦 (라벨이 100% 결정론적이지 않게)
risk_score = risk_score + np.random.normal(0, 0.3, N)

# 점수를 3분위로 나눠 성향 라벨 부여
low_cut, high_cut = np.percentile(risk_score, [33.33, 66.67])


def label_from_score(score):
    if score <= low_cut:
        return "안정형"
    elif score <= high_cut:
        return "중립형"
    else:
        return "공격형"


risk_profile = np.array([label_from_score(s) for s in risk_score])

df = pd.DataFrame(
    {
        "손실감수수준": loss_tolerance,
        "기대수익률": expected_return,
        "투자기간": investment_period,
        "배당선호도": dividend_preference,
        "변동성감수수준": volatility_tolerance,
        "투자성향": risk_profile,
    }
)

df.to_csv("investment_users.csv", index=False, encoding="utf-8-sig")
print("investment_users.csv 저장 완료 (샘플 수:", len(df), ")")
print(df["투자성향"].value_counts())
print(df.head())

# ------------------------------------------------------------
# 2. 학습/테스트 데이터 분리 (8:2)
# ------------------------------------------------------------
feature_cols = ["손실감수수준", "기대수익률", "투자기간", "배당선호도", "변동성감수수준"]
X = df[feature_cols]
y = df["투자성향"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------
# 3. RandomForestClassifier 학습
# ------------------------------------------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ------------------------------------------------------------
# 4. 모델 평가
# ------------------------------------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n모델 정확도: {acc:.4f}")
print("\n분류 결과:")
print(classification_report(y_test, y_pred))

# ------------------------------------------------------------
# 5. 모델 저장
# ------------------------------------------------------------
joblib.dump(model, "risk_model.pkl")
print("risk_model.pkl 저장 완료")
