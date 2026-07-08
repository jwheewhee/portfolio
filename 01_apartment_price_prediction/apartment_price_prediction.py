"""
아파트 실거래가 예측 - 데이콘 해커톤 (2위)
================================================

[프로젝트 개요]
데이콘 주최 아파트 실거래가 예측 해커톤 참가 코드입니다.
전용면적, 층수, 건축연도, 거래 연월 정보를 바탕으로 아파트 실거래가를 예측하고,
MAE(평균절대오차)를 최소화하는 것이 목표입니다.

[핵심 아이디어]
단일 모델로는 예측 오차 개선에 한계가 있어, 성격이 다른 8개의 회귀 모델을
보팅 앙상블(Voting Ensemble)로 결합하여 예측 안정성과 정확도를 높였습니다.

사용 모델: LightGBM, Random Forest, XGBoost, CatBoost,
          Ridge, Gradient Boosting, SVR, AdaBoost

[검증 결과]
보팅 앙상블 적용 후 검증 데이터 기준 MAE 약 11,578 달성
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from sklearn.ensemble import (
    VotingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


# -----------------------------------------------------------------------
# 1. 데이터 불러오기
# -----------------------------------------------------------------------
def load_data(train_path: str = "train.csv", test_path: str = "test.csv"):
    """train/test 데이터를 불러온다."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


# -----------------------------------------------------------------------
# 2. 전처리 (EDA 결과를 바탕으로 예측에 사용할 변수만 선택)
# -----------------------------------------------------------------------
def preprocess(train: pd.DataFrame, test: pd.DataFrame):
    """
    - 예측에 실제로 사용할 변수만 선택
      (전용면적, 거래연월, 층수, 건축연도 [+ target: 실거래가])
    - 거래연월(YYYYMM 형태)을 연도/월로 분리해서 별도 변수로 활용
    """
    use_cols_train = [
        "exclusive_use_area",       # 전용면적
        "transaction_year_month",   # 거래 연월 (YYYYMM)
        "floor",                    # 층수
        "year_of_completion",       # 건축연도
        "transaction_real_price",   # 실거래가 (target)
    ]
    use_cols_test = use_cols_train[:-1]  # test에는 target이 없음

    train = train[use_cols_train].copy()
    test = test[use_cols_test].copy()

    # 거래연월(YYYYMM) -> 연도 / 월로 분리
    for df in (train, test):
        df["transaction_year"] = df["transaction_year_month"].astype(str).str[:4].astype(int)
        df["transaction_month"] = df["transaction_year_month"].astype(str).str[4:6].astype(int)

    # 분리 후 원본 컬럼은 제거
    train.drop("transaction_year_month", axis=1, inplace=True)
    test.drop("transaction_year_month", axis=1, inplace=True)

    return train, test


# -----------------------------------------------------------------------
# 3. 모델 정의 (성격이 다른 8개 회귀 모델)
# -----------------------------------------------------------------------
def build_voting_ensemble() -> VotingRegressor:
    """
    8개의 서로 다른 회귀 모델을 하이퍼파라미터 튜닝하여 구성하고,
    이를 VotingRegressor로 묶어 평균 예측하는 앙상블 모델을 반환한다.

    단일 모델의 편향을 서로 다른 모델 조합으로 상쇄시켜
    예측 오차(MAE)를 안정적으로 낮추는 것이 목적이다.
    """
    model_lgb = LGBMRegressor(
        colsample_bytree=0.9,
        learning_rate=0.2,
        max_depth=8,
        n_estimators=200,
        subsample=0.8,
    )
    model_rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        random_state=42,
    )
    model_xgb = XGBRegressor(
        learning_rate=0.01,
        n_estimators=150,
        max_depth=6,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.005,
        random_state=42,
    )
    model_catboost = CatBoostRegressor(
        iterations=10000,
        learning_rate=1,
        depth=4,
        loss_function="MAE",  # 대회 평가지표(MAE)에 맞춰 손실함수 지정
        verbose=False,
    )
    model_ridge = Ridge(alpha=0.1)
    model_gb = GradientBoostingRegressor(
        learning_rate=0.1,
        n_estimators=150,
        max_depth=3,
        random_state=42,
    )
    model_svr = SVR(kernel="rbf", C=1.0, epsilon=0.2)
    model_adaboost = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=100),
        n_estimators=200,
        learning_rate=0.01,
        loss="linear",
    )

    voting_models = [
        ("lgb", model_lgb),
        ("rf", model_rf),
        ("xgb", model_xgb),
        ("catboost", model_catboost),
        ("ridge", model_ridge),
        ("gb", model_gb),
        ("svr", model_svr),
        ("adaboost", model_adaboost),
    ]

    return VotingRegressor(estimators=voting_models)


# -----------------------------------------------------------------------
# 4. 학습 및 검증
# -----------------------------------------------------------------------
def train_and_validate(train: pd.DataFrame):
    """
    train 데이터를 학습/검증셋으로 분리해 보팅 앙상블 모델을 학습하고,
    검증셋 MAE를 확인한다.
    """
    train_x = train.drop("transaction_real_price", axis=1)
    train_y = train["transaction_real_price"]

    X_train, X_val, y_train, y_val = train_test_split(
        train_x, train_y, test_size=0.03273, random_state=42
    )

    voting_ensemble = build_voting_ensemble()
    voting_ensemble.fit(X_train, y_train)

    val_pred = voting_ensemble.predict(X_val)
    mae = mean_absolute_error(y_val, val_pred)
    print(f"보팅 앙상블 검증 MAE: {mae:.4f}")

    return voting_ensemble


# -----------------------------------------------------------------------
# 5. 예측 및 제출 파일 생성
# -----------------------------------------------------------------------
def predict_and_submit(
    model,
    test: pd.DataFrame,
    sample_submission_path: str = "sample_submission.csv",
    output_path: str = "voting_ensemble.csv",
):
    """
    학습된 모델로 test 데이터를 예측하고,
    대회에서 제공하는 sample_submission.csv 양식(id, transaction_real_price)에 맞춰
    제출 파일을 생성한다.
    """
    pred = model.predict(test)

    submission = pd.read_csv(sample_submission_path)
    submission["transaction_real_price"] = pred
    submission.to_csv(output_path, index=False)

    print(f"제출 파일 저장 완료: {output_path}")


# -----------------------------------------------------------------------
# 실행부
# -----------------------------------------------------------------------
if __name__ == "__main__":
    train, test = load_data()
    train, test = preprocess(train, test)
    model = train_and_validate(train)
    predict_and_submit(model, test)
