"""
감성분석 및 Opportunity Score 산출 (CAM)
====================================================

[목적]
Actor × Action 조합별로 만족도(Satisfaction)와 중요도(Importance)를 계산하고,
이를 결합한 Opportunity Score로 개선이 시급한 기회 영역(CAM)을 도출합니다.

[방법 요약]
1. KNU 감성사전으로 형태소 토큰과 단어를 대조해 감성 점수 산출
2. Actor-Action 조합별 평균 감성 점수를 만족도(Satisfaction)로 사용, -10~10으로 정규화
3. 전체 문서 대비 각 Actor-Action 조합의 등장 비율을 중요도(Importance)로 사용, 0~10으로 정규화
4. Opportunity Score = Importance + max(Importance - Satisfaction, 0)
   → 중요도는 높은데 만족도는 낮을수록 점수가 커지도록 설계해 개선 우선순위를 표시
5. Importance(x) × Satisfaction(y) 산점도로 기회 영역(CAM)을 시각화
"""

import json
from collections import Counter

import numpy as np
import pandas as pd
from konlpy.tag import Okt
from kiwipiepy import Kiwi
from sklearn.preprocessing import MinMaxScaler

okt = Okt()
kiwi = Kiwi()


# =========================================================
# 1. 감성 점수 산출
# =========================================================
def load_sentiment_dict(path: str = "SentiWord_info.json") -> list[dict]:
    """KNU 감성사전(단어/어간/극성 정보)을 불러온다."""
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def tokenize_for_sentiment(text: str) -> list[str]:
    """Kiwi로 띄어쓰기를 보정한 뒤 Okt로 형태소(기본형)를 분석한다."""
    corrected = kiwi.space(text)
    return okt.morphs(corrected, stem=True, norm=True)


def sentiment_score(sent_dict: list[dict], tokens: list[str]) -> int:
    """토큰이 감성사전에 존재하면 해당 극성(polarity) 점수를 모두 합산한다."""
    word_to_polarity = {s["word"]: int(s["polarity"]) for s in sent_dict}
    return sum(word_to_polarity.get(t, 0) for t in tokens if t in word_to_polarity)


def add_sentiment_scores(df: pd.DataFrame, sent_dict: list[dict]) -> pd.DataFrame:
    df = df.copy()
    df["sentiment_score"] = df["review"].apply(lambda x: sentiment_score(sent_dict, tokenize_for_sentiment(x)))
    return df


# =========================================================
# 2. Satisfaction (Actor-Action별 평균 감성점수, -10~10 정규화)
# =========================================================
def compute_satisfaction(df: pd.DataFrame) -> pd.DataFrame:
    scores = {}
    for actor in df["actor_cluster"].unique():
        actor_df = df[df["actor_cluster"] == actor]
        for action in actor_df["action_cluster"].unique():
            key = f"Actor{actor}_Action{action}"
            scores[key] = actor_df.loc[actor_df["action_cluster"] == action, "sentiment_score"].mean()

    values = np.array(list(scores.values())).reshape(-1, 1)
    normalized = MinMaxScaler(feature_range=(-10, 10)).fit_transform(values).flatten()

    return pd.DataFrame(
        {"Action": list(scores.keys()), "satisfaction": np.round(normalized, 4)}
    )


# =========================================================
# 3. Importance (전체 대비 등장 비율, 0~10 정규화)
# =========================================================
def compute_importance(df: pd.DataFrame, sents_df: pd.DataFrame) -> pd.DataFrame:
    combo_labels = [f"Actor{a}_Action{b}" for a, b in zip(df["actor_cluster"], df["action_cluster"])]
    freq = Counter(combo_labels)
    total = sum(freq.values())

    importance = {k: (v / total) * 100 for k, v in freq.items()}
    values = np.array(list(importance.values())).reshape(-1, 1)
    normalized = MinMaxScaler(feature_range=(0, 10)).fit_transform(values).flatten()
    importance = dict(zip(importance.keys(), np.round(normalized, 4)))

    sents_df = sents_df.copy()
    sents_df["importance"] = sents_df["Action"].map(importance)
    return sents_df


# =========================================================
# 4. Opportunity Score
# =========================================================
def compute_opportunity_score(satisfaction: float, importance: float) -> float:
    """중요도는 높은데 만족도는 낮을수록 값이 커지는 개선 우선순위 지표."""
    return importance + max(importance - satisfaction, 0)


def add_opportunity_scores(sents_df: pd.DataFrame) -> pd.DataFrame:
    sents_df = sents_df.copy()
    sents_df["Opportunity_Score"] = [
        compute_opportunity_score(sat, imp)
        for sat, imp in zip(sents_df["satisfaction"], sents_df["importance"])
    ]
    return sents_df


# =========================================================
# 5. CAM 시각화 (Importance x Satisfaction 산점도)
# =========================================================
def plot_opportunity_area(sents_df: pd.DataFrame, output_path: str = "opportunity_area.png"):
    import matplotlib.pyplot as plt
    from adjustText import adjust_text

    actions = sents_df["Action"]
    importance = sents_df["importance"]
    satisfaction = sents_df["satisfaction"]
    colors = np.random.rand(len(actions), 3)

    plt.figure(figsize=(17, 10))
    for i, action in enumerate(actions):
        plt.scatter(importance[i], satisfaction[i], c=[colors[i]], label=action, s=50, edgecolors="black")

    plt.legend(title="Actions", fontsize=8, title_fontsize=10, loc="best", bbox_to_anchor=(1, 1))
    plt.xlabel("Importance")
    plt.ylabel("Satisfaction")
    plt.axhline(satisfaction.mean(), color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print("저장 완료:", output_path)
    plt.show()


if __name__ == "__main__":
    df = pd.read_pickle("action_cluster_result.pkl")
    sent_dict = load_sentiment_dict()

    df = add_sentiment_scores(df, sent_dict)

    satisfaction_df = compute_satisfaction(df)
    result_df = compute_importance(df, satisfaction_df)
    result_df = add_opportunity_scores(result_df)

    result_df = result_df.sort_values("Opportunity_Score", ascending=False)
    print(result_df)

    result_df.to_csv("opportunity_score_result.csv", index=False, encoding="utf-8-sig")
    plot_opportunity_area(result_df)
