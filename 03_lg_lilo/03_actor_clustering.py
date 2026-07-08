"""
Actor 도출 — TF-IDF + KMeans 클러스터링
====================================================

[목적]
가전 구독을 언급한 게시글·댓글 3만 8천여 건을 유사한 맥락끼리 묶어,
서비스 기획에 활용할 수 있는 6개의 Actor(고객 유형)로 정리합니다.

[방법 요약]
1. Kiwi 형태소 분석으로 제목+본문에서 의미 있는 토큰만 추출
2. TF-IDF(Dense)로 벡터화 (min_df=4, max_df=0.40, max_features=300, sublinear_tf=True)
3. k=4~9 구간에서 실루엣 지수를 비교해 군집 수 결정
   - 실루엣 지수만 보면 k=8이 최고점이지만, 군집이 과도하게 세분화되어
     해석이 어려워 점수 차이가 크지 않은 구간(6~9) 중 k=6을 최종 선택
4. MiniBatchKMeans로 최종 군집화 후, 클러스터별 TF-IDF 핵심 키워드 추출

[결과] Actor 0~5 (생활변화 가전교체 고민러 / 조건비교 가입러 / 이용중 AS·관리 민원러 /
       정수기렌탈 고민러 / 설치·이전비용러 / 계약해지·위약금 불만러)
"""

import re
import numpy as np
import pandas as pd
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42
MAX_DOC_CHARS = 1200
MIN_TOKEN_COUNT = 3
K_RANGE = range(4, 10)
FINAL_N_CLUSTERS = 6

# 군집화 결과가 제품/브랜드명만으로 갈리는 것을 막기 위한 불용어
ACTOR_STOPWORDS = set([
    "이", "그", "저", "것", "수", "등", "및", "더", "도", "을", "를", "가", "은", "는", "의", "에",
    "에서", "로", "으로", "와", "과", "만", "에게", "한", "하다", "있다", "되다", "않다", "없다",
    "같다", "보다", "많이", "정말", "너무", "아주", "매우", "조금", "좀", "바로", "그냥",
    "LG", "lg", "엘지", "삼성", "삼성전자", "전자", "제품", "모델", "가전", "구독", "렌탈", "렌트", "서비스",
    "혜택", "사은품", "지원금", "할인", "최대", "바로가기", "상담", "문의", "행사", "공식", "업체",
])

# 광고/이벤트성 글이 군집을 왜곡하지 않도록 사전 제거
PROMO_PATTERN = re.compile(
    r"혜택\s*바로가기|최대\s*혜택|신청\s*바로가기|사은품|지원금|현금\s*지원|"
    r"전국\s*시공|전국\s*설치|상담\s*문의|공식\s*인증\s*대리점|체험단|협찬|이벤트\s*일정|응모"
)


def build_actor_text(row) -> str:
    """제목 + 본문(또는 근거 발췌)을 합쳐 군집화 입력 텍스트를 구성한다."""
    title = str(row.get("title_clean", "") or "")
    body = str(row.get("evidence_excerpt", "") or row.get("text_clean", "") or "")
    combined = f"{title} {body[:MAX_DOC_CHARS]}"
    return re.sub(r"\s+", " ", combined).strip()


def tokenize_with_kiwi(kiwi: Kiwi, text: str) -> list[str]:
    """Kiwi로 형태소 분석 후, 불용어를 제거하고 의미 있는 토큰만 남긴다."""
    tokens = [t.form for t in kiwi.tokenize(text) if t.tag.startswith(("N", "V", "MAG"))]
    return [t for t in tokens if t not in ACTOR_STOPWORDS and len(t) >= 2]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """군집화 입력 텍스트 생성, 광고성 게시글 제거, 형태소 분석까지 수행한다."""
    df = df.copy()
    df["actor_text"] = df.apply(build_actor_text, axis=1)
    df = df[~df["actor_text"].apply(lambda x: bool(PROMO_PATTERN.search(x)))]

    kiwi = Kiwi()
    df["tagged_text"] = df["actor_text"].apply(lambda x: tokenize_with_kiwi(kiwi, x))
    df["token_count"] = df["tagged_text"].apply(len)
    df = df[df["token_count"] >= MIN_TOKEN_COUNT].reset_index(drop=True)
    return df


def vectorize(df: pd.DataFrame):
    """Kiwi 토큰을 공백으로 이어붙여 TF-IDF Dense 벡터로 변환한다."""
    tokenized_docs = df["tagged_text"].apply(lambda x: " ".join(x)).tolist()

    vectorizer = TfidfVectorizer(
        tokenizer=str.split, preprocessor=None, token_pattern=None, lowercase=False,
        ngram_range=(1, 1), min_df=4, max_df=0.40, max_features=300, sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(tokenized_docs)
    vectors = Normalizer(copy=False).fit_transform(matrix.toarray().astype("float32"))
    return np.ascontiguousarray(vectors, dtype="float32"), vectorizer


def search_best_k(vectors: np.ndarray, sample_size: int = 1000) -> pd.DataFrame:
    """k=4~9 구간에서 실루엣 지수를 비교해 최적 군집 수를 탐색한다."""
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(vectors), size=min(sample_size, len(vectors)), replace=False)
    sample = vectors[idx]

    results = []
    for k in K_RANGE:
        model = MiniBatchKMeans(
            n_clusters=k, random_state=RANDOM_STATE, init="random",
            n_init=1, max_iter=15, batch_size=4096, max_no_improvement=5, reassignment_ratio=0.01,
        )
        labels = model.fit_predict(sample)
        score = silhouette_score(sample, labels, sample_size=min(300, len(sample)), random_state=RANDOM_STATE)
        results.append({"n_cluster": k, "silhouette_score": score})

    return pd.DataFrame(results)


def cluster_final(vectors: np.ndarray, n_clusters: int = FINAL_N_CLUSTERS) -> np.ndarray:
    """실루엣 탐색 결과와 해석 가능성을 함께 고려해 선택한 k로 최종 군집화한다."""
    model = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=RANDOM_STATE, init="random",
        n_init=1, max_iter=10, batch_size=4096, max_no_improvement=3, reassignment_ratio=0.01,
    )
    return model.fit_predict(vectors)


def extract_cluster_keywords(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """클러스터별 토큰을 하나의 문서로 합쳐 TF-IDF 상위 키워드를 추출한다."""
    cluster_ids = sorted(df["actor_cluster"].unique())
    cluster_docs = [
        " ".join(" ".join(words) for words in df.loc[df["actor_cluster"] == cid, "tagged_text"])
        for cid in cluster_ids
    ]

    vectorizer = TfidfVectorizer(tokenizer=str.split, preprocessor=None, token_pattern=None, lowercase=False)
    tfidf = vectorizer.fit_transform(cluster_docs)
    feature_names = vectorizer.get_feature_names_out()

    rows = []
    for i, cid in enumerate(cluster_ids):
        scores = tfidf[i].toarray().flatten()
        top_idx = scores.argsort()[::-1][:top_n]
        for rank, idx in enumerate(top_idx, start=1):
            rows.append({"actor_cluster": cid, "rank": rank, "keyword": feature_names[idx], "tfidf": scores[idx]})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df_raw = pd.read_csv("actor_ready_corpus.csv")

    df = preprocess(df_raw)
    vectors, _ = vectorize(df)

    silhouette_df = search_best_k(vectors)
    print(silhouette_df)  # k=8이 최고점이나, 해석 가능성을 고려해 k=6 최종 선택

    df["actor_cluster"] = cluster_final(vectors, n_clusters=FINAL_N_CLUSTERS)
    print(df["actor_cluster"].value_counts().sort_index())

    keyword_df = extract_cluster_keywords(df)
    keyword_df.to_csv("actor_cluster_keywords.csv", index=False, encoding="utf-8-sig")

    df.to_pickle("actor_cluster_result.pkl")
    print("Actor 도출 완료: actor_cluster_result.pkl 저장")
