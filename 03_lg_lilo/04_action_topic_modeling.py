"""
Action 도출 — Actor별 LDA 토픽 모델링
====================================================

[목적]
Actor(고객 유형)별로 묶인 게시글 안에서 실제로 어떤 세부 행동(Action)을
하고 있는지 LDA 토픽 모델링으로 도출합니다.

[방법 요약]
1. Actor 클러스터 하나씩 필터링해 개별적으로 LDA 적용
   (Actor마다 관심사가 달라 토픽 수를 공통으로 고정하지 않고 개별 산정)
2. 단어 사전(Dictionary) 생성 → BoW corpus 변환
3. Perplexity(낮을수록 좋음) / Coherence(높을수록 좋음)를 비교해
   Actor별로 적합한 토픽 수 선정
4. 최종 LDA 모델로 문서별 대표 토픽(Action) 라벨링
5. pyLDAvis 시각화 토픽 번호와 LDA 모델 토픽 번호가 다르게 정렬될 수 있어
   두 번호 체계를 수동으로 대응시켜 최종 Action 번호를 확정

[LDA 파라미터] passes=20, iterations=50 (커뮤니티 단문 텍스트 특성상 넉넉하게 설정)
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import gensim
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel


def build_corpus(df_actor: pd.DataFrame):
    """Actor 클러스터 하나의 토큰 리스트로 사전(Dictionary)과 BoW corpus를 생성한다."""
    dictionary = Dictionary(df_actor["tagged_text"])
    corpus = [dictionary.doc2bow(doc) for doc in df_actor["tagged_text"]]
    return dictionary, corpus


def search_topic_num(corpus, dictionary, texts, topic_range=range(2, 10), passes=20, iterations=50):
    """토픽 수 2~9개 구간에서 Coherence/Perplexity를 비교해 적합한 토픽 수를 탐색한다."""
    results = []
    for n in tqdm(topic_range, desc="토픽 수 탐색"):
        model = gensim.models.ldamodel.LdaModel(
            corpus, num_topics=n, id2word=dictionary, passes=passes, iterations=iterations
        )
        coherence = CoherenceModel(model=model, texts=texts, dictionary=dictionary, topn=5).get_coherence()
        perplexity = np.exp(-model.log_perplexity(corpus))
        results.append({"num_topics": n, "coherence": coherence, "perplexity": perplexity})

    return pd.DataFrame(results)


def fit_lda(corpus, dictionary, num_topics: int, passes=20, iterations=50):
    """선정한 토픽 수로 최종 LDA 모델을 학습한다."""
    return gensim.models.ldamodel.LdaModel(
        corpus, num_topics=num_topics, id2word=dictionary, passes=passes, iterations=iterations
    )


def assign_dominant_topic(model, corpus) -> list[int]:
    """문서별로 확률이 가장 높은 토픽을 대표 Action으로 라벨링한다."""
    result = []
    for doc_topics in model.get_document_topics(corpus):
        topics = [t for t, _ in doc_topics]
        probs = [p for _, p in doc_topics]
        result.append(topics[int(np.argmax(probs))])
    return result


def remap_action_numbers(action_labels: list[int], topic_number_map: dict[int, int]) -> list[int]:
    """
    pyLDAvis 표시 번호(토픽 크기 기준 1,2,3...)와 LDA 모델 내부 토픽 번호(0,1,2...)가
    다르게 정렬되는 경우가 있어, 발표/보고서에 쓸 최종 Action 번호로 수동 매핑한다.

    예: topic_number_map = {0: 1, 2: 2, 1: 3}
    """
    return [topic_number_map[label] for label in action_labels]


if __name__ == "__main__":
    df = pd.read_pickle("actor_cluster_result.pkl")

    all_results = []
    for actor_id in sorted(df["actor_cluster"].unique()):
        df_actor = df[df["actor_cluster"] == actor_id].copy()

        dictionary, corpus = build_corpus(df_actor)

        search_df = search_topic_num(corpus, dictionary, df_actor["tagged_text"])
        print(f"\nActor {actor_id} 토픽 수 탐색 결과:\n{search_df}")

        # 탐색 결과를 보고 Actor별로 가장 해석 가능한 토픽 수를 선택 (예: 3~4개)
        best_num_topics = int(search_df.loc[search_df["coherence"].idxmax(), "num_topics"])

        lda_model = fit_lda(corpus, dictionary, num_topics=best_num_topics)
        df_actor["action_cluster"] = assign_dominant_topic(lda_model, corpus)

        # pyLDAvis 번호와 대응 확인 후 필요 시 remap_action_numbers()로 최종 번호 확정
        all_results.append(df_actor[["review", "tagged_text", "actor_cluster", "action_cluster"]])

    action_result = pd.concat(all_results, ignore_index=True)
    action_result.to_pickle("action_cluster_result.pkl")
    print("Action 도출 완료: action_cluster_result.pkl 저장")
