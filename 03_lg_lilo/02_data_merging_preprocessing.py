"""
전체 수집 데이터 통합 및 1차 전처리
====================================================

[목적]
팀원 각자 담당 채널(카페, 유튜브, 블라인드, 디시인사이드, 블로그,
레몬테라스, 인테리어카페 등)에서 수집한 데이터를 하나의 코퍼스로 통합해,
이후 Actor/Action 분석에 사용할 수 있는 형태로 정리합니다.

[처리 내용]
- 채널별 전처리 완료 파일(xlsx)을 모두 불러와 하나로 병합
- 결측치/공백 정리, 빈 텍스트 제거
- 제목+텍스트 기준 완전 중복 게시글 제거
- 출처(channel) 컬럼 추가로 이후 채널별 분석 가능하도록 정리
"""

import pandas as pd

BASE_PATH = "./data/preprocessed/"  # 채널별 전처리 완료 파일이 위치한 경로

SOURCE_FILES = {
    "directwedding": "directwedding_preprocessed.xlsx",
    "youtube": "youtube_preprocessed.xlsx",
    "blind": "blind_preprocessed.xlsx",
    "dcinside": "dcinside_preprocessed.xlsx",
    "blog": "blog_preprocessed.xlsx",
    "lemonterrace": "lemonterrace_preprocessed.xlsx",
    "interiorcafe": "interiorcafe_preprocessed.xlsx",
    "cafe": "allcafe_preprocessed.xlsx",
}


def load_and_merge(base_path: str = BASE_PATH) -> pd.DataFrame:
    """채널별 전처리 완료 파일을 불러와 출처 컬럼과 함께 하나로 병합한다."""
    dfs = []
    for source_name, filename in SOURCE_FILES.items():
        df = pd.read_excel(base_path + filename)
        df["출처"] = source_name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def clean_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """결측치/공백을 정리하고, 빈 텍스트 및 완전 중복 게시글을 제거한다."""
    for col in ["제목", "텍스트"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df = df[df["텍스트"] != ""].reset_index(drop=True)
    df = df.drop_duplicates(subset=["제목", "텍스트"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    final_corpus = load_and_merge()
    final_corpus = clean_corpus(final_corpus)

    print("최종 데이터 크기:", final_corpus.shape)
    print(final_corpus["출처"].value_counts())

    final_corpus.to_excel(BASE_PATH + "final_corpus.xlsx", index=False, engine="openpyxl")
    print("final_corpus.xlsx 저장 완료")
