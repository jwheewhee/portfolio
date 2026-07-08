"""
Google Play Store LG ThinQ 앱 리뷰 크롤링 & 부정 키워드 분석
====================================================================

[목적]
"설치율은 높으나 실제 사용 시간과 활성화 기기 수가 적다"는
핵심 난제의 실질적인 원인을 정량적으로 파악하기 위해 진행했습니다.

[분석 방향]
- 변화가 컸던 5.1.x 버전 리뷰만 수집해, 업데이트 이후에도 사용자에게
  '편의'가 아닌 '복잡함'으로 인식되지 않았는지 검증
- '안 된다', '어렵다' 등 한국어의 다양한 형태소 변용을 모두 포함하는
  의미 중심 분석으로 실질적인 학습 결손 구간 식별

[수행 전략]
- 최근 1년치 리뷰 전수 수집으로 분석 신뢰도 확보
- 핵심 부정 키워드를 의미 그룹 단위 정규표현식으로 정의해 문법적 변용에 대응
- 키워드를 포함한 리뷰 비중(%)을 산출해 서비스 개선 우선순위 도출
- 별점 1~3점 리뷰만 워드클라우드로 시각화해 핵심 불만 요인을 직관적으로 파악
"""

import re
from datetime import datetime, timedelta

import pandas as pd
from google_play_scraper import Sort, reviews

APP_ID = "com.lgeha.nuts"  # LG ThinQ 앱 패키지명


# =========================================================
# 1. 리뷰 수집 (최근 1년)
# =========================================================
def crawl_reviews(app_id: str = APP_ID) -> pd.DataFrame:
    """
    최신순으로 리뷰를 페이지네이션 방식으로 수집한다.
    1년 이전 리뷰가 나오면 수집을 중단한다.
    """
    one_year_ago = datetime.now() - timedelta(days=365)
    all_reviews = []
    continuation_token = None
    batch_size = 200
    max_iterations = 1000  # 안전장치
    stop_collecting = False

    for _ in range(max_iterations):
        try:
            result, continuation_token = reviews(
                app_id,
                lang="ko",
                country="kr",
                sort=Sort.NEWEST,
                count=batch_size,
                continuation_token=continuation_token,
            )
            if not result:
                break

            for r in result:
                if r["at"] < one_year_ago:
                    stop_collecting = True
                    break
                all_reviews.append(r)

            if stop_collecting or continuation_token is None:
                break

        except Exception as e:
            print(f"오류 발생: {e}")
            break

    print(f"=== 총 수집 리뷰 수: {len(all_reviews):,}개 ===")

    df = pd.DataFrame(all_reviews)
    df = df.rename(
        columns={
            "reviewId": "리뷰ID",
            "userName": "사용자명",
            "content": "리뷰내용",
            "score": "평점",
            "thumbsUpCount": "추천수",
            "reviewCreatedVersion": "앱버전",
            "at": "작성일시",
            "replyContent": "개발자답변",
            "repliedAt": "답변일시",
        }
    )
    return df


# =========================================================
# 2. 버전 필터링 (5.1.x만 남기기)
# =========================================================
def filter_target_version(df: pd.DataFrame) -> pd.DataFrame:
    """앱 버전이 5.1.x(변화가 컸던 업데이트 구간)인 리뷰만 남긴다."""

    def is_target_version(version) -> bool:
        if pd.isna(version) or version is None:
            return False
        match = re.match(r"^5\.1\.(\d+)$", str(version).strip())
        return bool(match) and 0 <= int(match.group(1)) <= 99999

    df_filtered = df[df["앱버전"].apply(is_target_version)].copy()
    return df_filtered.reset_index(drop=True)


# =========================================================
# 3. 부정 키워드 빈도 분석
# =========================================================
# 의미 그룹 단위 정규표현식: 형태소 변용(안돼/안됨/안되네요 등)을 한 번에 포착
NEGATIVE_KEYWORDS = {
    "안된다": r"안\s?[되돼됨됩됐된]",
    "힘들다": r"힘[들드듦듭든]",
    "어렵다": r"어[렵려]",
    "불편": r"불편",
    "에러": r"에러|error",
    "멍청": r"멍청",
    "버그": r"버그|bug",
    "최악": r"최악",
    "짜증": r"짜증",
    "자꾸": r"자꾸",
    "좀": r"좀",
    "ㅡㅡ": r"ㅡㅡ|--",
    "답답": r"답답",
    "실망": r"실망",
    "후회": r"후회",
}


def analyze_negative_keywords(df_filtered: pd.DataFrame) -> pd.DataFrame:
    """키워드별 총 출현 횟수, 포함 리뷰 수, 리뷰 비율(%)을 계산한다."""
    review_texts = df_filtered["리뷰내용"].dropna().astype(str).tolist()
    all_text = " ".join(review_texts)

    results = []
    for label, pattern in NEGATIVE_KEYWORDS.items():
        total_count = len(re.findall(pattern, all_text, re.IGNORECASE))
        review_count = sum(1 for r in review_texts if re.search(pattern, r, re.IGNORECASE))
        review_ratio = (review_count / len(review_texts)) * 100 if review_texts else 0
        results.append(
            {
                "키워드": label,
                "총 출현 횟수": total_count,
                "포함 리뷰 수": review_count,
                "리뷰 비율(%)": round(review_ratio, 2),
            }
        )
    return pd.DataFrame(results)


def visualize_negative_keywords(df_keywords: pd.DataFrame, total_reviews: int, output_path: str = "lg_thinq_negative_keywords.png"):
    """부정 키워드 빈도를 가로 막대그래프로 시각화한다."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 9))
    df_plot = df_keywords.sort_values("포함 리뷰 수", ascending=True)

    bars = ax.barh(df_plot["키워드"], df_plot["포함 리뷰 수"], color="#E74C3C", alpha=0.8)
    for bar, count, ratio in zip(bars, df_plot["포함 리뷰 수"], df_plot["리뷰 비율(%)"]):
        width = bar.get_width()
        ax.text(
            width + max(df_plot["포함 리뷰 수"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(count)}개 ({ratio:.1f}%)",
            ha="left", va="center", fontsize=12,
        )

    ax.set_xlabel("포함 리뷰 수", fontsize=14)
    ax.set_ylabel("키워드", fontsize=14)
    ax.set_title(
        f"LG ThinQ 앱 부정 키워드 빈도수 (버전 5.1.x, 전체 {total_reviews:,}개 리뷰 기준)",
        fontsize=18, pad=20,
    )
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(df_plot["포함 리뷰 수"]) * 1.18)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print("차트 이미지 저장:", output_path)
    plt.show()


# =========================================================
# 4. 저평점 리뷰 워드클라우드 (별점 1~3점)
# =========================================================
def visualize_low_score_wordcloud(df_filtered: pd.DataFrame, output_path: str = "lg_thinq_negative_wordcloud.png"):
    """별점 1~3점 리뷰만 모아 워드클라우드로 핵심 불만 요인을 시각화한다."""
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    df_low_score = df_filtered[df_filtered["평점"].between(1, 3)].copy()
    text = " ".join(df_low_score["리뷰내용"].dropna().astype(str).tolist())

    print(f"별점 1~3점 리뷰 수: {len(df_low_score):,}개")

    font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows 기준. Mac은 AppleGothic 경로로 변경

    wordcloud = WordCloud(
        font_path=font_path,
        width=1600,
        height=900,
        background_color="white",
        max_words=200,
        colormap="Reds",  # 저평점이므로 빨간색 계열
        relative_scaling=0.5,
        min_font_size=10,
    ).generate(text)

    plt.figure(figsize=(16, 9))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("LG ThinQ 앱 리뷰 워드클라우드 (버전 5.1.x, 별점 1~3점)", fontsize=20, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()


# =========================================================
# 실행부
# =========================================================
if __name__ == "__main__":
    raw_df = crawl_reviews()
    filtered_df = filter_target_version(raw_df)

    keyword_df = analyze_negative_keywords(filtered_df)
    keyword_df.to_csv("lg_thinq_negative_keywords.csv", index=False, encoding="utf-8-sig")

    visualize_negative_keywords(keyword_df, total_reviews=len(filtered_df))
    visualize_low_score_wordcloud(filtered_df)
