"""
유튜브 ThinQ 사용법 영상 댓글 크롤링 & 유형 분류
====================================================

[목적]
LG전자가 제공하는 공식 가이드 영상이 실제 사용자의 니즈를
충분히 해결해주고 있는지 확인하기 위해 진행했습니다.

[분석 방향]
- 조회수 높은 ThinQ 사용법 영상 중심으로 선정
- 공식 답변으로 해소되지 않은 미해결 페인포인트를 집중 수집
- 긴 댓글까지 전수 수집해 사용자가 겪는 구체적인 맥락까지 파악

[수행 전략]
- Selenium으로 유튜브의 무한 스크롤 구조를 제어
- try-except 구조와 명시적 대기를 적용해 네트워크 환경이나
  DOM 구조 변화에도 중단 없는 크롤링 파이프라인을 구축
- 사전 정의한 핵심 키워드 딕셔너리 기반 if-elif 계층 구조로
  댓글을 5개 유형 + 기타로 분류

* LG전자 가전제품 사용법 영상 댓글도 동일한 방법으로 크롤링/시각화 진행
"""

import time

import pandas as pd
from selenium import webdriver as wb
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    ElementNotInteractableException,
)


# =========================================================
# 1. 유튜브 댓글 크롤링
# =========================================================
def crawl_youtube_comments(video_urls: list[str]) -> pd.DataFrame:
    """
    영상 URL 목록을 순회하며 댓글을 전수 수집한다.
    무한 스크롤을 끝까지 내린 뒤, '자세히보기' 버튼을 모두 클릭해
    접힌 댓글까지 펼쳐서 수집한다.
    """
    driver = wb.Chrome()
    all_comments = []

    for url in video_urls:
        driver.get(url)
        time.sleep(5)
        print(f"\n현재 영상 : {url}")

        # 댓글 영역 활성화
        driver.execute_script("window.scrollTo(0, 500)")
        time.sleep(3)

        # ---- 무한 스크롤 ----
        last_height = driver.execute_script("return document.documentElement.scrollHeight")
        while True:
            driver.execute_script(
                "window.scrollTo(0, document.documentElement.scrollHeight);"
            )
            time.sleep(2)
            new_height = driver.execute_script(
                "return document.documentElement.scrollHeight"
            )
            if new_height == last_height:
                break
            last_height = new_height
        print("스크롤 완료!")

        # ---- '자세히보기(더보기)' 버튼 모두 클릭 ----
        more_buttons = driver.find_elements(
            By.CSS_SELECTOR,
            "tp-yt-paper-button#more.more-button.style-scope.ytd-comment-view-model",
        )
        clicked = 0
        for btn in more_buttons:
            try:
                if btn.is_displayed():
                    # 스크롤 위치 문제를 피하기 위해 JS로 클릭
                    driver.execute_script("arguments[0].click();", btn)
                    clicked += 1
                    time.sleep(0.1)
            except (ElementClickInterceptedException, ElementNotInteractableException):
                continue
            except Exception:
                continue
        print(f"자세히보기 클릭 완료 : {clicked}개")
        time.sleep(1)  # 펼쳐질 시간 확보

        # ---- 댓글 수집 ----
        comments = driver.find_elements(By.ID, "content-text")
        print(f"댓글 수 : {len(comments)}")

        for c in comments:
            text = c.text.strip().replace("\n", " ")
            if text:
                all_comments.append(text)

    driver.quit()

    df = pd.DataFrame(all_comments, columns=["comment"])
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# =========================================================
# 2. 댓글 유형 분류
# =========================================================
def classify_comment(comment: str) -> str:
    """
    사전 정의한 키워드 딕셔너리를 기반으로 댓글을 5개 유형 + 기타로 분류한다.
    우선순위: 오류 문의 > 사용법 질문 > 호환성 문의 > 개선요구/불만 > 긍정후기 > 기타
    """
    comment = str(comment)

    error_keywords = [
        "안돼", "안되", "오류", "실패", "멈춰", "끊", "안뜨",
        "wifi", "와이파이", "비밀번호", "5g", "89%", "trs", "연결이 안", "고장",
    ]
    question_keywords = [
        "어떻게", "가능", "궁금", "설정", "사용", "방법",
        "되나요", "되나요?", "되나요ㅠ", "되나요ㅜ", "할수", "할 수", "원격", "예약",
    ]
    compatibility_keywords = [
        "아이폰", "ios", "갤럭시", "안드로이드", "태블릿", "탭", "모델", "g6", "g7", "지원", "호환",
    ]
    complaint_keywords = ["아쉽", "불편", "짜증", "시간낭비", "헬지", "개선", "필요", "화나", "왜", "문제"]
    positive_keywords = ["감사", "좋아요", "좋습", "잘", "추천", "편해", "최고", "배우고", "믿고", "친절"]

    if any(k in comment for k in error_keywords):
        return "연결/설정 오류 문의"
    elif any(k in comment for k in question_keywords):
        return "기능 사용법 질문"
    elif any(k in comment for k in compatibility_keywords):
        return "호환성 문의"
    elif any(k in comment for k in complaint_keywords):
        return "개선 요구/불만"
    elif any(k in comment for k in positive_keywords):
        return "긍정 후기"
    else:
        return "기타"


# =========================================================
# 3. 파이차트 시각화
# =========================================================
def visualize_comment_types(df: pd.DataFrame, output_path: str = "lg_thinq_comment_type_pie.png"):
    """댓글 유형별 비율을 파이차트로 시각화한다."""
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"  # Windows 기준. Mac은 AppleGothic으로 변경
    plt.rcParams["axes.unicode_minus"] = False

    type_counts = df["comment_type"].value_counts()
    type_ratio = round((type_counts / len(df)) * 100, 1)

    print("\n유형별 댓글 수\n", type_counts)
    print("\n유형별 비율(%)\n", type_ratio)

    colors = ["#E74C3C", "#F1948A", "#F5B7B1", "#C0392B", "#E6B0AA", "#CD6155"]

    plt.figure(figsize=(16, 9))
    wedges, _, autotexts = plt.pie(
        type_ratio,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        colors=colors[: len(type_ratio)],
        textprops={"fontsize": 11},
    )
    for autotext in autotexts:
        autotext.set_fontsize(13)
        autotext.set_color("white")

    plt.legend(
        wedges, type_ratio.index, title="댓글 유형",
        loc="center left", bbox_to_anchor=(1, 0.5),
        fontsize=12, title_fontsize=13,
    )
    plt.title("LG ThinQ 사용법 유튜브 댓글 유형 비율", fontsize=20, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print("이미지 저장 완료:", output_path)
    plt.show()


# =========================================================
# 실행부
# =========================================================
if __name__ == "__main__":
    video_urls = [
        "https://www.youtube.com/watch?v=VIDEO_ID_1",
        "https://www.youtube.com/watch?v=VIDEO_ID_2",
        # ... 조회수 높은 ThinQ 사용법 가이드 영상 URL 목록
    ]

    df = crawl_youtube_comments(video_urls)
    df.to_excel("lg_youtube_comments.xlsx", index=False)
    print(f"\n엑셀 저장 완료! 총 {len(df)}개 댓글")

    df["comment_type"] = df["comment"].apply(classify_comment)
    visualize_comment_types(df)
