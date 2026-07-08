"""
82쿡 카페 게시글 제목 크롤링 & 키워드 워드클라우드
====================================================

[목적]
5060 세대가 주로 활동하는 커뮤니티에서 어떤 주제와 관심사를 중심으로
소통하고 있는지 파악하기 위해 진행했습니다.

[분석 방향]
- 5060 세대의 주 관심사가 반영되는 핵심 게시판을 선정해 데이터 수집
- 2024년 이후의 최신 게시물만 수집해 현재 시점의 관심사 파악
- 분석 목적을 흐릴 수 있는 정치 키워드 및 불용어 제거

[수행 전략]
- 정규표현식과 치환 로직으로 데이터 노이즈 일괄 제거
- KoNLPy(Okt)를 활용해 문장에서 의미 있는 2글자 이상의 핵심 키워드 정제

* '우아한 갱년기' 카페도 동일한 방법으로 크롤링/시각화 진행 (게시판 ID만 교체)
"""

import re
import time
from collections import Counter

import pandas as pd
from selenium import webdriver as wb
from selenium.webdriver.common.by import By
from tqdm import tqdm


# =========================================================
# 1. 게시글 제목 크롤링
# =========================================================
def crawl_titles(crawl_config: list[dict]) -> pd.DataFrame:
    """
    82쿡 게시판별로 지정한 페이지 범위를 순회하며 게시글 제목/작성일을 수집한다.
    - 2024년 이전 게시물만 있는 페이지를 만나면 해당 게시판 수집을 중단한다.

    Args:
        crawl_config: [{"bn": 게시판번호, "start": 시작페이지, "end": 끝페이지}, ...]
    """
    driver = wb.Chrome()
    all_titles = []

    for cfg in crawl_config:
        bn = cfg["bn"]
        stop_flag = False

        for page in tqdm(range(cfg["start"], cfg["end"]), desc=f"bn={bn}"):
            if stop_flag:
                break

            driver.get(f"https://www.82cook.com/entiz/enti.php?bn={bn}&page={page}")
            time.sleep(1)

            # 행(tr) 단위로 가져와서 제목/작성일을 같은 행에서 매칭
            rows = driver.find_elements(By.CSS_SELECTOR, "table tr")
            page_has_2024_or_later = False
            valid_row_count = 0

            for row in rows:
                title_elems = row.find_elements(By.CLASS_NAME, "title")
                date_elems = row.find_elements(By.CLASS_NAME, "regdate")

                if not title_elems or not date_elems:
                    continue

                title = title_elems[0].text.strip()
                date = date_elems[0].text.strip()

                if not title or not date or title == "제목":
                    continue

                # 날짜 파싱: "18:57:07"(오늘 글) / "2024.11.07" / "2026-05-12" 등
                if re.match(r"^\d{1,2}:\d{2}", date):
                    year = 2026  # 시간만 표시되면 당일 작성 글로 처리
                else:
                    year_match = re.search(r"(20\d{2})", date)
                    if not year_match:
                        continue
                    year = int(year_match.group(1))

                valid_row_count += 1

                if year >= 2024:
                    all_titles.append(title)
                    page_has_2024_or_later = True

            # 게시물이 아예 없거나, 2024년 이후 게시물이 더 이상 없으면 해당 게시판 종료
            if valid_row_count == 0 or not page_has_2024_or_later:
                stop_flag = True

    driver.quit()
    return pd.DataFrame({"title": all_titles})


# =========================================================
# 2. 텍스트 정제
# =========================================================
def clean_titles(df: pd.DataFrame) -> pd.DataFrame:
    """공지성 문구, 정치 관련 키워드를 제거하고 한글/영문만 남긴다."""
    remove_sentences = [
        "비밀번호를 변경해주세요",
        "회원님들께 당부의 말씀 올립니다",
        "뉴스기사 등 무단 게재 관련 공지입니다",
        "자유게시판은",
        "모자모숨",
    ]
    remove_keywords = ["이재명", "주식", "한동훈", "민주당", "트럼프", "조국", "김건희", "윤석열"]

    def _clean(text: str) -> str:
        for s in remove_sentences:
            text = text.replace(s, "")
        for k in remove_keywords:
            text = text.replace(k, "")
        text = re.sub(r"[^가-힣a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    df["clean_title"] = df["title"].apply(_clean)
    return df[df["clean_title"] != ""].reset_index(drop=True)


# =========================================================
# 3. 형태소 분석 + 워드클라우드 시각화
# =========================================================
def visualize_wordcloud(df: pd.DataFrame, output_path: str = "82cook_title.png"):
    """
    KoNLPy(Okt)로 형태소를 분석해 불용어를 제거한 뒤,
    출현 빈도 기반 워드클라우드를 생성한다.
    """
    from konlpy.tag import Okt
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    okt = Okt()

    # 의미 없는 조사/대명사/추임새성 단어는 불용어로 제거
    stopwords = [
        "있는", "없는", "하는", "되는", "같은", "저는", "제가", "그냥", "그게",
        "그거", "이거", "저거", "여기", "저기", "거기", "그런", "이런", "저런",
        "근데", "그리고", "하지만", "그래서", "너무", "진짜", "정말", "아주",
        "많이", "조금", "약간", "계속", "자꾸", "항상", "가끔", "때문", "동안",
        "경우", "정도", "생각", "느낌", "사람", "저희", "자기", "자신",
        "말씀", "얘기", "이야기", "내용", "관련", "에서", "feat",
        "가능", "어디서", "무엇", "누구", "언제", "어떤", "에게", "으로",
        "이번", "다음", "지난", "요즘", "최근", "예전", "이전", "이후", "오늘",
        "어제", "내일", "아침", "저녁", "하나", "둘", "셋", "여러", "모두",
        "전부", "그것", "이것", "저것", "여러분", "님들", "여러가지", "하시",
        "드시", "보시", "되시", "있어요", "없어요", "같아요", "해요", "돼요",
        "있고", "없고", "하고", "되고", "같고", "입니다", "습니다", "합니다",
        "인데", "한데", "는데", "아니", "맞아", "몰라", "글쎄",
    ]

    text_all = " ".join(df["clean_title"].astype(str).tolist())
    words = okt.morphs(text_all)
    filtered_words = [w for w in words if w not in stopwords and len(w) >= 2]

    counts = Counter(filtered_words)
    print("상위 30개 단어:", counts.most_common(30))

    font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows 기준. Mac은 AppleGothic 경로로 변경

    wc = WordCloud(
        font_path=font_path,
        width=1000,
        height=700,
        background_color="white",
        colormap="Reds",
    ).generate_from_frequencies(counts)

    plt.figure(figsize=(16, 9))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print("이미지 저장 완료:", output_path)
    plt.show()


# =========================================================
# 실행부
# =========================================================
if __name__ == "__main__":
    # bn: 게시판 번호 (엔지토크/키친토크/요리물음표/자유게시판/이런저런 질문)
    crawl_config = [
        {"bn": 15, "start": 1, "end": 9999},
        {"bn": 6, "start": 1, "end": 9999},
        {"bn": 8, "start": 1, "end": 9999},
        {"bn": 16, "start": 1, "end": 9999},
    ]

    raw_df = crawl_titles(crawl_config)
    clean_df = clean_titles(raw_df)
    clean_df[["clean_title"]].to_excel("82cook_clean_titles.xlsx", index=False)
    print(f"엑셀 저장 완료! 총 {len(clean_df)}개 제목 수집")

    visualize_wordcloud(clean_df)
