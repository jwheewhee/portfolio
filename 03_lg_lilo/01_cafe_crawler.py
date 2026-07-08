"""
네이버 카페(directwedding) 가전구독 게시글·댓글 크롤러
====================================================

[목적]
신혼가전 준비 과정에서 가전 구독을 고민하는 실제 목소리를 수집하기 위해,
결혼 준비 커뮤니티인 directwedding 카페의 게시글과 댓글을 크롤링합니다.
(이 외 디시인사이드, 블라인드, 블로그 등은 팀원이 각각 담당 채널을 크롤링했습니다.)

[수집 방식]
- '가전 구독', 'TV 구독', '정수기 구독' 등 18개 키워드로 카페 내부 검색
- 키워드당 최대 20페이지, 광고/특정 작성자 게시글 제외, 중복 게시글 제거
- 목록 수집 후 게시글 상세 페이지에 재접속해 본문 + 댓글까지 수집

[실행]
pip install selenium beautifulsoup4 pandas openpyxl tqdm
python 01_cafe_crawler.py
(최초 실행 시 브라우저에서 네이버 로그인 필요)
"""

import time
import csv
import json
import re
import os
import pandas as pd
from datetime import datetime
from urllib.parse import quote
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup

# ── 설정 ─────────────────────────────────────────────────────────
KEYWORDS = [
    "가전 구독", "TV 구독", "티비 구독", "구독 고민", "정수기 구독",
    "냉장고 구독", "세탁기 구독", "에어컨 구독", "건조기 구독", "가전 렌탈",
    "구독 vs 구매", "구독이나 구매", "가전 월정액", "구독 가전",
    "렌탈 vs 구매", "렌탈 고민", "구독 후기", "렌탈 후기",
]

EXCLUDE_AUTHORS = {"웨딩컨시어지", "긍정긍정", "가전컨시어지"}  # 광고성 계정 제외
PAGES_PER_KEYWORD = 20
COLLECT_DETAIL = True
DELAY = 2.5
DETAIL_DELAY = 2.5
CAFE_ID = "25228091"
CAFE_URL = "https://cafe.naver.com/directwedding"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"cafe_crawl_{timestamp}"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"cafe_data_{timestamp}.csv")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"cafe_data_{timestamp}.json")
OUTPUT_XLSX = os.path.join(OUTPUT_DIR, f"cafe_data_{timestamp}.xlsx")

SEARCH_URL = (
    f"https://cafe.naver.com/f-e/cafes/{CAFE_ID}/menus/0"
    "?viewType=L&ta=ARTICLE_COMMENT&page={page}&q={q}"
)


# ── 드라이버 및 로그인 ──────────────────────────────────────────
def make_driver():
    opts = Options()
    opts.add_argument("--window-size=1400,900")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(options=opts)
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


def naver_login(driver):
    """네이버 카페는 로그인 세션이 있어야 검색·본문 접근이 가능해 수동 로그인을 거친다."""
    driver.get("https://nid.naver.com/nidlogin.login")
    print("브라우저에서 네이버 로그인 완료 후 Enter를 눌러주세요.")
    input("▶ Enter: ")
    time.sleep(2)
    driver.get(CAFE_URL)
    time.sleep(3)


def build_url(keyword, page):
    return SEARCH_URL.format(q=quote(keyword), page=page)


def _extract_article_id(url):
    m = re.search(r"/articles/(\d+)", url)
    return m.group(1) if m else url


def _try_switch_iframe(driver):
    try:
        WebDriverWait(driver, 5).until(
            EC.frame_to_be_available_and_switch_to_it((By.ID, "cafe_main"))
        )
        return True
    except TimeoutException:
        return False


# ── 게시글 목록 수집 ─────────────────────────────────────────────
def get_search_list(driver, keyword):
    """키워드로 카페 내부 검색을 수행하고, 결과가 없는 페이지를 만나면 종료한다."""
    posts = []

    for page in range(1, PAGES_PER_KEYWORD + 1):
        url = build_url(keyword, page)
        driver.get(url)
        time.sleep(DELAY)

        in_iframe = _try_switch_iframe(driver)
        time.sleep(1.0)

        # 로딩 실패 감지 시 카페 메인을 거쳐 세션을 갱신하고 재시도
        if len(driver.page_source) < 50000:
            if in_iframe:
                driver.switch_to.default_content()
            driver.get(CAFE_URL)
            time.sleep(3)
            driver.get(url)
            time.sleep(DELAY + 1)
            in_iframe = _try_switch_iframe(driver)
            time.sleep(1.0)

        soup = BeautifulSoup(driver.page_source, "html.parser")

        if any(msg in driver.page_source for msg in ["검색 결과가 없습니다", "일치하는 게시물이 없습니다"]):
            if in_iframe:
                driver.switch_to.default_content()
            break

        rows = soup.select("tr")
        page_count = 0

        for row in rows:
            title_tag = row.select_one("a.article")
            if not title_tag:
                continue

            title = title_tag.get_text(strip=True)
            href = title_tag.get("href", "")
            if not title or not href:
                continue
            if not href.startswith("http"):
                href = "https://cafe.naver.com" + href

            if re.match(r"^\[\d+\]$", title):  # 댓글수 표시 링크 제외
                continue

            author_tag = row.select_one(".nickname")
            author = author_tag.get_text(strip=True) if author_tag else ""
            if author in EXCLUDE_AUTHORS:
                continue

            date_td = row.select_one("td.td_normal")
            date_str = date_td.get_text(strip=True) if date_td else ""

            posts.append({
                "keyword": keyword,
                "article_id": _extract_article_id(href),
                "title": title,
                "author": author,
                "date": date_str,
                "url": href,
                "content": "",
                "comments": [],
            })
            page_count += 1

        if in_iframe:
            driver.switch_to.default_content()

        if page_count == 0:
            break

        time.sleep(1.0)

    return posts


# ── 게시글 본문·댓글 수집 ────────────────────────────────────────
def _load_all_comments(driver):
    for _ in range(30):
        try:
            btn = driver.find_element(By.CSS_SELECTOR, ".comment_more_wrap button, .CommentMore button")
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(1)
        except NoSuchElementException:
            break


def get_post_detail(driver, post):
    """게시글 원본 페이지에 재접속해 본문과 전체 댓글을 수집한다."""
    try:
        driver.get(post["url"])
    except Exception:
        post["content"] = "[페이지 열기 실패]"
        return post
    time.sleep(DETAIL_DELAY)

    in_iframe = _try_switch_iframe(driver)
    time.sleep(1.0)

    content = ""
    for sel in [".article_viewer", ".se-main-container", "#postListBody", ".ContentRenderer", "#postContent"]:
        try:
            el = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, sel)))
            content = el.text.strip()
            if content:
                break
        except TimeoutException:
            continue
    post["content"] = content or "[본문 없음]"

    _load_all_comments(driver)

    comments = []
    for csel in [".text_comment", ".CommentText", ".comment_text_box span"]:
        cels = driver.find_elements(By.CSS_SELECTOR, csel)
        if cels:
            comments = [{"text": c.text.strip()} for c in cels if c.text.strip()]
            break
    post["comments"] = comments

    if in_iframe:
        driver.switch_to.default_content()

    return post


# ── 저장 ─────────────────────────────────────────────────────────
def save_results(posts):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)

    fields = ["keyword", "article_id", "title", "author", "date", "url", "content", "comments"]
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in posts:
            row = {k: p.get(k, "") for k in fields}
            row["comments"] = json.dumps(p.get("comments", []), ensure_ascii=False)
            w.writerow(row)

    # LDA 등 후속 텍스트 분석에 바로 쓸 수 있도록 본문+댓글 결합 시트로도 저장
    rows_combined = []
    for p in posts:
        cmt_texts = [c.get("text", "") for c in p.get("comments", []) if c.get("text")]
        rows_combined.append({
            "검색키워드": p.get("keyword", ""),
            "article_id": p.get("article_id", ""),
            "제목": p.get("title", ""),
            "작성자": p.get("author", ""),
            "작성일": p.get("date", ""),
            "본문": p.get("content", ""),
            "댓글": "\n---\n".join(cmt_texts),
            "댓글수": len(cmt_texts),
            "URL": p.get("url", ""),
        })

    df = pd.DataFrame(rows_combined)
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="본문+댓글결합", index=False)
        (df.groupby("검색키워드").agg(게시글수=("article_id", "nunique"))
            .reset_index().sort_values("게시글수", ascending=False)
            .to_excel(writer, sheet_name="키워드별요약", index=False))


# ── 메인 ─────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    driver = make_driver()
    all_posts = {}

    try:
        naver_login(driver)

        for kw in KEYWORDS:
            for p in get_search_list(driver, kw):
                aid = p["article_id"]
                if aid not in all_posts:
                    all_posts[aid] = p
                elif kw not in all_posts[aid]["keyword"]:
                    all_posts[aid]["keyword"] += f", {kw}"

        posts = list(all_posts.values())
        print(f"중복 제거 후 총 {len(posts)}개 게시글")

        if COLLECT_DETAIL:
            for i, post in enumerate(posts, 1):
                posts[i - 1] = get_post_detail(driver, post)
                if i % 50 == 0:
                    save_results(posts)  # 중간 저장

        save_results(posts)
        print(f"완료! 총 {len(posts)}개 저장")

    except KeyboardInterrupt:
        if all_posts:
            save_results(list(all_posts.values()))
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
