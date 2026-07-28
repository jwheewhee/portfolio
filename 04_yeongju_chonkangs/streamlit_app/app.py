"""
빈집 활용 촌캉스 사업 — Streamlit 데모
====================================================

2024 영주시 데이터 분석·활용 공모전 프로젝트를, 당시엔 구현하지 못했던 웹사이트 형태로
다시 만든 데모입니다. 원본 크롤링 데이터(CSV)는 보관되어 있지 않아, 실제로 도출했던 최종
결과와 분석 로직(하버사인 거리 계산, 관광지×맛집×빈집등급 스코어링)을 그대로 재현했습니다.

실행:
    pip install -r requirements.txt
    streamlit run app.py

배포: streamlit.io (Community Cloud) 에 GitHub 레포만 연결하면 무료로 배포 가능
"""

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="빈집 활용 촌캉스 사업", page_icon="🏡", layout="wide")

# -----------------------------------------------------------------------
# 실제 공모전에서 도출한 최종 결과 (발표자료 기준)
# -----------------------------------------------------------------------
FINAL_RESULTS = pd.DataFrame({
    "읍면동": ["문수면", "순흥면", "하망동", "부석면", "가흥2동"],
    "종합점수": [1.0000, 0.4712, 0.3603, 0.2840, 0.2833],
    "위도": [36.7817, 36.9130, 36.8065, 36.9427, 36.8446],   # 영주시 내 대략적인 공개 좌표
    "경도": [128.5686, 128.6741, 128.6270, 128.6997, 128.5972],
    "선정여부": ["최종 선정", "최종 선정", "인프라 접근성으로 제외", "최종 선정", "미선정"],
})


# -----------------------------------------------------------------------
# 실제 사용한 거리 계산 함수 (하버사인 공식 변형)
# -----------------------------------------------------------------------
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # 지구 반지름(km)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


EMPTY_HOUSE_SCORE = {"1등급": 50, "2등급": 40, "3등급": 25, "4등급": 10}


def compute_final_score(tourist_proximity: float, restaurant_proximity: float, house_grade: str) -> float:
    """
    실제 프로젝트의 스코어링 로직을 단순화해 재현:
    종합 점수 = 관광지 근접도 점수 × 맛집 근접도 점수 × 빈집 등급 점수(정규화)
    (0~1 슬라이더로 근접도를 직접 입력받는 데모 버전입니다.)
    """
    grade_score = EMPTY_HOUSE_SCORE[house_grade] / 50  # 1등급 기준 정규화
    return round(tourist_proximity * restaurant_proximity * grade_score, 4)


# -----------------------------------------------------------------------
# 화면 구성
# -----------------------------------------------------------------------
st.title("🏡 빈집 활용 촌캉스 사업")
st.caption("2024 영주시 데이터 분석·활용 공모전 · 팀장 · Python / Google Maps API / Folium")

st.markdown(
    """
    방치된 빈집을 촌캉스 숙박시설로 전환할 최적 입지를, 관광지·맛집·빈집 등급 데이터를
    종합 점수화해 도출한 프로젝트입니다. 아래에서 실제 분석 결과와 스코어링 로직을 직접
    체험해볼 수 있습니다.
    """
)

tab1, tab2, tab3 = st.tabs(["📊 최종 결과", "🗺️ 지도로 보기", "🧮 스코어링 계산기"])

# ── 탭 1: 최종 결과 ──────────────────────────────────────────────────
with tab1:
    st.subheader("종합 점수 기준 상위 5개 지역")
    st.dataframe(
        FINAL_RESULTS[["읍면동", "종합점수", "선정여부"]],
        use_container_width=True,
        hide_index=True,
    )
    st.bar_chart(FINAL_RESULTS.set_index("읍면동")["종합점수"])
    st.info("문수면·순흥면·부석면이 최종 선정되었고, 하망동은 종합 점수는 높았지만 "
            "주변 인프라 접근성 문제로 최종 후보에서 제외했습니다.")

# ── 탭 2: 지도 ───────────────────────────────────────────────────────
with tab2:
    st.subheader("최종 후보 지역 지도")
    m = folium.Map(location=[36.8065, 128.6270], zoom_start=11)
    for _, row in FINAL_RESULTS.iterrows():
        color = "green" if row["선정여부"] == "최종 선정" else "gray"
        folium.Marker(
            location=[row["위도"], row["경도"]],
            popup=f"{row['읍면동']} (종합점수 {row['종합점수']:.4f}) - {row['선정여부']}",
            icon=folium.Icon(color=color),
        ).add_to(m)
    st_folium(m, width=None, height=500)

# ── 탭 3: 스코어링 계산기 (인터랙티브 데모) ───────────────────────────
with tab3:
    st.subheader("스코어링 로직 직접 체험해보기")
    st.markdown("실제 분석에 쓴 `관광지 점수 × 맛집 점수 × 빈집 등급 점수` 로직을 슬라이더로 체험해볼 수 있습니다.")

    col1, col2, col3 = st.columns(3)
    with col1:
        tourist_score = st.slider("관광지 근접도 점수", 0.0, 1.0, 0.7, 0.05)
    with col2:
        restaurant_score = st.slider("맛집 근접도 점수", 0.0, 1.0, 0.6, 0.05)
    with col3:
        grade = st.selectbox("빈집 등급", list(EMPTY_HOUSE_SCORE.keys()))

    result = compute_final_score(tourist_score, restaurant_score, grade)
    st.metric("계산된 종합 점수", result)

    comparison = FINAL_RESULTS[["읍면동", "종합점수"]].copy()
    comparison.loc[len(comparison)] = ["내가 입력한 값", result]
    st.bar_chart(comparison.set_index("읍면동")["종합점수"])

    st.caption(
        "실제 분석에서는 하버사인 공식으로 계산한 실거리를 MinMaxScaler로 정규화해 "
        "근접도 점수를 산출했습니다. 이 데모에서는 그 정규화된 점수를 슬라이더로 직접 입력해볼 수 있게 했습니다."
    )

st.divider()
st.caption(
    "당시엔 이 분석 결과를 노션 정리로만 남겼는데, 이후 Streamlit을 배워 실제 웹사이트로 다시 구현했습니다."
)
