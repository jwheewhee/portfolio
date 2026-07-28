"""
빈집 활용 촌캉스 최적 입지 선정
====================================================

[목적]
2024 영주시 데이터 분석·활용 공모전 출품작으로, 방치된 빈집을 촌캉스 숙박시설로
전환하기에 가장 적합한 지역을 데이터 기반으로 선정합니다.

[핵심 제약과 해결 방법]
빈집의 정확한 주소는 개인정보 문제로 제공받을 수 없었고, 읍면동 단위 등급별
(1~4등급) 빈집 개수만 주어졌습니다. 이를 해결하기 위해 19개 읍면동 각각의
행정복지센터 위치를 해당 지역 빈집의 대표 좌표로 삼는 방식을 사용했습니다.

[분석 흐름]
1. 관광지·맛집·빈집·버스정류장·행정복지센터 데이터 전처리
2. Google Maps Geocoding API로 주소를 위경도 좌표로 변환
3. 하버사인 공식으로 행정복지센터-관광지/맛집 간 실거리 계산 및 점수화
4. 빈집 등급별 점수 산정(MinMaxScaler 정규화)
5. 관광지 점수 × 맛집 점수 × 빈집 등급 점수를 곱해 종합 점수 산출, 최종 입지 선정
6. 최종 후보 읍면동 내에서 실제 인기 관광지·맛집 좌표 평균으로 세부 추천 지점까지 도출
"""

import numpy as np
import pandas as pd
import googlemaps
import folium
from sklearn.preprocessing import MinMaxScaler

GOOGLE_MAPS_API_KEY = "YOUR_API_KEY"  # 발급받은 API 키로 교체

EMPTY_HOUSE_GRADE_SCORE = {"1등급": 50, "2등급": 40, "3등급": 25, "4등급": 10}


# =========================================================
# 1. 데이터 불러오기 및 전처리
# =========================================================
def load_raw_data():
    tourist = pd.read_csv("center_tourist_spots.csv", encoding="cp949")
    restaurants = pd.read_csv("local_restaurants.csv", encoding="cp949")
    empty_houses = pd.read_csv("empty_houses.csv")
    community_centers = pd.read_csv("community_centers.csv", encoding="cp949")
    return tourist, restaurants, empty_houses, community_centers


def clean_data(tourist: pd.DataFrame, restaurants: pd.DataFrame, community_centers: pd.DataFrame):
    """숙박시설(경쟁시설)은 관광지 목록에서 제외하고, 분석에 불필요한 컬럼을 정리한다."""
    tourist = tourist[tourist["중심카테고리 명_대"] != "숙박"].copy()
    tourist = tourist.drop(["중심카테고리 명_대", "분류"], axis=1, errors="ignore")
    tourist = tourist.rename(columns={"중심 POI X 좌표": "경도", "중심 POI Y 좌표": "위도"})

    restaurants = restaurants.drop(["분류"], axis=1, errors="ignore")

    community_centers = community_centers.drop(
        ["대표전화번호", "팩스번호", "데이터기준일자"], axis=1, errors="ignore"
    )
    community_centers = community_centers[community_centers["기관명"] != "영주시청"]

    return tourist, restaurants, community_centers


def clean_empty_houses(df_empty_houses: pd.DataFrame) -> pd.DataFrame:
    """헤더 정리, 총계 행 제거, 결측값('-')을 0으로 대체 후 정수형으로 변환한다."""
    df = df_empty_houses.copy()
    df.columns = df.iloc[0]
    df = df.drop(df.index[0]).reset_index(drop=True)
    df = df.drop(columns=["계"], errors="ignore")
    df = df[df["읍면동"] != "총계"]  # 총계 행 제거

    df.replace("-", 0, inplace=True)
    grade_cols = ["1등급", "2등급", "3등급", "4등급"]
    df[grade_cols] = df[grade_cols].astype(int)
    return df


# =========================================================
# 2. 지오코딩 (주소 → 위경도)
# =========================================================
def geocode_address(gmaps_client: googlemaps.Client, address: str):
    try:
        result = gmaps_client.geocode(address)
        if result:
            loc = result[0]["geometry"]["location"]
            return loc["lat"], loc["lng"]
    except Exception as e:
        print("Geocoding 오류:", e)
    return None, None


def add_coordinates(df: pd.DataFrame, address_column: str, gmaps_client: googlemaps.Client) -> pd.DataFrame:
    df = df.copy()
    df["위도"], df["경도"] = None, None
    for idx, addr in df[address_column].items():
        lat, lng = geocode_address(gmaps_client, addr)
        df.at[idx, "위도"], df.at[idx, "경도"] = lat, lng
    return df


# =========================================================
# 3. 거리 계산 (하버사인 공식 변형)
# =========================================================
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 사이의 실제 거리(km)를 구면 삼각법(하버사인 공식)으로 계산한다."""
    R = 6371  # 지구 반지름(km)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def calculate_proximity_score(center_row: pd.Series, target_df: pd.DataFrame) -> float:
    """
    행정복지센터 기준, 가장 가까운 대상 지점의 정규화 점수를 거리 가중치와 함께 반환한다.
    거리가 가까울수록(min_distance가 작을수록) 점수가 높아지도록 설계했다.
    """
    lat1, lon1 = center_row["위도"], center_row["경도"]
    max_distance, min_distance, min_normal = 0, float("inf"), 0

    for _, row in target_df.iterrows():
        distance = calculate_distance(lat1, lon1, row["위도"], row["경도"])
        if distance < min_distance:
            min_distance = distance
            min_normal = row["정규"]
        if distance > max_distance:
            max_distance = distance

    # 분모가 0이 되는 것을 방지하기 위해 +1
    return (max_distance - min_distance) / (max_distance - min_distance + 1) * min_normal


# =========================================================
# 4. 점수 산출
# =========================================================
def score_tourist_and_restaurants(tourist: pd.DataFrame, restaurants: pd.DataFrame):
    """관광지는 순위(인기도), 맛집은 방문객 수 기준으로 정규화 점수를 부여한다."""
    scaler = MinMaxScaler()

    # 맛집: 총 방문객 수 기준 정규화
    restaurants = restaurants.copy()
    restaurants["정규"] = scaler.fit_transform(restaurants[["총방문객수"]])

    # 관광지: 순위가 높을수록(숫자가 작을수록) 높은 점수를 받도록 순위를 뒤집어 정규화
    tourist = tourist.copy()
    rank_length = len(tourist["순위"])
    tourist["순위"] = list(range(rank_length, 0, -1))
    tourist["정규"] = scaler.fit_transform(tourist[["순위"]])

    return tourist, restaurants


def score_empty_houses(df_empty_houses: pd.DataFrame, grades: list[str] = None) -> pd.DataFrame:
    """
    빈집 등급별 가중치를 곱해 총점을 구하고 정규화한다.
    grades를 ["1등급", "2등급"]으로 좁히면 숙박 전환이 용이한 빈집만 대상으로 분석할 수 있다.
    """
    grades = grades or list(EMPTY_HOUSE_GRADE_SCORE.keys())
    df = df_empty_houses.copy()

    df["총점"] = df[grades].apply(
        lambda row: sum(row[g] * EMPTY_HOUSE_GRADE_SCORE[g] for g in grades), axis=1
    )

    scaler = MinMaxScaler()
    df["E_S"] = scaler.fit_transform(df[["총점"]])
    df.loc[df["총점"] == 0, "E_S"] = 0.00001  # 완전히 0으로 나오는 것 방지

    return df.sort_values(by="E_S", ascending=False)


def compute_final_ranking(community_centers, tourist, restaurants, empty_houses) -> pd.DataFrame:
    """관광지 점수 × 맛집 점수 × 빈집 등급 점수를 곱해 종합 점수를 산출하고 정규화한다."""
    community_centers = community_centers.copy()
    community_centers["Score_Tourist"] = community_centers.apply(
        calculate_proximity_score, axis=1, target_df=tourist
    )
    community_centers["Score_Restaurant"] = community_centers.apply(
        calculate_proximity_score, axis=1, target_df=restaurants
    )
    community_centers["Final_Score"] = (
        community_centers["Score_Tourist"] * community_centers["Score_Restaurant"]
    )

    merged = community_centers.sort_values("기관명").reset_index(drop=True)
    empty_houses_sorted = empty_houses.sort_values("읍면동").reset_index(drop=True)
    merged = pd.concat([merged, empty_houses_sorted], axis=1)

    merged["Final_Score"] = round(merged["Final_Score"] * merged["E_S"], 6)

    scaler = MinMaxScaler()
    merged["종합점수_정규화"] = scaler.fit_transform(merged[["Final_Score"]])

    return merged.sort_values("Final_Score", ascending=False)


# =========================================================
# 5. 지도 시각화
# =========================================================
def build_map(community_centers, tourist, restaurants, output_path="all_map.html"):
    m = folium.Map(location=[36.8065, 128.6270], zoom_start=13)  # 영주시 중심 좌표

    for _, row in community_centers.iterrows():
        folium.Marker([row["위도"], row["경도"]], popup=row["기관명"], icon=folium.Icon(color="blue")).add_to(m)
    for _, row in tourist.iterrows():
        folium.Marker([row["위도"], row["경도"]], popup=row["관광지명"], icon=folium.Icon(color="green")).add_to(m)
    for _, row in restaurants.iterrows():
        folium.Marker([row["위도"], row["경도"]], popup=row["업소명"], icon=folium.Icon(color="orange")).add_to(m)

    m.save(output_path)
    print("지도 저장 완료:", output_path)


def find_optimal_point(tourist: pd.DataFrame, restaurants: pd.DataFrame, tourist_names: list[str], restaurant_names: list[str]):
    """
    최종 선정된 읍면동 내에서, 실제 인기 있는 관광지·맛집 좌표의 평균을 구해
    '면 안에서 가장 적절한 지점'을 한 단계 더 구체화한다.
    """
    selected_tourist = tourist[tourist["관광지명"].isin(tourist_names)]
    selected_restaurants = restaurants[restaurants["업소명"].isin(restaurant_names)]

    total_lat = selected_tourist["위도"].sum() + selected_restaurants["위도"].sum()
    total_lon = selected_tourist["경도"].sum() + selected_restaurants["경도"].sum()
    count = len(selected_tourist) + len(selected_restaurants)

    return total_lat / count, total_lon / count


# =========================================================
# 실행부
# =========================================================
if __name__ == "__main__":
    tourist, restaurants, empty_houses, community_centers = load_raw_data()
    tourist, restaurants, community_centers = clean_data(tourist, restaurants, community_centers)
    empty_houses = clean_empty_houses(empty_houses)

    gmaps_client = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
    restaurants = add_coordinates(restaurants, "주소", gmaps_client)
    community_centers = add_coordinates(community_centers, "소재지 도로명주소", gmaps_client)

    tourist, restaurants = score_tourist_and_restaurants(tourist, restaurants)

    # 1~2등급 빈집만 사용 (숙박 전환이 용이한 등급)
    empty_houses_12 = score_empty_houses(empty_houses, grades=["1등급", "2등급"])

    final_ranking = compute_final_ranking(community_centers, tourist, restaurants, empty_houses_12)

    print("최종 입지 순위 (종합 점수 기준):")
    print(final_ranking[["기관명", "Final_Score", "종합점수_정규화"]].head(10))

    build_map(community_centers, tourist, restaurants)

    # 최종 선정 지역(예: 문수면) 내 세부 추천 지점 계산 예시
    lat, lon = find_optimal_point(
        tourist, restaurants,
        tourist_names=["천지인전통사상체험관", "무섬외나무다리", "무섬외나무다리축제", "무섬마을한옥체험관"],
        restaurant_names=["사느레정원", "카페월호", "무섬식당"],
    )
    print(f"문수면 내 세부 추천 지점: 위도 {lat:.6f}, 경도 {lon:.6f}")

    final_ranking.to_csv("final_ranking.csv", index=False, encoding="cp949")
