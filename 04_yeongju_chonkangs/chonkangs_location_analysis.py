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
2. 동 단위로 묶여 있던 빈집 데이터를 면적 비율로 세부 지역에 재분배
3. Google Maps Geocoding API로 주소를 위경도 좌표로 변환
4. Folium으로 전체 데이터를 지도에 시각화
5. 행정복지센터-관광지/맛집/정류장 간 거리 계산
6. 인기도 점수 + 주거환경(빈집 등급) 점수 - 평균거리로 종합 점수 산출, 최종 입지 선정
"""

import pandas as pd
import googlemaps
import folium
from geopy.distance import geodesic

GOOGLE_MAPS_API_KEY = "YOUR_API_KEY"  # 발급받은 API 키로 교체


# =========================================================
# 1. 데이터 불러오기 및 전처리
# =========================================================
def load_raw_data():
    tourist = pd.read_csv("center_tourist_spots.csv", encoding="cp949")
    restaurants = pd.read_csv("local_restaurants.csv", encoding="cp949")
    empty_houses = pd.read_csv("empty_houses.csv", encoding="cp949")
    bus_stops = pd.read_csv("bus_stops.csv", encoding="cp949")
    community_centers = pd.read_csv("community_centers.csv", encoding="cp949")
    return tourist, restaurants, empty_houses, bus_stops, community_centers


def clean_data(tourist, restaurants, community_centers):
    """분석에 불필요한 컬럼을 제거하고, 숙박업소는 경쟁시설이므로 관광지 목록에서 제외한다."""
    tourist = tourist[tourist["중심카테고리"] != "숙박"].copy()
    tourist = tourist.drop(["중심카테고리", "분류"], axis=1)
    tourist = tourist.rename(columns={"중심 POI X 좌표": "경도", "중심 POI Y 좌표": "위도"})

    restaurants = restaurants.drop(["분류", "방문자수"], axis=1)
    community_centers = community_centers.drop(["대표전화번호", "팩스번호", "데이터기준일자"], axis=1)

    return tourist, restaurants, community_centers


# =========================================================
# 2. 빈집 데이터 면적 비율 재분배
# =========================================================
# 통계상 하나로 묶여 있던 동을 실제 세부 행정동 면적 비율로 나눔
AREA_DATA = {
    "가흥1동": 7.08, "가흥2동": 17.12,
    "영주1동": 1.02, "영주2동": 0.49,
    "휴천1동": 5.24, "휴천2동": 0.88, "휴천3동": 10.32,
}


def redistribute_empty_houses(df_empty_houses: pd.DataFrame) -> pd.DataFrame:
    """가흥동/영주동/휴천동으로 묶여 있던 빈집 등급 데이터를 세부 동 면적 비율로 재분배한다."""
    base_groups = {
        "가흥": (df_empty_houses[df_empty_houses["읍면동"] == "가흥동"], 7.08 + 17.12),
        "영주": (df_empty_houses[df_empty_houses["읍면동"] == "영주동"], 1.02 + 0.49),
        "휴천": (df_empty_houses[df_empty_houses["읍면동"] == "휴천동"], 5.24 + 0.88 + 10.32),
    }

    grade_cols = ["계", "1등급", "2등급", "3등급", "4등급"]
    df_new = pd.DataFrame()

    for subregion, area in AREA_DATA.items():
        prefix = next(p for p in base_groups if subregion.startswith(p))
        df_base, base_area = base_groups[prefix]

        ratio = area / base_area
        df_sub = df_base.copy()
        df_sub["읍면동"] = subregion
        df_sub[grade_cols] = (df_sub[grade_cols] * ratio).astype(int)
        df_new = pd.concat([df_new, df_sub])

    df_result = df_empty_houses[~df_empty_houses["읍면동"].isin(["가흥동", "영주동", "휴천동"])]
    df_result = pd.concat([df_result, df_new], ignore_index=True)
    return df_result


# =========================================================
# 3. 지오코딩 (주소 → 위경도)
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
# 4. 지도 시각화
# =========================================================
def build_map(community_centers, tourist, restaurants, bus_stops, output_path="all_map.html"):
    m = folium.Map(location=[36.8065, 128.6270], zoom_start=13)  # 영주시 중심 좌표

    for _, row in community_centers.iterrows():
        folium.Marker([row["위도"], row["경도"]], popup=row["기관명"], icon=folium.Icon(color="blue")).add_to(m)
    for _, row in tourist.iterrows():
        folium.Marker([row["위도"], row["경도"]], popup=row["관광지명"], icon=folium.Icon(color="green")).add_to(m)
    for _, row in restaurants.iterrows():
        folium.Marker([row["위도"], row["경도"]], popup=row["업소명"], icon=folium.Icon(color="orange")).add_to(m)
    for _, row in bus_stops.iterrows():
        folium.Marker([row["위도"], row["경도"]], popup=row["정류장명"], icon=folium.Icon(color="red")).add_to(m)

    m.save(output_path)
    print("지도 저장 완료:", output_path)


# =========================================================
# 5. 거리 계산
# =========================================================
def calculate_distances(center_df: pd.DataFrame, spot_df: pd.DataFrame) -> dict:
    """행정복지센터별로 대상 지점들까지의 거리(km) 목록을 계산한다."""
    distances = {}
    for _, center_row in center_df.iterrows():
        center_coords = (center_row["위도"], center_row["경도"])
        spot_distances = [
            geodesic(center_coords, (spot_row["위도"], spot_row["경도"])).kilometers
            for _, spot_row in spot_df.iterrows()
        ]
        distances[center_row["기관명"]] = spot_distances
    return distances


# =========================================================
# 6. 종합 점수 계산 및 최종 입지 선정
# =========================================================
def calculate_popularity(spot_df: pd.DataFrame, center_df: pd.DataFrame) -> dict:
    """가까운 거리에 순위가 높은(=인기 있는) 관광지·맛집이 많을수록 높은 점수를 부여한다."""
    scores = {}
    for _, center_row in center_df.iterrows():
        center_coords = (center_row["위도"], center_row["경도"])
        total_score = 0
        for _, spot_row in spot_df.iterrows():
            distance = geodesic(center_coords, (spot_row["위도"], spot_row["경도"])).kilometers
            total_score += 1 / (distance + 0.1) * spot_row["순위"]
        scores[center_row["기관명"]] = total_score
    return scores


def assign_environment_scores(df_empty_houses: pd.DataFrame) -> dict:
    """숙박업 전환이 용이한 1~2등급 빈집에 더 높은 가중치를 부여해 환경 점수를 계산한다."""
    scores = {}
    for _, row in df_empty_houses.iterrows():
        score = row["1등급"] * 4 + row["2등급"] * 3 + row["3등급"] * 2 + row["4등급"] * 1
        scores[row["읍면동"]] = score
    return scores


def rank_final_locations(
    tourist_distances: dict, restaurant_distances: dict, bus_stop_distances: dict,
    popularity_scores: dict, environment_scores: dict, top_n: int = 10,
) -> list[tuple[str, float]]:
    """
    종합 점수 = 인기도 점수 + 주거환경 점수 - 평균 거리
    (관광지·맛집이 가깝고 인기 있을수록, 빈집 활용 가치가 높을수록 높은 점수)
    """
    combined_scores = {}
    for center, tourist_dist in tourist_distances.items():
        restaurant_dist = restaurant_distances[center]
        bus_stop_dist = bus_stop_distances[center]
        all_distances = tourist_dist + restaurant_dist + bus_stop_dist
        avg_distance = sum(all_distances) / len(all_distances)

        combined_scores[center] = (
            popularity_scores[center] + environment_scores.get(center, 0) - avg_distance
        )

    return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


# =========================================================
# 실행부
# =========================================================
if __name__ == "__main__":
    tourist, restaurants, empty_houses, bus_stops, community_centers = load_raw_data()
    tourist, restaurants, community_centers = clean_data(tourist, restaurants, community_centers)
    empty_houses = redistribute_empty_houses(empty_houses)

    gmaps_client = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
    restaurants = add_coordinates(restaurants, "주소", gmaps_client)
    community_centers = add_coordinates(community_centers, "소재지 도로명주소", gmaps_client)

    build_map(community_centers, tourist, restaurants, bus_stops)

    tourist_distances = calculate_distances(community_centers, tourist)
    restaurant_distances = calculate_distances(community_centers, restaurants)
    bus_stop_distances = calculate_distances(community_centers, bus_stops)

    popularity_scores = calculate_popularity(tourist, community_centers)
    environment_scores = assign_environment_scores(empty_houses)

    final_ranking = rank_final_locations(
        tourist_distances, restaurant_distances, bus_stop_distances,
        popularity_scores, environment_scores,
    )

    print("최종 입지 순위 (상위 10개 행정복지센터 기준):")
    for i, (center, score) in enumerate(final_ranking, start=1):
        print(f"{i}. {center}: {score:.2f}")

    pd.DataFrame(final_ranking, columns=["센터명", "종합 점수"]).to_csv(
        "top_10_centers.csv", index=False, encoding="cp949"
    )
