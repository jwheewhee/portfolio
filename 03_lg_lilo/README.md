# 🔄 LG LILO — LG전자 DX School CX 프로젝트 (우수상, 2위)

가전 구독 시장에서 유독 낮은 MZ세대 전환율의 원인을 텍스트 데이터로 구조화하고,
해지 대신 다음 사용자에게 이어지는 **순환형 구독 승계 서비스 'LG LILO'**를 기획한 프로젝트입니다.

이 레포지토리는 문제 정의와 솔루션 근거를 뒷받침하는 데이터 분석 코드를 담고 있습니다.

## 📌 프로젝트 배경
가전 구독 시장은 빠르게 성장하고 있지만, 정작 구독 친화적인 세대로 여겨지는 MZ세대의 실제 이용률은
낮게 나타났습니다. 콘텐츠 구독과 달리 가전 구독은 설치·이전이 필요하고 3~6년의 장기 계약이 걸려 있어,
"라이프스타일이 바뀌었을 때 계약을 어떻게 정리할 것인가"가 진입장벽이라는 가설을 세웠고,
이를 검증하기 위해 디시인사이드, 블라인드, 결혼준비 카페, 블로그, 유튜브 등에서 3만 8천여 건의
텍스트 데이터를 수집·분석했습니다.

## 🔍 분석 파이프라인

| 순서 | 파일 | 내용 |
|---|---|---|
| 1 | `01_cafe_crawler.py` | directwedding(결혼준비 카페) 게시글·댓글 크롤링 |
| 2 | `02_data_merging_preprocessing.py` | 팀원별 채널(카페/유튜브/블라인드/디시인사이드 등) 수집 데이터 통합 + 1차 전처리 |
| 3 | `03_actor_clustering.py` | TF-IDF + KMeans로 6개 Actor(고객 유형) 도출 |
| 4 | `04_action_topic_modeling.py` | Actor별 LDA 토픽 모델링으로 세부 Action 도출 |
| 5 | `05_sentiment_opportunity_analysis.py` | 감성분석 + Opportunity Score로 개선 우선순위(CAM) 도출 |

> 팀 전체가 크롤링부터 참여했고, 그중 directwedding 카페 크롤링을 담당했습니다.
> 이후 전체 데이터 통합·1차 전처리, Actor/Action 도출, 감성분석·CAM 제작까지 분석 파트를 맡아 진행했습니다.

### 1) 데이터 수집 및 통합
2030 세대가 많이 쓰는 채널(디시인사이드, 블라인드, 결혼준비 카페, 인테리어 카페, 블로그, 유튜브)에서
'#첫자취가전', '#신혼가전', '#가전구독후기' 등 관련 키워드로 수집한 데이터를 하나의 코퍼스로 통합하고,
결측치·중복 제거 등 1차 정제를 진행했습니다.

### 2) Actor 도출 — TF-IDF + KMeans
Kiwi 형태소 분석 후 TF-IDF(min_df=4, max_df=0.40, max_features=300)로 벡터화하고, k=4~9 구간에서
실루엣 지수를 비교했습니다. 지수상으로는 k=8(0.0468)이 최고점이었지만 군집이 과도하게 세분화되어
해석이 어려워, 점수 차이가 크지 않은 구간에서 **k=6**을 최종 선택했습니다.

도출된 6개 Actor:

| Actor | 특징 |
|---|---|
| Actor 0 | 생활변화 가전교체 고민러 (메인 페르소나 — 정희원, 32세) |
| Actor 1 | 조건비교 가입러 — 구매·할부·구독·렌탈 총비용을 비교하는 실속형 |
| Actor 2 | 이용 중 AS 및 관리 민원러 — 필터·청소·AS 등 관리 품질을 중시 |
| Actor 3 | 정수기 렌탈 고민러 — 생수 vs 정수기 렌탈을 생활 패턴 기준으로 비교 |
| Actor 4 | 설치 및 이전비용러 — 이사 시 설치비·배관·철거 조건을 꼼꼼히 확인 |
| Actor 5 | 계약해지 및 위약금 불만러 — 해지 대신 승계·양도 대안을 먼저 탐색 |

### 3) Action 도출 — LDA 토픽 모델링
Actor별로 Perplexity/Coherence를 비교해 개별적으로 토픽 수를 선정하고(passes=20, iterations=50),
문서별 대표 토픽을 Action으로 라벨링했습니다. pyLDAvis 표시 번호와 LDA 모델 내부 토픽 번호가
다르게 정렬되는 경우가 있어, 이를 수동으로 대응시켜 최종 Action 번호를 확정했습니다.

### 4) 감성분석 및 Opportunity Score (CAM)
KNU 감성사전으로 Actor-Action 조합별 만족도(Satisfaction, -10~10)를 계산하고, 전체 문서 대비
등장 비율로 중요도(Importance, 0~10)를 산출했습니다.

```
Opportunity Score = Importance + max(Importance − Satisfaction, 0)
```

중요도는 높은데 만족도는 낮은 조합일수록 점수가 커지도록 설계해 개선 우선순위를 정량화했습니다.
예를 들어 Actor 5(계약해지 불만러)의 "가입 시 사은품·할인반환금까지 추가 청구되는 문제"는
감성 점수 -7.11로 가장 낮은 축에 속해 핵심 기회 영역으로 도출됐고, 이는 LILO의 **공식 승계 마켓**
설계로 직접 이어졌습니다.

## 🛠 사용 기술
`Python` `Kiwi` `Okt` `TF-IDF` `KMeans` `LDA(gensim)` `KNU 감성사전` `MinMaxScaler` `Selenium`

## 📁 파일 구성
```
03_lg_lilo/
 ├─ README.md
 ├─ 01_cafe_crawler.py
 ├─ 02_data_merging_preprocessing.py
 ├─ 03_actor_clustering.py
 ├─ 04_action_topic_modeling.py
 └─ 05_sentiment_opportunity_analysis.py
```

## ▶️ 실행 방법
```bash
pip install selenium beautifulsoup4 pandas openpyxl tqdm
pip install kiwipiepy konlpy scikit-learn gensim adjustText

python 01_cafe_crawler.py
python 02_data_merging_preprocessing.py
python 03_actor_clustering.py
python 04_action_topic_modeling.py
python 05_sentiment_opportunity_analysis.py
```
> 각 스크립트는 이전 단계의 출력 파일(pickle/csv)을 입력으로 받는 순차 파이프라인입니다.
> `05_sentiment_opportunity_analysis.py`는 KNU 감성사전(`SentiWord_info.json`)이 별도로 필요합니다.

## 👤 본인 역할
- directwedding 카페 크롤링 담당 (팀 전체는 채널별로 분담해 크롤링 진행)
- 팀원별 수집 데이터 통합 및 1차 전처리
- TF-IDF + KMeans 기반 Actor 도출
- Actor별 LDA 토픽 모델링으로 Action 도출
- 감성분석 및 Opportunity Score 산출, CAM(기회영역) 제작

## 💡 배운 점
Actor 단계에서 키워드를 너무 타이트하게 잡으면 이후 Action 단계에서 데이터가 제대로 나뉘지 않아
되돌아가야 한다는 것을 미니 프로젝트에서 미리 겪어봤고, 이 교훈을 전처리 방식(불용어 처리, 형태소
분석 파라미터)에 반영해 시행착오를 줄일 수 있었습니다. 또한 실루엣 지수 같은 정량 지표가 항상
정답은 아니며, 통계적 최적값과 실제 해석 가능성 사이에서 균형을 잡는 것이 분석가의 판단 영역이라는
것을 체감했습니다.
