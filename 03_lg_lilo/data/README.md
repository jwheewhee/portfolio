## 데이터 출처
- 출처: 직접 크롤링 (디시인사이드, 블라인드, 결혼준비 카페(directwedding), 인테리어 카페, 블로그, 유튜브) + KNU 감성사전
- 취득 방법: `01_cafe_crawler.py` 등 채널별 크롤링 스크립트로 수집, '#첫자취가전' 등 키워드 기반
- 취득 시점: 2026 · 12만 7천여 건 수집 → 3만 8천여 건 정제

## 주요 컬럼 설명
| 컬럼명 | 설명 | 타입 |
|---|---|---|
| text | 게시글/댓글 원문 | str |
| source | 수집 채널명 | str |
| date | 작성일 | str |
| actor_cluster | KMeans로 도출된 Actor 번호(0~5) | int |
| action_topic | LDA로 도출된 Action 토픽 번호 | int |
| satisfaction | 감성분석 기반 만족도 점수(-10~10) | float |
| importance | 등장 비율 기반 중요도 점수(0~10) | float |

## 외부 리소스
- **KNU 감성사전**(`SentiWord_info.json`): 감성분석에 필요. [군산대 소프트웨어융합공학과 KNU 한국어 감성사전](https://github.com/park1200656/KnuSentiLex)에서 별도로 받아 `03_lg_lilo/` 폴더에 위치시켜야 합니다. (라이선스 문제로 저장소에 직접 포함하지 않음)

## 배치 경로
크롤링 원본 데이터는 개인정보(작성자 닉네임 등) 포함 가능성이 있어 저장소에는 포함하지 않았습니다. `02_data_merging_preprocessing.py` 이후 단계부터는 이전 스크립트가 생성한 pickle/csv 파일을 입력으로 사용합니다.
