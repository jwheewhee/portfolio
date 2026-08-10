## 데이터 출처
- 출처: 직접 크롤링 (82쿡/우아한 갱년기 카페, LG전자 공식 유튜브 가이드 영상 댓글, Google Play Store 리뷰)
- 취득 방법: 각 스크립트(`82cook_crawling.py`, `youtube_comment_crawling.py`, `playstore_review_crawling.py`) 실행 시 자동 수집
- 취득 시점: 2026 (각 채널 모두 하루 만에 수집 — 82쿡 3,010건 / 유튜브 댓글 99건 / 플레이스토어 리뷰 963건)

## 주요 컬럼 설명 (크롤링 결과 공통)
| 컬럼명 | 설명 | 타입 |
|---|---|---|
| title / comment | 게시글 제목 또는 댓글 원문 | str |
| date | 작성일 | str |
| source | 수집 채널(82cook / youtube / playstore) | str |
| comment_type | (유튜브 댓글만) 오류문의/사용법질문/호환성문의/개선요구/긍정후기/기타 | str |
| rating | (플레이스토어 리뷰만) 별점 1~5 | int |

## 배치 경로
크롤링 결과 csv는 각 스크립트 실행 시 프로젝트 폴더에 자동 생성됩니다. 원본 크롤링 데이터는 개인정보(작성자 닉네임 등) 포함 가능성이 있어 저장소에는 포함하지 않았습니다.
