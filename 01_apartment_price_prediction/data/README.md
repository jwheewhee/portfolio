## 데이터 출처
- 출처: [데이콘 아파트 실거래가 예측 해커톤](https://dacon.io/competitions/open/236130)
- 취득 방법: 대회 페이지에서 `train.csv`, `test.csv`, `sample_submission.csv` 다운로드
- 취득 시점: 2023.07

## 주요 컬럼 설명
| 컬럼명 | 설명 | 타입 |
|---|---|---|
| exclusive_use_area | 전용면적(㎡) | float |
| floor | 층수 | int |
| year_of_completion | 건축연도 | int |
| transaction_real_price | 실거래가(예측 대상, target) | int |
| transaction_year | 거래연도 (거래연월에서 분리) | int |
| transaction_month | 거래월 (거래연월에서 분리) | int |

## 샘플 데이터 (5행)
```
exclusive_use_area,floor,year_of_completion,transaction_real_price,transaction_year,transaction_month
158.54,13,1983,174000,2014,1
127.61,6,1983,157500,2014,1
127.61,5,1983,150000,2014,1
127.61,9,1983,152000,2014,2
84.81,3,1983,116000,2014,2
```

## 배치 경로
`train.csv`, `test.csv`, `sample_submission.csv`를 이 폴더(`data/`)에 위치시키면 됩니다.
데이콘 규정상 원본 데이터는 저장소에 포함하지 않았습니다.
