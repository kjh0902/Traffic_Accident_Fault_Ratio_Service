# Traffic Accident Fault Ratio Service

교통사고 영상을 입력하면 classification, detection/tracking, base ratio table, adjustment model을 사용해 최종 과실비율을 추론하는 Gradio 서비스입니다.

## 폴더 구조

```text
traffic_accident/
├─ app.py
├─ requirements.txt
├─ inference/
│  ├─ config.py
│  ├─ video_classifier.py
│  ├─ detector.py
│  ├─ tracker.py
│  ├─ trajectory.py
│  ├─ base_ratio.py
│  ├─ adjustment.py
│  ├─ pipeline.py
│  └─ report.py
├─ data/
│  └─ lookup/base_ratio_table.csv
└─ weights/
   ├─ classifier/best.pth
   ├─ detector/best.pth
   └─ adjustment/adjustment_model.joblib
```

## 실행

```powershell
pip install -r requirements.txt
python app.py 
```

기본 device는 `cuda:0`입니다. GPU 서버에서 그대로 실행하면 되고, 로컬에서 구조만 확인할 때는 `--device cpu`로 바꿔 실행할 수 있습니다.

## 추론 흐름

1. 영상 classifier가 `accident_place_feature`, `vehicle_a_progress_info`, `vehicle_b_progress_info`를 예측합니다.
2. 사용자가 선택한 `accident_place`와 classifier 예측값 3개로 `base_ratio_table.csv`에서 기본 과실비율을 조회합니다.
3. detector와 IoU tracker가 차량 궤적을 만들고, 선진입/감속 여부 같은 evidence feature를 계산합니다.
4. `weights/adjustment/adjustment_model.joblib`가 evidence 기반 보정값을 예측합니다.
5. 기본 과실비율에 보정값을 더해 최종 A/B 과실비율과 탐지 영상을 출력합니다.

