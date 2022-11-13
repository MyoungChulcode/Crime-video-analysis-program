# crime-video-analysis-program

본 코드는,
Real-world Anomaly Detection in Surveillance Videos (2018, CVPR)
SLIC: Self-Supervised Learning with Iterative Clustering for Human Action Videos (2022, CVPR)
논문을 참조하여 작성하였으며, 자체 GUI 프로그램을 제작하여 범죄 유형을 분석하였다.

**1. 프로젝트 선정 배경 및 필요성**
1) [범죄 발생 건수의 증가추세] 범죄 예방과 신속한 조치를 위한 방안으로 최근 CCTV 설치 대수는 증가하였지만 정작 범죄 발생 건수는 줄지 않고 있다. 국정모니터링 시스템에서 조사한 공공기관 CCTV 설치 및 운영 현황을 살피면, 2016 에는 409,028 대, 2020 년에는 704,134 대로 CCTV 증가대수가 대략 1.7 배 증가하였다. 하지만 경찰청 범죄 통계에 따르면, CCTV 가 다수 분포하고 있는 주차장에서 발생한 전체 범죄 건수는 2016 년에는 23,259 건, 2020 년에는 27,839 건으로 증가 추세를 보이고 있다.
2) [영상 분석 시스템의 필요성] 교통조사계 경찰관, 주차장 관리자, CCTV 운영자, 일반인을 대상으로 진행한 ‘영상 분석 시스템에 대한 만족도 설문조사’에서 대부분 ‘불만족한다’는 의견을 보였다. 특히 사고 영상 부분을 검출하는데 오래 걸리며, 실시간 이상/위급 상황을 판단하는데 어려움을 겪고 있다.


**2. 프로젝트 주요 내용**
1) [Anomaly detection] 범죄 장면을 담은 Video dataset을 사용해 전체 frame 에서 돌발 범죄 내용이 담 긴 Anomaly frame을 추출한다.
2) [Clustering] 추출한 Anomaly video sequence를 활용하여 범죄 유형, 행동을 기준으로 Clustering 한다.
3) [GUI program] 자동화된 범죄 영상 분석 프로그램을 구현해 Anomaly score 와 Clustering result 를 시각화하고 범죄 유형을 분석한다.


**3. 목표**
1) [Anomaly detection] Anomaly score 를 Prediction 하여 해당 frame 의 anomaly 유무를 판별
2) [Crime action clustering] Anomaly sequence(범죄 장면)를 Clustering 하였을 때, 범죄 유형 분석을 효율적으로 진행할 수 있음을 성능 수치 비교 및 시각화를 통해 비교
3) [GUI program] Video Anomaly detection & Clustering 모델을 구현하고 GUI 시각화 프로그램 제작
4) [Performance Improvements] 기존 이상치 탐색 및 군집화 모델의 성능 향상
5) [Crime Video Retrieval System] 추가적으로 실제 현장에서 유용하게 사용될 수 있는 유사 범죄 영상
검색 시스템을 구현하는 것을 목표로 한다


**4. 개발 내용**
1) [Development process] Crime video dataset --> anomaly detection --> clustering with extracted anomalous frames --> export model network with ONNX --> GUI Program & visualization --> Comparison Clustering result --> Crime Video Retrieval System
2) [Crime video dataset] UCF Crime, AI-HUB data 등 폭력, 강도, 교통사고, 폭발 등의 Anomaly data set
3) [Anomaly detection] CNN(C3d feature extraction, Anomaly score prediction)으로 동영상에서 범죄가 발생하는 모습이 담긴 Anomaly sequence를 추출한다.
4) [clustering with extracted anomalous frames] Anomaly action & category clustering을 진행하며, Anomaly sequence(범죄 장면)를 Clustering 하였을 때, 범죄 유형 분석을 효율적으로 진행할 수 있음을 성능 수치 비교 및 시각화를 통해 비교
5) [Export model network with ONNX & GUI Program & visualization] Pytorch로 구현한 Neural network 구조를 ONNX로 변환한다. C++ GUI Program을 통해 prediction 과 visualization을 구현한다.
6) [Comparison Clustering result] Clustering 결과를 embedding space 내에서 visualize 및 정성적 평가를 하여 목표를 달성하였는지 판단한다. 전체 프레임을 모두 사용하지 않았으므로, 전체 영상을 사용할 결과와 정량적, 정성적 결과를 비교하여 성능 저하가 있는지 검토한다.
7) [Crime Video Retrieval System] 추가적으로, 실제 현장에서 유용하게 사용될 수 있는 유사 범죄 영상 검색 시스템을 구현하는 것을 목표로 한다.


**5. 기대효과 및 활용방안**
1) [영상 분석 시스템 개선] 범죄 영상분석과 직접적인 관련성을 가진 CCTV 통제실에서 긴 감시 카메라 영상을 수동으로 분석할 필요 없이 제안하는 시스템을 활용하여 자동으로 범죄 장면만 추출하고 범죄 유형과 행동을 분석할 있다.
2) [Crime video DB] Video Clustering 을 통해 유사한 범죄 영상들끼리 그룹화되어 있는 데이터베이스를 구축한다. 어떤 사건이 발생했을 때 과거에 발생한 유사한 영상을 찾아 낼 수 있어 경찰의 빠른 사건 ·사고 처리에 도움이 될 수 있다.
