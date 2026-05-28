# AI Research & Academic Archive

> **대학교 인공지능(AI) 전공 학술 연구 및 텀 프로젝트 아카이브**
> 컴퓨터 비전(CV) 및 데이터 전처리 도메인에서 딥러닝(DL) 모델의 근본적인 원리를 이해하고, 데이터 파이프라인 구축부터 모델 미세조정(Fine-Tuning), 하이퍼파라미터 최적화까지 직접 수행한 연구 기록입니다.

---

## Projects List

### 1. CWT 이미지 변환 및 IMU 센서 기반 환자 활동 상태 분류
1D 시계열 센서(IMU) 데이터를 2D 이미지 형태(CWT Transform)로 변환하여 CNN 모델에 학습시키고, 환자의 6가지 행동 상태를 예측하는 '인공지능' 전공 딥러닝 분류 연구입니다.
* **Domain**: Artificial Intelligence (Data Preprocessing & Classification)
* **Model**: ResNet18
* **Tech Challenge**: 초기 학습 시 사전 학습된 모델과 데이터 간의 스케일 불일치로 성능이 0.61에 정체되는 문제 발생. 이를 해결하기 위해 Min-Max Normalization 전처리 코드를 직접 구현하여 데이터 분포를 0~1로 재조정.
* **Result**: 정규화 파이프라인 및 백본 미세조정(Fine-Tuning) 도입 후 정확도를 기존 대비 약 30% 향상시킨 **0.802 달성**.
* 📂 **[상세 보고서 및 코드 보기](./01_sensor-activity-classification)**

### 2. PASCAL VOC 데이터셋 객체 검출 모델 한계 돌파 및 성능 최적화
사전 학습된(Pre-trained) 모델을 새로운 데이터셋에 적용할 때 발생하는 오탐지(False Positive) 이슈를 해결하고, 학습 효율을 극대화하기 위한 하이퍼파라미터 및 백본(Backbone) 튜닝 연구입니다.
* **Domain**: Computer Vision (Object Detection)
* **Model**: SSD MobileNet V3 Large
* **Tech Challenge**: 단순 Epoch 증가의 비효율성을 배제하고, NMS(Non-Maximum Suppression) 임계값 조정 및 전체 레이어 미세조정(Full Fine-Tuning)을 통한 근본적 성능 개선.
* **Result**: 중복 검출 및 배경 오탐지 문제를 해결하고, 베이스라인(0.451)을 넘어 최종 **mAP 0.471**의 최고 성능 달성.
* 📂 **[상세 보고서 및 코드 보기](./02_pascal-voc-detection)**

### 3. 실시간 산불 예방을 위한 딥러닝 기반 흡연자 객체 검출 시스템
산불 발생의 핵심 원인인 '입산자 실화'를 사전에 예방하기 위해, CCTV 영상 내에서 흡연자와 담배 객체를 실시간으로 탐지하는 딥러닝 비전 시스템 연구입니다.
* **Domain**: Computer Vision (Object Detection)
* **Model**: SSD MobileNet V3 Large (Pre-trained on COCO)
* **Tech Challenge**: 픽셀 비중이 매우 작은 '담배' 객체 탐지를 위해 Random Affine, Color Jitter 등 강력한 데이터 증강(Augmentation) 도입 및 Cosine Annealing 스케줄러 적용.
* **Result**: 복잡한 배경 및 조명 변화 속에서도 흡연 행위를 정확히 탐지해내며 훈련 손실(Loss)을 4.35 ➔ 0.69로 획기적으로 단축.
* 📂 **[상세 보고서 및 코드 보기](./03_smoker-detection-system)**

---

## 🛠️ Key AI/DL Skills
* **Framework**: PyTorch
* **Data Processing**: Min-Max Normalization, CWT(Continuous Wavelet Transform), COCO JSON Parsing
* **Augmentation**: Color Jitter, Random Horizontal Flip, Random Affine
* **Optimization**: SGD/Adam Optimizer, Cosine Annealing LR Scheduling, NMS Threshold Tuning
* **Methodology**: Transfer Learning, Full/Partial Fine-Tuning
