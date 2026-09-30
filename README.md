# WM-811K Wafer Map Defect Classification

PyTorch 기반 CNN을 활용하여 **WM-811K wafer map의 결함 패턴을 9개 클래스로 분류**하고,  
데이터 불균형과 데이터 중복·lot leakage 문제를 점검하여 보다 엄격한 조건에서 모델의 일반화 성능을 평가한 프로젝트입니다.

단순히 높은 정확도를 얻는 것보다, **실제 새로운 wafer lot에서도 모델이 일반화할 수 있는지 검증하는 것**에 초점을 두었습니다.

---

## 1. Project Overview

반도체 제조 과정에서 wafer map에는 불량 die의 공간적 분포가 나타나며, 이러한 패턴은 공정 이상 원인을 분석하는 데 활용될 수 있습니다.

본 프로젝트에서는 **WM-811K dataset**의 wafer map을 CNN으로 학습하여 다음 9개 패턴을 분류합니다.

| Class | Description |
|---|---|
| `none` | 결함 패턴 없음 |
| `Center` | 중앙 집중형 |
| `Donut` | 도넛 형태 |
| `Edge-Loc` | 가장자리 국부 결함 |
| `Edge-Ring` | 가장자리 링 형태 |
| `Loc` | 국부적 결함 |
| `Random` | 무작위 결함 |
| `Scratch` | 스크래치 형태 |
| `Near-full` | wafer 대부분이 결함 |

WM-811K에는 심한 클래스 불균형이 존재하므로 Accuracy만으로 모델을 평가하지 않고 **Macro F1을 주요 평가 지표**로 사용했습니다.

---

## 2. Key Objectives

이 프로젝트의 주요 목표는 다음과 같습니다.

- CNN을 이용한 wafer defect pattern classification
- WM-811K의 심한 class imbalance 분석
- Standard Cross Entropy와 Weighted Cross Entropy 비교
- wafer lot 단위의 데이터 leakage 여부 점검
- 동일 wafer map의 duplicate 및 conflicting label 탐지
- Train / Validation / Test 사이의 exact input overlap 제거
- 여러 random seed를 이용한 실험 안정성 확인
- unseen lot / unseen input 환경에서 최종 일반화 성능 평가

---

## 3. Dataset

사용 데이터:

```text
WM-811K / LSWMD.pkl
```

원본 데이터셋의 크기는 다음과 같습니다.

```text
811,457 wafers
```

원본 데이터 컬럼:

```text
waferMap
dieSize
lotName
waferIndex
trianTestLabel
failureType
```

이 중 명확한 `failureType` label을 가지고 있는 wafer만 추출했습니다.

```text
Labeled wafers: 172,950
```

### Class Distribution

| Class | Samples |
|---|---:|
| none | 147,431 |
| Center | 4,294 |
| Donut | 555 |
| Edge-Loc | 5,189 |
| Edge-Ring | 9,680 |
| Loc | 3,593 |
| Random | 866 |
| Scratch | 1,193 |
| Near-full | 149 |

`none` class가 대부분을 차지하는 **극심한 class imbalance**를 확인할 수 있습니다.

---

## 4. Why Data Leakage Matters

초기 실험에서는 전체 labeled dataset을 stratified random split하여 다음과 같이 구성했습니다.

```text
Train      : 138,360
Validation : 17,295
Test       : 17,295
```

Weighted Cross Entropy를 적용한 초기 held-out test에서는 다음 결과를 얻었습니다.

| Metric | Score |
|---|---:|
| Accuracy | 0.9612 |
| Macro F1 | 0.7776 |
| Weighted F1 | 0.9574 |

하지만 이후 데이터 구조를 추가로 조사하면서 **단순 random split이 실제 일반화 성능을 과대평가할 가능성**을 점검하게 되었습니다.

따라서 최종 실험에서는 WM-811K가 제공하는 original Training/Test split과 `lotName`을 이용하여 보다 엄격한 검증을 진행했습니다.

---

## 5. Dataset Audit

### 5.1 Official Train / Test Split

WM-811K의 original split을 확인한 결과:

```text
Training : 54,355
Test     : 118,595
```

정제 과정에서 conflicting samples를 제거한 이후:

```text
Official Training : 54,341
Official Test     : 118,581
```

### 5.2 Lot Leakage Check

Training과 Test의 `lotName` overlap을 조사했습니다.

```text
Official Train/Test lot overlap: 0
```

즉, official split 자체는 **lot-disjoint** 상태였습니다.

### 5.3 Conflicting Labels

동일한 raw wafer map에 서로 다른 label이 부여된 사례를 hash 기반으로 탐지했습니다.

```text
Raw conflicting hash groups: 14
Rows removed: 28
```

해당 샘플은 학습 전에 제거했습니다.

### 5.4 Exact Input Duplicate

Train과 Official Test 사이에서 동일한 wafer input이 존재하는지 확인했습니다.

```text
Official Test raw-map overlap rows       : 3,144
Official Test processed-input overlap rows: 3,144
```

보다 엄격한 평가를 위해 해당 입력을 Test set에서 제거했습니다.

최종 test set은 다음과 같습니다.

```text
Strict unseen-input Test: 115,437
```

최종적으로:

```text
Train/Test lot overlap       : 0
Processed input overlap      : 0
```

이 되도록 실험 데이터를 구성했습니다.

---

## 6. Data Split

Official Training set 내부에서 `lotName`을 group으로 사용하여 validation set을 구성했습니다.

`StratifiedGroupKFold`를 사용해 class distribution을 최대한 유지하면서 Train과 Validation의 lot을 분리했습니다.

최종 데이터 크기:

| Dataset | Samples |
|---|---:|
| Train | 43,441 |
| Validation | 10,889 |
| Strict Test | 115,437 |

Test set은 모델 선택이나 hyperparameter 변경에 사용하지 않고 최종 평가에만 사용했습니다.

---

## 7. Preprocessing

Wafer map은 각각 크기가 다르기 때문에 CNN 입력을 위해 `64 × 64` 크기로 변환했습니다.

단순 stretch resize 대신 다음 과정을 적용했습니다.

```text
Original Wafer Map
        ↓
Aspect-ratio Preserving Resize
        ↓
Nearest-neighbor Interpolation
        ↓
Center Zero Padding
        ↓
64 × 64 Wafer Map
```

### Why Nearest-neighbor?

Wafer map의 값은 연속적인 pixel intensity가 아니라 categorical 값이므로 interpolation 과정에서 새로운 중간값이 생성되지 않도록 **nearest interpolation**을 사용했습니다.

최종 wafer map은 세 가지 categorical 값을 one-hot encoding하여 다음 형태로 CNN에 입력합니다.

```text
[64, 64]
    ↓
One-Hot Encoding
    ↓
[3, 64, 64]
```

---

## 8. Model Architecture

복잡한 모델보다 데이터 분할과 검증 방법에 집중하기 위해 비교적 간단한 CNN을 사용했습니다.

```text
Input (3 × 64 × 64)

Conv2D 3 → 32
BatchNorm
ReLU
MaxPool

Conv2D 32 → 64
BatchNorm
ReLU
MaxPool

Conv2D 64 → 128
BatchNorm
ReLU
MaxPool

Adaptive Average Pooling
Flatten
Linear 128 → 9

Output: 9 classes
```

PyTorch 구현:

```python
class WaferCNN(nn.Module):

    def __init__(self, num_classes=9):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

---

## 9. Handling Class Imbalance

WM-811K는 `none` class가 압도적으로 많은 imbalanced dataset입니다.

예를 들어 단순히 모든 wafer를 `none`으로 예측해도 Strict Test에서:

```text
Accuracy : 0.9325
Macro F1 : 0.1072
```

를 얻습니다.

따라서 높은 Accuracy만으로는 실제 defect classification 성능을 판단하기 어렵습니다.

본 프로젝트에서는 다음 두 loss를 비교했습니다.

### Standard Cross Entropy

```python
nn.CrossEntropyLoss()
```

### Sqrt-Inverse Weighted Cross Entropy

각 class의 학습 샘플 수를 `N_c`라고 할 때:

```text
w_c ∝ 1 / √N_c
```

로 class weight를 설정했습니다.

단순 inverse frequency인 `1 / N_c`보다 minority class에 대한 지나친 보정을 완화하기 위해 square root를 적용했습니다.

---

## 10. Training Setup

주요 학습 설정:

| Parameter | Value |
|---|---|
| Image Size | 64 × 64 |
| Batch Size | 128 |
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Max Epochs | 30 |
| Early Stopping Patience | 5 |
| Model Selection Metric | Validation Macro F1 |
| Seeds | 42, 123, 2026 |

각 방법은 동일한 조건에서 세 개의 random seed를 이용하여 반복 실험했습니다.

---

## 11. Baselines

CNN이 실제 wafer defect pattern을 학습하는지 확인하기 위해 추가 baseline을 평가했습니다.

### Always-none Baseline

```text
Accuracy : 0.9325
Macro F1 : 0.1072
```

높은 Accuracy에도 불구하고 minority defect pattern을 전혀 구분하지 못하기 때문에 Macro F1은 매우 낮습니다.

### Geometry-only Baseline

wafer image 자체가 아닌 geometry 정보만 사용한 baseline:

```text
Accuracy : 0.1225
Macro F1 : 0.0492
```

이를 통해 단순 wafer 크기나 shape 정보만으로는 defect pattern을 충분히 분류하기 어렵다는 것을 확인했습니다.

---

## 12. Experimental Results

### Standard CE vs Weighted CE

| Method | Val Macro F1 | Test Accuracy | Test Macro F1 |
|---|---:|---:|---:|
| CE — Seed 42 | 0.8310 | 0.5532 | 0.3615 |
| CE — Seed 123 | 0.8263 | 0.6845 | 0.3977 |
| CE — Seed 2026 | 0.7352 | 0.4383 | 0.3913 |
| WCE — Seed 42 | **0.8740** | 0.7869 | **0.4250** |
| WCE — Seed 123 | 0.8045 | 0.4381 | 0.4115 |
| WCE — Seed 2026 | 0.7511 | 0.4658 | 0.3844 |

### Mean ± Standard Deviation

| Method | Validation Macro F1 | Test Macro F1 |
|---|---:|---:|
| CE | 0.7975 ± 0.0540 | 0.3835 ± 0.0193 |
| WCE | **0.8098 ± 0.0617** | **0.4069 ± 0.0206** |

Seed별 `WCE - CE` Test Macro F1 차이:

```text
Seed 42   : +0.0635
Seed 123  : +0.0138
Seed 2026 : -0.0068

Mean difference: +0.0235
```

Validation 결과를 기준으로 최종 방법은 **Sqrt-Inverse Weighted Cross Entropy**로 선택했습니다.

---

## 13. Final Representative Model

최종 방법:

```text
Weighted Cross Entropy
w_c ∝ 1 / √N_c
```

대표 실험:

```text
Seed: 123
```

Strict unseen-input test 결과:

| Metric | Score |
|---|---:|
| Accuracy | 0.4381 |
| Macro Precision | 0.5515 |
| Macro Recall | 0.4654 |
| **Macro F1** | **0.4115** |
| Weighted F1 | 0.5824 |
| Balanced Accuracy | 0.4654 |

### Per-Class Performance

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| none | 0.9755 | 0.4375 | 0.6041 |
| Center | 0.0300 | 0.2107 | 0.0525 |
| Donut | 0.4648 | 0.2260 | 0.3041 |
| Edge-Loc | 0.3925 | 0.3258 | 0.3560 |
| Edge-Ring | 0.9422 | 0.4946 | 0.6487 |
| Loc | 0.0265 | 0.7869 | 0.0512 |
| Random | 0.7008 | 0.7037 | 0.7023 |
| Scratch | 0.5893 | 0.0484 | 0.0894 |
| Near-full | 0.8416 | 0.9551 | 0.8947 |

---

## 14. What I Learned

초기의 random split 실험에서는:

```text
Accuracy : 96.12%
Macro F1 : 77.76%
```

라는 높은 성능을 얻었습니다.

그러나 데이터 구조를 추가로 분석하고,

- lot 단위 split
- conflicting label 제거
- exact duplicate 탐지
- Train/Test processed input overlap 제거
- strict unseen-input test

를 적용한 이후 모델의 Macro F1은 크게 낮아졌습니다.

이 과정은 **모델 성능 자체만큼 평가 데이터의 독립성을 검증하는 것이 중요하다**는 점을 보여줍니다.

특히 심한 class imbalance를 가진 데이터에서는 Accuracy가 모델 성능을 오해하게 만들 수 있으며, Macro F1과 class별 precision/recall을 함께 확인해야 한다는 점을 확인했습니다.

또한 여러 seed에서 결과가 크게 달라지는 것을 통해 단일 실험 결과만으로 모델 성능을 판단하기보다 **반복 실험과 분산 확인이 필요함**을 확인했습니다.

---

## 15. Project Pipeline

```text
WM-811K
   │
   ▼
Label Cleaning
   │
   ▼
Duplicate / Conflict Audit
   │
   ├── Conflicting raw maps 제거
   │
   ▼
Official Train / Test Split
   │
   ├── Lot overlap 확인
   │
   ▼
Train 내부 Lot-disjoint Validation Split
   │
   ▼
Train/Test Exact Input Overlap 제거
   │
   ▼
Aspect-ratio Preserving Preprocessing
   │
   ▼
One-hot Encoding
   │
   ▼
CNN Training
   │
   ├── Cross Entropy
   └── Weighted Cross Entropy
   │
   ▼
3 Random Seeds Evaluation
   │
   ▼
Validation Macro F1 기반 Method Selection
   │
   ▼
Strict Unseen-input Test Evaluation
```

---

## 16. Environment

노트북 실행 환경에서 사용한 주요 라이브러리:

```text
Python
PyTorch
NumPy
Pandas
Matplotlib
scikit-learn
```

GPU가 사용 가능한 경우 CUDA를 자동으로 사용합니다.

```python
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
```

실험 당시 노트북에서는 PyTorch + CUDA 환경을 사용했습니다.

---

## 17. How to Run

### 1. Clone Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn torch
```

### 3. Prepare Dataset

WM-811K의 `LSWMD.pkl` 파일을 준비한 후 notebook의 dataset path를 수정합니다.

```python
PKL_PATH = "/path/to/LSWMD.pkl"
```

Google Colab을 사용할 경우 예시는 다음과 같습니다.

```python
PKL_PATH = "/content/drive/MyDrive/LSWMD.pkl/LSWMD.pkl"
```

### 4. Run Notebook

Jupyter Notebook 또는 Google Colab에서 notebook을 순서대로 실행합니다.

최종 confirmatory experiment는 다음 결과들을 생성합니다.

```text
wm811k_final_results/
```

최종 summary는 다음 파일로 저장됩니다.

```text
wm811k_final_results/FINAL_SUMMARY.csv
```

---

## 18. Limitations

현재 프로젝트에는 다음과 같은 한계가 있습니다.

1. **Simple CNN architecture**

   모델 구조 자체는 비교적 단순하며 ResNet 등의 더 강력한 architecture와 비교하지 않았습니다.

2. **Extreme class imbalance**

   `Near-full`, `Donut`과 같은 minority class의 학습 데이터가 매우 적습니다.

3. **Seed sensitivity**

   동일한 방법에서도 random seed에 따라 Test 성능 차이가 크게 나타났습니다.

4. **Domain shift**

   Official Training과 unseen Test lot 사이의 distribution 차이가 존재할 가능성이 있으며 일부 class의 일반화 성능이 낮습니다.

5. **Single dataset**

   WM-811K 하나만을 이용했기 때문에 다른 semiconductor manufacturing dataset에서의 일반화 여부는 확인하지 않았습니다.

---

## 19. Future Work

향후에는 다음 방향으로 프로젝트를 확장할 수 있습니다.

- ResNet 등 stronger CNN backbone과 비교
- Focal Loss 적용
- Class-balanced Loss 비교
- Minority class augmentation
- wafer-map 특성에 맞는 augmentation 설계
- Group-aware cross validation
- Confusion matrix 기반 error analysis
- class별 failure case visualization
- 여러 seed 기반 confidence interval 분석
- embedding / feature visualization
- model explainability 적용
- unseen lot에 대한 domain generalization 개선

---

## 20. Conclusion

본 프로젝트에서는 WM-811K wafer map의 9개 defect pattern을 CNN으로 분류하고, class imbalance를 완화하기 위해 Weighted Cross Entropy를 적용했습니다.

그러나 프로젝트의 핵심은 단순히 높은 분류 성능을 얻는 것이 아니라 **데이터 분할 방법과 중복 데이터가 모델 평가에 미치는 영향을 직접 확인했다는 점**입니다.

초기 random split에서는 높은 성능을 얻었지만, lot과 input overlap을 엄격하게 통제한 unseen test 환경에서는 성능이 크게 낮아졌습니다.

이를 통해 다음을 확인했습니다.

> **신뢰할 수 있는 머신러닝 모델을 만들기 위해서는 모델 architecture뿐만 아니라 데이터 분할, leakage 검증, 평가 지표 설계가 중요하다.**

특히 불균형한 실제 산업 데이터에서는 단순 Accuracy보다 **Macro F1, class별 recall, 데이터 독립성**을 함께 고려해야 합니다.