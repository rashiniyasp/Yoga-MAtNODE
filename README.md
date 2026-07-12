# Yoga-MAtNODE: Multi-View Attention Neural ODE for Skeleton-Based Yoga Pose Recognition
**Accepted at ICPR 2026**

**Authors:** Rashi Niyas P, Hitika Tiwari, and Tushar Shinde

---

## 📌 Overview

Fine-grained yoga pose recognition is challenging due to subtle inter-class differences, self-occlusions, and varying camera viewpoints. **Yoga-MAtNODE** is a lightweight, skeleton-only framework that achieves state-of-the-art performance (89.17% on Yoga-82) with only **~75K parameters**.
<img width="1291" height="1118" alt="image" src="https://github.com/user-attachments/assets/b17c1ee5-dfc7-415e-b166-b5ec52c6cf42" />

Key components:
* **Implicit Multi-View Generation:** Dynamically rotates skeletons to 16 views to learn viewpoint-invariant features.
* **Continuous-Time Dynamics (Neural ODE):** Models joint trajectories as smooth latent dynamics, capturing subtle motion patterns.
* **Multi-View Attention:** Aggregates features across views to align semantic correspondences.

**PyTorch Implementation**

---

## 📂 Project Structure

```
.
├── data_preparation/           # Scripts to reproduce the dataset (ETL pipeline)
├── dataset.py                  # Implements Sec 3.1 & 3.2 (Normalization & Multi-View)
├── model.py                    # Implements Sec 3.3 & 3.4 (Neural ODE & Attention)
├── train.py                    # Main training loop with Label Smoothing
├── test.py                     # Evaluation & Confusion Matrix generation
├── util.py                     # Geometric feature extraction (Angles/Bones)
├── loss.py                     # Weighted Cross Entropy Loss
└── requirements.txt            # Python dependencies
```

---

## 🛠️ Installation

**Requirements:** Python 3.9+

Install the required Python packages:
```bash
pip install -r requirements.txt
```
**Key dependencies:** `torch`, `torchdiffeq`, `mediapipe==0.10.14`, `numpy<2`, `scikit-learn`.

---

## 💾 Data Availability (RRPR Compliance)

To reproduce the results, the datasets are publicly available. Download and extract them into a `datasets/` folder in the root directory:

* **Yoga-82 Dataset:** [Kaggle Link](https://www.kaggle.com/datasets/rashiniyasp/yoga-82-keypoints-2026/data)
* **Yoga-16 Dataset:** [Kaggle Link](https://www.kaggle.com/datasets/rashiniyasp/yoga-16-keypoint-dataset)

Structure after extraction:
```
datasets/
├── Yoga16/
│   ├── train/
│   ├── valid/
│   └── test/
└── Yoga82/
    ├── train/
    ├── validation/
    └── test/
```

---

## 🧠 Training

The model expects inputs representing rotated views of kinematic features.

To start training on the **Yoga-82** dataset:
```bash
python train.py --dataset_root datasets/Yoga82 --epochs 100 --batch_size 256
```

To train on the **Yoga-16** dataset from scratch:
```bash
python train.py --dataset_root datasets/Yoga16 --epochs 100 --batch_size 256
```
*(Note: We do not provide pretrained weights for Yoga-16, so you will need to train the model yourself to evaluate it).*

**Hyperparameters (as used in experiments):**
* Epochs: 100
* Batch size: 256
* Optimizer: AdamW (lr = 1e-3)
* LR schedule: Cosine Annealing

---

## 📊 Evaluation

To evaluate and reproduce the reported accuracy (89.17% on Yoga-82) and to generate a confusion matrix, use the testing script.

**Pretrained Weights (Yoga-82):** 
The pretrained model checkpoint (`yoga82_pretrained.pth`) is included directly in the root of this repository. No additional downloads are required.

For **Yoga-82** (Using the provided pretrained weights):
```bash
python test.py --dataset_root datasets/Yoga82 --model_path yoga82_pretrained.pth
```

For **Yoga-16** (Using weights you train yourself):
```bash
python test.py --dataset_root datasets/Yoga16 --model_path attention_yoga_node_final.pth
```

---

## 🔍 Performance Comparison

|           Method | Modality | Parameters | Yoga-82 Acc (%) |
| ---------------: | :------: | ---------: | --------------: |
|     DenseNet-201 |   Image  |      18.2M |           74.91 |
|          SYD-Net |   Image  |      33.5M |           97.29 |
|       Pose Tutor | Skeleton |        18M |           79.00 |
| **Yoga-MAtNODE** | Skeleton |   **~75K** |       **89.17** |

---

## 🔬 Methodology Mappings

* **Skeleton Normalization:** Implemented in `util.py` and `data_preparation/04_extract_skeletons.py` .
* **Kinematic Encoding:** Implemented in `util.py` .
* **Implicit Multi-View Generation:** Implemented in `dataset.py` .
* **Neural ODE Encoding:** Implemented in `model.py` .
* **Attention Aggregation:** Implemented in `model.py` .

---

## 📄 License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.
