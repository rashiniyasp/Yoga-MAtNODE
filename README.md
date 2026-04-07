# Yoga-MAtNODE: Multi-View Attention Neural ODE for Skeleton-Based Yoga Pose Recognition
**Accepted at ICPR 2026**

**Authors:** Rashi Niyas P, Hitika Tiwari, and Tushar Shinde

---

## 📌 Overview

Fine-grained yoga pose recognition is challenging due to subtle inter-class differences, self-occlusions, and varying camera viewpoints. **Yoga-MAtNODE** is a lightweight, skeleton-only framework that achieves state-of-the-art performance (89.17% on Yoga-82) with only **~75K parameters**.
<img width="1231" height="770" alt="image" src="https://github.com/user-attachments/assets/21daabbb-7ccd-44cb-ae73-260bd935cc3c" />

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

Install the required Python packages:

```bash
pip install -r requirements.txt
```

**Key dependencies:** `torch`, `torchdiffeq`, `mediapipe`, `numpy`, `scikit-learn`.

---

## 🚀 Data Preparation

> To reproduce the experiments, strictly follow the data preparation pipeline. This converts raw images into the normalized 3D skeleton format required by Yoga-MAtNODE.

Run the scripts in order:

1. **Download Images**

```bash
python data_preparation/01_download_images.py
```

2. **Sanitize Directory Names**

```bash
python data_preparation/02_clean_names.py
```

3. **Generate Splits (Train/Val/Test)**

```bash
python data_preparation/03_organize_splits.py
```

4. **Extract 3D Skeletons (MediaPipe)**

> Extracts 33-joint skeletons, centers them, and normalizes scale.

```bash
python data_preparation/04_extract_skeletons.py
```

5. **Augment & Balance**

> Ensures 500 samples per class using kinematic augmentation.

```bash
python data_preparation/05_augment_balance.py
```

---

## 🧠 Training

The model expects inputs shaped `(Batch, 16, 212)`, representing 16 rotated views of 212-dimensional kinematic features (coordinates + angles + bone vectors).

To start training:

```bash
python train.py
```

**Hyperparameters (as used in experiments):**

* Epochs: 100
* Batch size: 256
* Optimizer: AdamW (lr = 1e-3)
* LR schedule: Cosine Annealing

---

## 📊 Evaluation

To evaluate and reproduce the reported accuracy (89.17% on Yoga-82) and to generate a confusion matrix:

```bash
python test.py
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


