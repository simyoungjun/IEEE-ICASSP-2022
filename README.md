# SPCUP2022: Speech Spoof Detection Challenge

The **SPCUP2022 Speech Spoof Detection Challenge**, part of the IEEE Signal Processing Cup, focused on **Synthetic Speech Attribution**. Participants were tasked with identifying the specific synthesis method used to generate a given synthetic speech recording, a critical step in ensuring the security of speech-related technologies.

---

## 🔍 Challenge Overview

The challenge aimed to develop systems capable of distinguishing between various speech synthesis methods, including:
- **Text-to-Speech (TTS)** techniques.
- **Voice Conversion (VC)** methods.

Participants were provided with audio recordings generated using multiple synthesis methods. The evaluation focused on the system's ability to correctly classify the synthesis technique, particularly in challenging, perturbed conditions.

---

## 📂 Dataset

The dataset for this challenge can be accessed at the following link:  
[SPCUP2022 Dataset](https://doi.org/10.34740/kaggle/dsv/2866458)

---

## 📄 Detailed Report
For more detailed information about the dataset, approach, and results, refer to the official report included in this repository:  
[SPCUP2022 Report (PDF)](2022_SPCUP_report.pdf)


## 📊 Results

Our method achieved **4th place** in the SPCUP2022 challenge, demonstrating robust performance in both clean and noisy conditions:
- **Part 1 (Clean Data)**: Achieved **96.5% accuracy** on the evaluation set.
- **Part 2 (Noisy Data)**: Achieved **95.5% accuracy** under noisy and distorted conditions.

---

## References

1. **Official Challenge Page**:  
   [IEEE SPCUP2022](https://signalprocessingsociety.org/community-involvement/ieee-signal-processing-cup-2022)  
   - Provides official details about the challenge, including objectives, dataset composition, evaluation criteria, and participation guidelines.

2. **Paper**:  
   [Syn-Att: Synthetic Speech Attribution via Semi-Supervised Unknown Multi-Class Ensemble of CNNs](https://arxiv.org/abs/2309.08146)  
   - Describes a CNN-based approach combined with semi-supervised learning techniques for synthetic speech classification, specifically tailored for SPCUP2022.
