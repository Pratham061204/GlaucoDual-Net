# GlaucoDual-Net
A Two-Pronged Explainable AI Framework for Optic Disc–Cup Segmentation and Multi-Class Glaucoma Screening
## Overview

**GlaucoDual-Net** is an end-to-end, clinically aligned deep learning framework for **automated glaucoma screening** from retinal fundus images.  
Unlike conventional binary systems, this project explicitly models **three clinically meaningful categories**:

- **Healthy**
- **Glaucoma Suspected**
- **Glaucoma**

The framework combines:

- **Anatomical interpretability** via optic disc–cup segmentation  
- **Multi-class risk stratification** via deep image classification  
- **Explainable AI (XAI)** using **Grad-CAM** and **SHAP**

---

## Motivation

Glaucoma is a leading cause of irreversible blindness, with early stages often remaining asymptomatic.  
The **glaucoma suspect** category is particularly challenging due to subtle structural changes and overlapping visual cues.

Most prior work:

- Reduces screening to **binary classification**
- Lacks **anatomical transparency**
- Offers limited **clinical interpretability**

**GlaucoDual-Net** addresses these gaps by unifying **segmentation**, **classification**, and **explainability** in a single, clinically grounded pipeline.

---

## 🗂 Dataset

- **Dataset:** Standardized Multi-Channel Glaucoma Dataset (**SMDG-19**)  
- **Images:** 12,449 color fundus images  
- **Sources:** 19 public datasets    

The dataset is **highly heterogeneous** and **class-imbalanced**, reflecting real-world glaucoma screening conditions.

---

##  Framework Architecture

### 1️⃣ Optic Disc–Cup Segmentation

- **Model:** Custom U-Net (from scratch)  
- **Input:** 256 × 256 fundus images  
- **Output:**
  - Background  
  - Optic Disc  
  - Optic Cup  
<img width="1797" height="687" alt="unet" src="https://github.com/user-attachments/assets/bd074096-b611-4da6-bf66-c6c2f839b934" />



**Purpose:**
- Anatomical verification  
- Cup-to-Disc Ratio (CDR) estimation   

**Performance (Test Set):**
- Dice: **0.8843**
- IoU: **0.8031**
- Pixel Accuracy: **99.51%**

---

### 2️⃣ Multi-Class Glaucoma Classification

- **Backbone:** EfficientNetV2-M (timm)  
- **Input:** 512 × 512 RGB fundus images  
- **Loss:** Multi-class Focal Loss (γ = 2.0)  
- **Validation:** Stratified 5-fold cross-validation  

**Performance (OOF):**
- Accuracy: **89%**
- One-vs-Rest AUROC: **0.945**
- Weighted F1-score: **0.89**

## Explainable AI (XAI)

To validate clinical trust, **post-hoc explainability** is applied to the classification model.

---

### Grad-CAM


Grad-CAM highlights **class-discriminative regions** influencing model predictions.
<img width="1389" height="394" alt="GRAD1" src="https://github.com/user-attachments/assets/7e6cf85d-eb61-4852-b17d-1577b57877ad" />

**Observation:**  
Model attention consistently concentrates around the **optic nerve head**, aligning with established ophthalmic biomarkers.

---

### SHAP (Paper-Style Attribution Maps)

SHAP visualizations provide **class-wise evidence attribution**, illustrating how different regions support or oppose each diagnostic class.

<img width="1169" height="348" alt="SHAP2" src="https://github.com/user-attachments/assets/d8e92546-05e4-4ca8-9296-01e557dcefb5" />


- **Red** → Positive contribution  
- **Blue** → Negative contribution  
- **Block-wise visualization** for improved interpretability

## Setup & Requirements

Install the required dependencies using:

```bash
pip install torch torchvision timm shap opencv-python matplotlib numpy pandas
jupyter notebook Segmentation.ipynb
jupyter notebook Classification_Glaucoma.ipynb
```
## Future Work

- Integration of segmentation-derived biomarkers (e.g., CDR) into classification  
- Joint multi-task learning for segmentation and screening  
- Extension to longitudinal glaucoma progression analysis  



