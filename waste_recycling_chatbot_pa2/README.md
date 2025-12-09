# Swiss Waste Recycling Assistant - PA2 Project

An AI-powered waste classification and advisory system tailored for strict compliance with Swiss recycling guidelines. This project integrates Computer Vision for image recognition with a Local LLM (Ollama) to provide accurate disposal instructions.

## Project Overview

This system automates waste sorting assistance by recognizing waste types from images and providing disposal advice based on authoritative **Swiss Recycle** rules.

* **Recognition:** Identifies **16 distinct waste categories** using Deep Learning.
* **Compliance:** Prevents "hallucinations" by adhering to strict disposal channels (e.g., distinguishing between PET collection points and general plastic disposal).
* **Performance:** Achieves **95.29% accuracy** on the test dataset.
* **Interface:** Features both a professional Web Dashboard and a CLI Chatbot.

**Context:**
* *Period:* September 2025 - December 2025
* *Course:* Projektarbeit 2 (PA2) / ZHAW Life Sciences und Facility Management, ICLS

---

## Dataset

### Source & Distribution
The project utilizes the "Helene waste dataset" augmented with custom collected images. The data was stratified to ensure balanced class distribution across splits.

* **Total Images:** 5,795
* **Train:** 4,056 images (70%)
* **Validation:** 869 images (15%)
* **Test:** 870 images (15%)

### Waste Categories (16 Classes)
The model is trained to recognize the following classes:
1.  Aluminium
2.  Brown Glass
3.  Cardboard
4.  Composite Carton
5.  Green Glass
6.  Hazardous Waste (Battery)
7.  Metal
8.  Organic Waste
9.  Paper
10. PET
11. Plastic
12. Plastic & Aluminium (Composite)
13. Residual Waste
14. Rigid Plastic Container
15. White Glass
16. White Glass & Metal

---

## Model Architecture & Performance

### Image Classifier
* **Base Model:** MobileNetV3-Large (Pretrained).
* **Custom Head:** Fine-tuned classifier with Linear, Hardswish, and Dropout layers for 16 output classes.
* **Training:** Trained for 23 epochs using Early Stopping to prevent overfitting.

### Results
* **Test Accuracy:** **95.29%**
* **Test Loss:** 0.7106
* **Perfect Classes (100% Accuracy):** Aluminium, Composite Carton, Batteries, Organic Waste, Paper, Plastic/Aluminium composite.
* **Challenging Classes:** Plastic (81.97%), White Glass (89.36%).

### LLM Integration (RAG-Style)
* **Engine:** Ollama (Local Inference).
* **Model:** `qwen2.5:7b-instruct`.
* **Compliance Logic:** The system injects classification confidence and a predefined `RECYCLING_GUIDE` into the system prompt. This enforces specific Swiss disposal rules (e.g., prohibiting curbside collection for PET bottles).

---

## Installation & Usage

### Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) installed and running locally.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Local LLM
Run the setup script to check system resources and pull the required Ollama models:
```bash
python setup_opensource_chatbot.py
```
*Recommended Model:* `qwen2.5:7b-instruct`

### 3. Run the Application
You can run the project in two modes:

**Option A: Web Dashboard (Recommended)**
A Bootstrap-based UI with image upload and chat history.
```bash
python FINAL_DASHBOARD.py
```

## Timeline

* **KW 38-42:** Research & Project Disposition
* **KW 43-44:** Dataset Preparation & Preprocessing
* **KW 45-47:** MobileNetV3 Fine-tuning & Evaluation
* **KW 48-49:** Ollama Integration, Prompt Engineering, and Dashboard Development
* **KW 50:** Final Documentation & Submission

## Credits

**Author:** Lejla Beganovic 
**Supervisor:** Martin Schüle

ZHAW Life Sciences und Facility Management Institut für Computational Life Sciences Schloss, 8820 Wädenswil