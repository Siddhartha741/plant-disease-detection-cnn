# 🌿 Plant Disease Detection using CNN

This project aims to detect plant diseases using **Convolutional Neural Networks (CNN)** to assist farmers in diagnosing plant health from leaf images, improving agricultural productivity and food security.

## 👨‍💻 Authors

- **N. Siddhartha** – SR University  
- **K. Shiva Shankar** – SR University  
- **S. Rahul Sunny** – SR University  
---

## 📌 Table of Contents

- [Abstract](#abstract)
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Requirements](#requirements)
- [Proposed Solution](#proposed-solution)
- [System Architecture](#system-architecture)
- [Simulation Setup](#simulation-setup)
- [Implementation](#implementation)
- [Prediction](#prediction)
- [Results and Analysis](#results-and-analysis)
- [Learning Outcomes](#learning-outcomes)
- [Conclusion](#conclusion)
- [Future Scope](#future-scope)
- [References](#references)

---

## 📄 Abstract

The model uses image processing and **CNN (ResNet34)** for detecting 7 types of plant diseases. It was trained on a dataset of 8,685 labeled leaf images. Achieved an accuracy of **97.2%** and **F1-score > 96.5%**. The model is also deployed as a web app for real-time disease recognition.

---

## 📘 Introduction

Agriculture is essential for human survival. Plant diseases, caused by bacteria, fungi, or climate changes, reduce crop yield. Manual detection is slow, expensive, and often inaccurate. Deep learning automates disease detection accurately and efficiently, helping small-scale farmers.

---

## ⚠️ Problem Statement

Plant diseases impact food production and income. Manual methods are inefficient. An automated system for early detection is needed to reduce crop losses and ensure food security.

---

## 📋 Requirements

### Functional
- Real-time plant disease detection
- Simple web interface
- Regular updates to improve model

### Non-functional
- High accuracy
- Fast response (real-time)
- Scalable and reliable

### Technologies
- **Python**, **TensorFlow/Keras**, **OpenCV**, **NumPy**
- **Google Colab** for GPU support

---

## 💡 Proposed Solution

Train a CNN (ResNet34) to classify healthy vs diseased plant leaves. Include preprocessing (resize, normalize, augment) and evaluation using precision, recall, F1-score. Deploy the trained model into a user interface for real-time image uploads and detection.

---

## 🧱 System Architecture

- **Data Collection**: Gather diverse images of healthy and diseased leaves.
- **Preprocessing**: Normalize, resize, and augment images.
- **Model Training**: Fine-tune ResNet34 using transfer learning.
- **Evaluation**: Use confusion matrix, accuracy, F1-score.
- **Deployment**: Web application for real-time prediction.

---

## 🧪 Simulation Setup

- **Platform**: Google Colab
- **Language**: Python
- **Frameworks**: TensorFlow, PyTorch, OpenCV
- **Libraries**: Pandas, Matplotlib, Seaborn
- **Dataset**: Healthy and diseased leaf images (from Kaggle, FaceForensics++, etc.)

---

## 🔧 Implementation

1. **Data Gathering**: Combined various public datasets.
2. **Preprocessing**: Normalized, resized, augmented, and labeled.
3. **Model Training**: Used CNN layers, dropout, and softmax for multi-class classification.
4. **Model Evaluation**: Achieved over 97% validation accuracy.
5. **Testing**: Evaluated on unseen data using confusion matrix.

---

## 🤖 Prediction Features

- Binary (Healthy vs Diseased)
- Multi-class (Identify specific disease)
- Severity categorization
- Treatment recommendations
- Anomaly detection

---

## 📊 Results and Analysis

- **Accuracy**: 87.2%
- **F1-score**: >86.5%
- **Loss Curves**: Analyzed using training vs validation plots
- **Confusion Matrix**: Used to analyze false positives/negatives

---

## 🎯 Learning Outcomes

- Applied ML techniques on real data
- Understood CNN architecture
- Hands-on experience with Keras/TensorFlow
- Model evaluation and optimization
- Improved communication and collaboration skills
- Awareness of ethical ML usage

---

## 🧩 Conclusion

CNNs have proven effective in detecting plant diseases with high accuracy. The system can significantly assist farmers in diagnosing and managing crops. The project achieved 90%+ accuracy across all disease types.

---

## 🔮 Future Scope

- Mobile app for disease detection from camera images
- Drone-based aerial detection of large farms
- Automated pesticide recommendation system
- Real-time monitoring using IoT + AI integration

---

## 📚 References

1. Mohanty SP, Hughes DP. *Using Deep Learning for Image-Based Plant Disease Detection*. Front Plant Sci. 2016.
2. Ferentinos K. *Deep learning models for plant disease detection and diagnosis*. Comp & Elec in Agri. 2018.
3. Chohan M., Khan A. *Plant Disease Detection using Deep Learning*, IJRTET, 2020.
4. Various authors – Kaggle plant disease detection datasets, PyImageSearch, UpGrad blog, etc.

---

## 🧑‍💻 Sample Code Snippet

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

