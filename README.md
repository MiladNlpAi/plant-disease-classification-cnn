# 🌱 Plant Disease Classification with CNN

This project implements a Convolutional Neural Network (CNN) to classify **plant leaf diseases** using the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset).  
The model is trained to recognize multiple disease categories and can predict new images provided by the user.

---

## 📌 Features
- Data preprocessing and resizing to `128x128`
- Train/Validation/Test split with stratification
- CNN model with **Batch Normalization** and **Dropout**
- **Early Stopping** and **Model Checkpointing**
- Training & Validation performance plots
- Custom image prediction support

---

## ⚠️ Important Note
Due to **hardware and computational limitations**, this project focuses only on **3 plant types**:
- 🍒 **Cherry** (healthy / powdery mildew)  
- 🍇 **Grape** (healthy / black measles / black rot / leaf blight)  
- 🌽 **Corn** (healthy / northern leaf blight / gray leaf spot / common rust)  

This subset was chosen to demonstrate the methodology. The same pipeline can be extended to the **full PlantVillage dataset** if more resources are available.

---

## ⚙️ Installation
Clone the repository:
```bash
git clone https://github.com/MiladNlpAi/plant-disease-classification-cnn.git
cd plant-disease-classification-cnn



  🛠️ Requirements

See requirements.txt


Model achieves good accuracy on unseen test data (subset of PlantVillage).






