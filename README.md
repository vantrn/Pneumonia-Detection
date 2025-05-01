# Pneumonia Detection from Chest X-rays 🫁

This project uses a Convolutional Neural Network (CNN) to detect pneumonia from chest X-ray images. The model is trained on a labeled dataset and classifies images as either **NORMAL** or **PNEUMONIA**.

The main goals of this project are:
- Build and train a CNN using TensorFlow/Keras
- Evaluate predictions on test images
- Visualize model output using matplotlib
- Eventually implement Grad-CAM to understand which parts of the image the model focuses on

---

## 🚧 Status: Work In Progress

This project is still in progress. Currently, the model can make predictions, but:
- Grad-CAM is not yet implemented
- Model performance may be affected by dataset imbalance
- Improvements and testing are ongoing

---

## ✅ How to Run

1. Clone the repo and set up a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt

2. Download and extract the Chest X-Ray Pneumonia dataset into the data/ folder
[Text to display](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

3. Train the model
```bash
python main.py
```

4. Predict with test images
```
python evaluate.py
```
