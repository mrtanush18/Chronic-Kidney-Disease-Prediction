# Chronic Kidney Disease Prediction 
This project is a machine learning-based web application that predicts the presence of chronic kidney disease (CKD) using medical parameters. The system is built in three phases: feature selection, model training, and user interaction via a web interface.

## Project Overview
The goal of this project is to help individuals assess their risk of CKD early by leveraging machine learning algorithms. Users can input medical data and instantly receive a prediction based on trained models.

## How It Works
1. Feature Selection
Identifies the most relevant health attributes that impact CKD.

Reduces noise and improves model performance.

2. Model Training
Uses AdaBoost, K-Nearest Neighbors (KNN), and Random Forest algorithms.

Models are trained and tested using standard classification metrics.

The best-performing model is saved for prediction.

3. User Interface
Developed using HTML, CSS (Bootstrap), and JS for a smooth UI experience.

Backend (app.js) receives user input and invokes the trained model.

Returns prediction results in a user-friendly format.

## Testing the Project
Clone the repository:
```
git clone https://github.com/yourusername/ckd-prediction-app.git
cd ckd-prediction-app
```
Install dependencies:
```
pip install -r requirements.txt
```
Run the backend:
```
python app.py
```
Open your browser and go to:
```
http://localhost:5000
```
Enter sample input values and submit to get the CKD prediction.

## Machine Learning Models
Random Forest	

AdaBoost	

KNN	

## Tech Stack
Python (Pandas, Scikit-learn)

Flask 

HTML/CSS/JS
