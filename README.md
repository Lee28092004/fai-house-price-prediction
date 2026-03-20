# FAI House Price Prediction

## Project Overview
This project is developed for the Further Artificial Intelligence assignment.
The goal is to predict house prices using machine learning techniques based on housing features.

## Problem Statement
House price estimation is often inconsistent when based only on intuition or outdated asking prices.
This project applies machine learning to provide a more data-driven and objective price prediction approach.

## Objectives
- Preprocess and clean housing data
- Perform exploratory data analysis
- Select relevant features
- Train regression models
- Evaluate performance using MAE, RMSE, and R²
- Visualize model results
- Provide a reusable prediction prototype

## Dataset
The dataset used is `houses.csv`, which contains housing-related attributes such as:
- Bedroom
- Bathroom
- Property Size
- Category
- Tenure Type
- Completion Year
- Property Type
- Parking Lot
- price

## Folder Structure
```text
fai-house-price-prediction/
├── data/
│   └── houses.csv
├── models/
│   └── saved_model.pkl
├── notebooks/
│   └── eda.ipynb
├── outputs/
│   ├── actual_vs_predicted.png
│   ├── correlation_heatmap.png
│   └── feature_importance.png
├── src/
│   ├── evaluate.py
│   ├── predict.py
│   ├── preprocess.py
│   └── train.py
├── .gitignore
└── README.md