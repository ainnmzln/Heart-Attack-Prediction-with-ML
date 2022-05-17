![badge](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

# Heart Attack Prediction with Machine Learning

# 1. Summary
The main objective of this project is to develop an app to predict the chance of a patient having heart attackwith machine learning model.

# 2. Datasets

This projects is trained with  [Heart Attack Analysis & Prediction Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset). The 'outcome' field refers to the presence of heart disease of the patient. It is integer valued 0 = no disease and 1 = disease.

# 3. Requirements
This project is created using Spyder as the main IDE. The main frameworks used in this project are Pandas, Matplotlib, Seaborn, Scikit-learn and Streamlit.

# 4. Methodology
The flow of the projects are as follows:

## 1. Importing the libraries and dataset

The data are loaded from the dataset and usefull libraries are imported.

## 2. Exploratory data analysis

The datasets is cleaned with necessary step. The duplicate is removed. The correlation between features are computed. 

![This is an image](https://github.com/ainnmzln/heart_attack_prediction_using_ML/blob/main/images/Figure%202022-05-17%20162035.png)

It is shown that chest pain (cp), maximum heart rate achieved (thalach) and slope (slp) have highest corrolation with target. 
The data are scaled with MinMax Scaler to refine the outliers. Next, the data is splitted into 70:30 train and test ratio. 

## 3. Machine learning model 

Few machine learning model suits for binary classfification problem are selected and built into the pipeline such as 

1. Logistic regression
2. K Neighbors Classifier
3. Random Forest Classifier
4. Support Vector Classifier
5. Decision Tree Classifier

## 4. Model Prediction and Accuracy

The results with the best accuracy score is K Neighbors Classifier with 84 % accuracy score. The classification report of the training is shown below. 

![](https://github.com/ainnmzln/heart_attack_prediction_using_ML/blob/main/images/acuracy%20score.png)

![](https://github.com/ainnmzln/heart_attack_prediction_using_ML/blob/main/images/report.png)

## 5. Deployment

The data is then tested with few cases.

## 6. Build the app using Streamlit

An app to predict the chance of a person to get heart attack is then build using Streamlit. 
![](https://github.com/ainnmzln/heart_attack_prediction_using_ML/blob/main/images/apps.png)
