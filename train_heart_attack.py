# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:53:48 2022

@author: ACER
"""

import os 
import math
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

PATH = os.path.join(os.getcwd(),'heart.csv')
MODEL_SAVE_PATH=os.path.join(os.getcwd(),'statics','model.h5')
MM_SACLER_SAVE_PATH=os.path.join(os.getcwd(),'statics','mm_scaler.pkl')

#%% EDA

#%% Step 1 Load data

df = pd.read_csv(PATH)

#%% Step 2 Data inspection

print(df.shape)
print(df.info())
print(df.describe().T)

# To remove duplicate data
df=df.drop_duplicates()

# To find the target values and plot the graph 
print(df['output'].value_counts())

df['output'].value_counts().plot(kind='bar')
plt.ylabel('Amount')
plt.title('Heart Diesease Value')
plt.show()

#%% Step 3 Data visualization

# To plot boxplot
df.boxplot()

#%% Step 4 Data cleaning

# No NaN data
df.isnull().sum()

# To plot correlation 
sns.heatmap(df.corr(),annot=True)
# Top 3 with highest relation are: 
# cp: 0.43
# thalachh: 0.42
# slp: 0.35          
            
#%% Step 5 Pre processing

X = df.drop('output',axis=1)
y = df['output']
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25,
                                                    random_state = 42)
# To scale the data with MinMax Scaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pickle.dump(scaler,open(MM_SACLER_SAVE_PATH,'wb'))  # save scaler into pickle

#%% Step 6 Machine learning pipeline

# To create pipeline list
steps_logis=[('Logis',LogisticRegression(solver='liblinear'))]
steps_knn=[('Logis',KNeighborsClassifier(n_neighbors=10))]
steps_forest=[('Forest',RandomForestClassifier(n_estimators=10))]
steps_svc=[('SVC',SVC())]
steps_tree=[('Tree',DecisionTreeClassifier())]
  
logis_pipeline=Pipeline(steps_logis)
knn_pipeline=Pipeline(steps_knn)
forest_pipeline=Pipeline(steps_forest)
svc_pipeline=Pipeline(steps_svc) 
tree_pipeline=Pipeline(steps_tree)

pipelines=[logis_pipeline,knn_pipeline,forest_pipeline,svc_pipeline,
           tree_pipeline]

# To fit the data in pipeline
for pipe in pipelines:
    pipe.fit(X_train,y_train)
    
pipe_dict = {0:'Logistic Regression', 1:'KNN', 2: 'Random Forest', 
             3: 'SVC',4: 'Decision Tree'}

# Print the accuracy score

for index,model in enumerate(pipelines):
    y_pred = model.predict(X_test)
    print("{} Accuracy Score: {}".format(pipe_dict[index],model.score(X_test, y_test)*100 ))
    #print(classification_report(y_test, y_pred))

# Conclusion: KNN has the highest accuracy score, 81.3%

#%% Save the best model

pickle.dump(knn_pipeline, open(MODEL_SAVE_PATH, 'wb'))
