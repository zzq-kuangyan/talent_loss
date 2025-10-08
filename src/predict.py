import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,classification_report
from src import train

def Predict():
    model = joblib.load('../model/LogisticRegression.pkl')
    y_predict = model.predict(train.x_test)
    acc = 'accuracyRate:',accuracy_score(train.y_test,y_predict)
    cls_report = classification_report(train.y_test,y_predict)
    print(cls_report)

    return acc

print(Predict())
