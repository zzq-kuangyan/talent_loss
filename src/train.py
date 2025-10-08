import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import  accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,classification_report
import joblib

logging.info('123')
data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')
def feature_engineering(data):
    result = data.copy(deep=True)
    x = result[['Age','BusinessTravel','DistanceFromHome','EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','MonthlyIncome','NumCompaniesWorked','OverTime','RelationshipSatisfaction','TrainingTimesLastYear','WorkLifeBalance','YearsSinceLastPromotion','YearsWithCurrManager']]
    y = result['Attrition']

    x_dummies = pd.get_dummies(x,columns=['BusinessTravel'])
    #OverTime列数据映射
    def dataMapping(data):
        if data == 'Yes':
            return 1
        else:
            return 0
    x_dummies['OverTime']=x_dummies.OverTime.map(dataMapping)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_dummies)


    return x_train,y


x_train,y_train = feature_engineering(data)
# pd.DataFrame(x_train).boxplot()
# pyplot.show()
x_test,y_test = feature_engineering(test_data)

def CreatModel(x,y):
    model = LogisticRegression()
    model.fit(x_train,y_train)
    joblib.dump(model,'../model/LogisticRegression.pkl')

    return 'Sucess!'

CreatModel(x_train,y_train)