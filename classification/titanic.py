import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

PATH_TRAIN = r"C:\repos\ds_projects\classification\train.csv"
PATH_TEST = r"C:\repos\ds_projects\classification\test.csv"

class AgeImputer(BaseEstimator, TransformerMixin):
    """FIXME"""
    def fit(self, X):
        return self
    def transform(self, X):
        pass
            
class AttributeAdder(BaseEstimator, TransformerMixin):
    """Adds Total Companions Attribute"""
    def fit(self, X):
        return self
    def transform(self, X):
        Total_Companions = X[:, 1] + X[:, 2]
        return np.c_[X, Total_Companions]

def showHead(data):
    print(data.info())
    return


def main():
    # loading data
    trainData = pd.read_csv(PATH_TRAIN)
    testData = pd.read_csv(PATH_TEST)
    showHead(trainData)
    
    # setting data indices as passenger ID
    trainData.set_index(["PassengerId"])
    testData.set_index(["PassengerId"])
    
    # splitting attributes into numerical/categorical
    num_attributes = ["Age", "SibSp", "Parch", "Fare"]
    cat_attributes = ["Pclass", "Sex", "Embarked"]
    
    # making pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("attribute_adder", AttributeAdder()),
        ("std_scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("oh_encoder", OneHotEncoder(sparse=False)),
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])
    
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attributes),
        ("cat", cat_pipeline, cat_attributes)
    ])
    
    # training data
    X_train = full_pipeline.fit_transform(trainData[num_attributes + cat_attributes])
    y_train = trainData["Survived"]
    
    # classifying data as survived or not
    forestClassifier = RandomForestClassifier(random_state=42)
    forestClassifier.fit(X_train, y_train)
    
    X_test = full_pipeline.transform(testData[num_attributes + cat_attributes])
    y_pred = forestClassifier.predict(X_test)
    
    forest_scores = cross_val_score(forestClassifier, X_train, y_train, cv=10)
    print(forest_scores.mean())
    
main()