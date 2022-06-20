from multiprocessing.spawn import prepare
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from pandas_datareader import test
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from pandas.plotting import scatter_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from scipy import stats

# transformer to add attributes to dataset

miles_idx, year_idx = 5, 3
class CombinedAttributeAdder(BaseEstimator, TransformerMixin): 
    """Adding combined attribute to dataset"""
    def __init__(self, add_miles_per_year=True):
        self.add_miles_per_year = add_miles_per_year
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.add_miles_per_year:
            miles_per_year = X[:, miles_idx] / X[:, year_idx]
            return np.c_[X, miles_per_year]

def splitTrain(df):
    """Splitting data into testing and training sets"""
    train_set, test_set = train_test_split(df, test_size=.2, random_state=42)
    return train_set, test_set
    
def visualize(df):
    """Matplotlib graph of data"""
    df.plot(kind="scatter", x="mileage", y="price", alpha=.2, c=df["price"],
            cmap=plt.get_cmap("gnuplot2"))
    df.plot(kind="scatter", x="year", y="price", c=df["price"], 
            cmap=plt.get_cmap("gnuplot"), alpha=.2)
    plt.show()
    
def correlations(df):
    """Shows correlation matrix of data"""
    corr_matrix = df.corr()
    print(corr_matrix["price"].sort_values(ascending=False))
    
    attributes = ["price", "year", "lot", "mileage"]
    scatter_matrix(df[attributes], figsize=(12, 8))
    plt.show()
    

    
def clean(df):
    """Cleans car dataset"""
    # changing title status to boolean representing clean/salvage vehicle
    df["title_status"].replace({"clean vehicle": True, "salvage insurance": False}, inplace=True)
    
    # dropping unnecessary values that obviously have no correlation
    df.drop("vin", axis=1)
    df.drop("lot", axis=1)
    
def linear_regression(df_prepared, df_labels):
    lin_reg = LinearRegression()
    lin_reg.fit(df_prepared, df_labels)
    predictions = lin_reg.predict(df_prepared)
    lin_mse = mean_squared_error(df_labels, predictions)
    lin_rmse = np.sqrt(lin_mse)
    return lin_rmse
    
def decision_tree_regression(df_prepared, df_labels):
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(df_prepared, df_labels)
    predictions = tree_reg.predict(df_prepared)
    tree_mse = mean_squared_error(df_labels, predictions)
    tree_rmse = np.sqrt(tree_mse) # gives 0 error... need to validate
    
    # validation using 10 k-folds
    scores = cross_val_score(tree_reg, df_prepared, df_labels, scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    return tree_rmse_scores
    
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())

def random_forest_regression(df_prepared, df_labels):
    forest_reg = RandomForestRegressor()
    forest_reg.fit(df_prepared, df_labels)
    predictions = forest_reg.predict(df_prepared)
    forest_mse = mean_squared_error(df_labels, predictions)
    forest_rmse = np.sqrt(forest_mse)
    
    scores = cross_val_score(forest_reg, df_prepared, df_labels, scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-scores)
    return forest_rmse_scores


PATH = r"C:\Users\rtosh\OneDrive\Desktop\ML Book\Ch. 2\dataset\USA_cars_datasets.csv"
carData = pd.read_csv(PATH)
print(carData.info())
print(carData.head())
rand_train_set, rand_test_set = train_test_split(carData, test_size=.2, random_state=42)

# copying data from train set
carData = rand_train_set.copy()
carData.rename(columns={"Unnamed: 0": "index"}, inplace=True)
visualize(carData)
correlations(carData)

# getting data ready to be cleaned
carData = carData.drop("price", axis=1)
car_labels = rand_train_set["price"].copy()
clean(carData)

# splitting data into numerical and categorical data
car_num = carData[["year", "mileage"]]
car_cat = carData[["brand", "model", "title_status", "color", "state", "country", "condition"]]
num_attributes = list(car_num)
cat_attributes = list(car_cat)

# numerical pipeline/transform
num_pipeline = Pipeline([
    # ("attribs_adder", CombinedAttributeAdder()),
    ("std_scaler", StandardScaler())
])

# full_pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attributes),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attributes)
])

carData_prepared = full_pipeline.fit_transform(carData)


lin_rmse = linear_regression(carData_prepared, car_labels) # rmse of 5829.2
tree_scores = decision_tree_regression(carData_prepared, car_labels) # supposed rmse of 0... let's test that
# display_scores(tree_scores) # rmse of 8675... yeah that didn't work as planned
# forest_scores = random_forest_regression(carData_prepared, car_labels)
# display_scores(forest_scores) # rmse of 6765

# Grid Search setup
param_grid = [
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]}
    ]
    
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                            scoring="neg_mean_squared_error",
                            return_train_score=True)
grid_search.fit(carData_prepared, car_labels)

# Final model
final_model = grid_search.best_estimator_
X_test = rand_test_set.drop("price", axis=1)
y_test = rand_test_set["price"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

confidence = .95
squared_errors = (final_predictions - y_test) ** 2
final_ci = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, loc=squared_errors.mean()
                            , scale=stats.sem(squared_errors)))
print(final_ci)