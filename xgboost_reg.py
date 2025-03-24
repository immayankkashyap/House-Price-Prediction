import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np

X = pd.read_csv('train_set.csv')

y_train = X['Total_Sales']
X_train = X.drop(['Total_Sales','Product_Code'], axis=1)
X_test = pd.read_csv('test_set.csv')
X_test = X_test.drop(['Product_Code'], axis= 1)
# y_test = pd.read_csv('kaggle_submission.csv')
# y_test = y_test.drop(['Id'],axis=1)

num_imputer = SimpleImputer(strategy='median')
X_train[['Product_Weight', 'Year_Opened']] = num_imputer.fit_transform(X_train[['Product_Weight', 'Year_Opened']])
X_test[['Product_Weight', 'Year_Opened']] = num_imputer.transform(X_test[['Product_Weight', 'Year_Opened']])  # Change to transform


categorical_cols = ['Store_Code', 'Store_Category', 'Location_Class', 'Store_Size'] 
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])  # Change to transform

# One-Hot Encoding
X_train_encoded = pd.get_dummies(X_train, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, drop_first=True)


X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)


# model = XGBRegressor(objective ='reg:squarederror', n_estimators = 10, seed = 123)
model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=492,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.7, # it decides the amount of data(rows) to be used for training // here 70%
    colsample_bytree=0.9, # it decides amount of columns to be used for training// here 90%
    gamma=0.1,  # Example addition
    min_child_weight=1,
    seed=123
)

cv_scores = -1 * cross_val_score(model, X_train_encoded, y_train, cv=5, scoring='neg_mean_squared_error')
cv_mse = np.mean(cv_scores)
print(f"Cross-validated MSE: {cv_mse}")

model.fit(X_train_encoded, y_train)

predict = model.predict(X_test_encoded)
# print(predict)
print(np.mean(predict))
'''
mse = mean_squared_error(y_test, predict)  # y_test is the true target values for X_test_encoded
print(f"Mean Squared Error: {mse}")
'''

Id = X_test['Id']

submission = pd.DataFrame({
    'Id': Id,
    'Total_Sales': predict
})

submission.to_csv('xgboost_submissions.csv', index=False)

# pd.DataFrame(predict, columns=['Predicted_Total_Sales']).to_csv('linear_predictions.csv', index=False)



print("Task completed!")

#1156291.3117051776
#2276.8416




