import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
# from sklearn.metrics import mean_squared_error


X = pd.read_csv('train_set.csv')

y_train = X['Total_Sales']
X_train = X.drop(['Total_Sales','Product_Code'], axis=1)
X_test = pd.read_csv('test_set.csv')
X_test = X_test.drop(['Product_Code'], axis= 1)

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

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_encoded, y_train, test_size=0.2, random_state=123)


model = XGBRegressor(
    objective='reg:squarederror',
    seed=123
)

my_grid = {
    'n_estimators' : [490,491,492,493],
    'learning_rate': [0.01],
    'max_depth': [3,4,5,6],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.8, 0.9, 1],
    'gamma' : [0.05,0.1,0.2]
    
}
print('Grid search started..')
grid_search = GridSearchCV(estimator=model, param_grid=my_grid  , cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search.fit(X_train_encoded, y_train)

best_params = grid_search.best_params_
print(f"Best hyperparameters: {best_params}")
best_model = grid_search.best_estimator_
print('grid search completed..')
cv_scores = -1 * cross_val_score(best_model, X_train_encoded, y_train, cv=5, scoring='neg_mean_squared_error')
cv_mse = np.mean(cv_scores)
print(f"Cross-validated MSE: {cv_mse}")

best_model.fit(X_train_encoded, y_train)

predict = best_model.predict(X_test_encoded)
# print(predict)
print(f'Predicted mean value: {np.mean(predict)}')

'''mse = mean_squared_error(y_test, predict)  # y_test is the true target values for X_test_encoded
print(f"Mean Squared Error: {mse}")'''

# Id = X_test['Id']

# submission = pd.DataFrame({
    # 'Id': Id,
    # 'Total_Sales': predict
# })

# submission.to_csv('xgboost_submissions.csv', index=False)

# pd.DataFrame(predict, columns=['Predicted_Total_Sales']).to_csv('linear_predictions.csv', index=False)



print("Task completed!")

#Cross-validated MSE: 1156149.8839077116
#2280.2437




