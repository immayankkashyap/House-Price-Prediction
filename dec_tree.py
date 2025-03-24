import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.metrics import mean_squared_error

X = pd.read_csv('train_set.csv')

y_train = X['Total_Sales']
X_train = X.drop(['Total_Sales'], axis=1)
X_test = pd.read_csv('test_set.csv')
y_test = pd.read_csv('kaggle_submission.csv')
y_test = y_test.drop(['Id'],axis=1)

num_imputer = SimpleImputer(strategy='median')
X_train[['Product_Weight', 'Year_Opened']] = num_imputer.fit_transform(X_train[['Product_Weight', 'Year_Opened']])
X_test[['Product_Weight', 'Year_Opened']] = num_imputer.transform(X_test[['Product_Weight', 'Year_Opened']])  # Change to transform

categorical_cols = ['Store_Code', 'Store_Category', 'Location_Class', 'Store_Size'] 
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])  # Change to transform

X_train_encoded = pd.get_dummies(X_train, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, drop_first=True)

X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

model =DecisionTreeRegressor(random_state=1)

cv_scores = -1 * cross_val_score(model, X_train_encoded, y_train, cv=5, scoring='neg_mean_squared_error')
cv_mse = np.mean(cv_scores)
print(f"Cross-validated MSE: {cv_mse}")

model.fit(X_train_encoded, y_train)

predict = model.predict(X_test_encoded)
# print(predict)
print(np.mean(predict))
mse = mean_squared_error(y_test, predict)  # y_test is the true target values for X_test_encoded
print(f"Mean Squared Error: {mse}")

'''
Id = X_test['Id']
# Combine Id and predictions into one DataFrame for the submission file
submission = pd.DataFrame({
    'Id': Id,
    'Total_Sales': predict
})


# Save the combined DataFrame to CSV
submission.to_csv('xgboost_submissions.csv', index=False)

# pd.DataFrame(predict, columns=['Predicted_Total_Sales']).to_csv('linear_predictions.csv', index=False)

'''

print("Task completed!")