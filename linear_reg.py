# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.impute import SimpleImputer
# import numpy as np


# # Load data
# X = pd.read_csv('train_set.csv')

# # Separate target variable
# y_train = X['Total_Sales']
# X_train = X.drop(['Total_Sales'], axis=1)
# X_test = pd.read_csv('test_set.csv')
# #train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=0)

# # Imputation
# # Numeric imputer
# num_imputer = SimpleImputer(strategy='median')  # Or mean, based on distribution
# X_train[['Product_Weight', 'Year_Opened']] = num_imputer.fit_transform(X_train[['Product_Weight', 'Year_Opened']])
# X_test[['Product_Weight', 'Year_Opened']] = num_imputer.transform(X_train[['Product_Weight', 'Year_Opened']])

# # Categorical imputer
# categorical_cols = ['Store_Code', 'Store_Category', 'Location_Class', 'Store_Size'] 
# cat_imputer = SimpleImputer(strategy='most_frequent')
# X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
# X_test[categorical_cols] = cat_imputer.transform(X_train[categorical_cols])

# # One-hot encoding
# X_train_encoded = pd.get_dummies(X_train, drop_first=True)
# # During testing, use the same columns from training
# X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# # Align the columns by reindexing
# X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)


# model = LinearRegression()
# #model.fit(train_X,train_y)

# cv_scores = -1 * cross_val_score(model, X_train_encoded, y_train, cv=5, scoring='neg_mean_squared_error')
# cv_mse = np.mean(cv_scores)
# print(cv_mse)

# model.fit(X_train_encoded, y_train)

# predict = model.predict(X_test_encoded)
# print(predict)

# pd.DataFrame(predict, columns=['Predicted_Total_Sales']).to_csv('predictions.csv', index=False)



import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np

X = pd.read_csv('train_set.csv')

y_train = X['Total_Sales']
X_train = X.drop(['Total_Sales'], axis=1)
X_test = pd.read_csv('test_set.csv')

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

model = LinearRegression()
cv_scores = -1 * cross_val_score(model, X_train_encoded, y_train, cv=5, scoring='neg_mean_squared_error')
cv_mse = np.mean(cv_scores)
print(f"Cross-validated MSE: {cv_mse}")

model.fit(X_train_encoded, y_train)

predict = model.predict(X_test_encoded)
# print(predict)


Id = X_test['Id']
# Combine Id and predictions into one DataFrame for the submission file
submission = pd.DataFrame({
    'Id': Id,
    'Total_Sales': predict
})


# Save the combined DataFrame to CSV
submission.to_csv('submission3.csv', index=False)

# pd.DataFrame(predict, columns=['Predicted_Total_Sales']).to_csv('linear_predictions.csv', index=False)


print("Task completed!")






