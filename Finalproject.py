# -*- coding: utf-8 -*-
"""Final Project

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ejaVk3JrcBaE8lGU3qMo8ONwFKUr1xMz
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from flask import Flask, request, jsonify
from prometheus_client import start_http_server, Counter, Summary, CollectorRegistry
from pyngrok import ngrok
from flask import Flask
import streamlit as st


data = pd.read_csv("mock_fashion_data_uk_us.csv")

print(data.head())

print(data.info())

print(data.describe())

print(data.isnull().sum())

data = data.drop_duplicates()

z_score_threshold = 3
data = data[(data['Price'] - data['Price'].mean()) / data['Price'].std() < z_score_threshold]

data['Review Sentiment'] = data['Rating'].apply(lambda x: 'Positive' if x > 3 else 'Negative')

scaler = MinMaxScaler()
data['Price'] = scaler.fit_transform(data['Price'].values.reshape(-1, 1))

data.to_csv("preprocessed_data.csv", index=False)

# Box plot to compare the distribution of Ratings by Category
sns.boxplot(x='Category', y='Rating', data=data)
plt.title('Distribution of Ratings by Category')
plt.show()

# Histogram to visualize the distribution of Prices
sns.histplot(data['Price'], bins=10)
plt.title('Distribution of Prices')
plt.xlabel('Price')
plt.ylabel('Count')
plt.show()

# Bar plot to compare the Average Rating by Brand
avg_rating_by_brand = data.groupby('Brand')['Rating'].mean().reset_index()
sns.barplot(x='Brand', y='Rating', data=avg_rating_by_brand)
plt.title('Average Rating by Brand')
plt.xticks(rotation=90)
plt.show()

# Heatmap to visualize the correlation matrix of numerical variables
numerical_vars = ['Price', 'Rating', 'Review Count', 'Age']
corr_matrix = data[numerical_vars].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Variables')
plt.show()

# Bar plot to compare the count of each Category
sns.countplot(x='Category', data=data)
plt.title('Count of Each Category')
plt.xticks(rotation=90)
plt.show()

# Violin plot to compare the distribution of Ratings by Season
sns.violinplot(x='Season', y='Rating', data=data)
plt.title('Distribution of Ratings by Season')
plt.show()

# Scatter plot matrix to visualize the relationships between numerical variables
num_vars = ['Price', 'Rating', 'Review Count', 'Age']
sns.pairplot(data[num_vars])
plt.title('Scatter Plot Matrix of Numerical Variables')
plt.show()

# Stacked bar plot to compare the count of each Color by Category
color_counts = data.groupby(['Category', 'Color']).size().unstack()
color_counts.plot(kind='bar', stacked=True)
plt.title('Count of Each Color by Category')
plt.xticks(rotation=90)
plt.show()

# Prepare the data
X = data[['Price', 'Review Count']]
y = data['Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the linear regression model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Predict on the test set
y_pred = reg_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

param_grid = {
    'fit_intercept': [True, False]
}

model = LinearRegression()

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Retrieve the best model from the grid search
best_model = grid_search.best_estimator_

# Make predictions on the testing set
y_pred = best_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)

print('R-squared:', r2)

# Use the model for predictions on new data
new_data = pd.DataFrame({'Price': [50, 100, 150], 'Review Count': [200, 500, 1000]})
predictions = best_model.predict(new_data)

print('Predictions:', predictions)

# Load the preprocessed data
data = pd.read_csv("preprocessed_data.csv")

# Serialize and save the trained model
joblib.dump(reg_model, 'model.pkl')

import streamlit as st
import pandas as pd
import numpy as np

#  Define the Streamlit app
def predict_rating(price, review_count):
    # Make predictions using the trained model
    input_features = np.array([[price, review_count]])
    prediction = model.predict(input_features)
    return prediction[0]

def log_prediction(price, review_count, prediction):
    log_entry = f"Price: {price}, Review Count: {review_count}, Prediction: {prediction}\n"
    with open('predictions.log', 'a') as file:
        file.write(log_entry)

def main():
    st.title("Fashion Rating Prediction")
    st.write("Enter the price and review count to predict the rating.")

    price = st.number_input("Price", min_value=0.0)
    review_count = st.number_input("Review Count", min_value=0)

    if st.button("Predict"):
        # Perform prediction
        prediction = predict_rating(price, review_count)

        # Log the prediction
        log_prediction(price, review_count, prediction)

        # Display the prediction
        st.success(f"The predicted rating is: {prediction}")

if __name__ == "__main__":
    main()