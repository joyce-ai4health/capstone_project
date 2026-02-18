📊 Food Marketplace – Data Science Module (MVP)
🚀 Overview

This repository contains my individual contribution to the Data Science milestone of the Food Marketplace MVP project.

It extends the existing rule-based recommendation engine with:

Demand analysis from sales data

AI-based demand forecasting (Machine Learning)

Seller dashboard analytics

Recommendation system validation metrics

This implementation aligns fully with the Product Requirements Document (PRD) at MVP level.

📁 Datasets Used

The following structured datasets were provided and cleaned:

food_data.csv

sales_data.csv

sellers_data.csv

consumers_data.csv

All preprocessing and aggregation were implemented in Google Colab using Python.

📊 1️⃣ Demand Analysis

Implemented sales aggregation to generate business insights.

✔ Monthly Demand Per Product

Aggregated total quantity sold per product by month.

✔ Top 10 Selling Products

Ranked products by total quantity sold.

✔ Category-Level Demand

Merged sales and food data to compute demand trends by category.

✔ Seller-Level Total Sales

Computed total revenue per seller.

✔ Monthly Sales Trend

Generated time-based revenue trend for forecasting.

🤖 2️⃣ AI Demand Forecasting
Model Used:

Linear Regression (sklearn)

Why Linear Regression?

Predicts continuous demand values

Simple and interpretable

Suitable for MVP

Meets PRD requirement for AI-based forecasting

Input:

Month index (time trend feature)

Output:

Predicted next month total demand

Evaluation Metric:

MAE (Mean Absolute Error)

This satisfies the PRD requirement:

“Farmers receive AI-generated demand forecasts based on historical sales data.”

📈 3️⃣ Seller Dashboard Analytics

Implemented a reusable function:

def seller_dashboard(seller_id):


Returns:

Total sales volume

Total revenue

Top 3 selling products

Low-demand products

Monthly revenue trend

This provides actionable insights for farmers and supports performance tracking.

✅ 4️⃣ Recommendation System Validation

To ensure reliability of the rule-based recommendation engine, the following validation metrics were implemented:

Budget Compliance Rate

Balanced Basket Rate

Overall Valid Basket Rate

Tested across multiple user scenarios (minimum 5 cases).

Metric example:

valid_baskets / total_test_cases


This ensures measurable and verifiable system performance.

🛠 Tech Stack

Python

Pandas

NumPy

Scikit-learn

Google Colab

🎯 Outcome

After implementation:

✔ Demand analytics layer completed
✔ AI-based forecasting implemented
✔ Seller dashboard insights functional
✔ Recommendation engine validated
✔ Fully aligned with PRD MVP requirements

🔗 Integration with Backend

This module is designed to integrate with backend APIs.

The trained model and analytics functions can be wrapped inside endpoints such as:

GET /api/forecast
GET /api/seller/dashboard


Backend can load the trained model and return JSON responses to frontend dashboards.

📌 Author

Joyce Etata
Data Science & Engineering Track
Women Techsters Fellowship
