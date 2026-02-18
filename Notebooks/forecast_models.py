
# -------------------------------
# forecast_models.py
# -------------------------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from google.colab import drive
drive.mount('/content/drive')

# LOAD DATA
df_food = pd.read_csv("/content/drive/MyDrive/CHAKULA DATA SET/food_data.csv")
df_sales = pd.read_csv("/content/drive/MyDrive/CHAKULA DATA SET/sales_data.csv")
df_consumers = pd.read_csv("/content/drive/MyDrive/CHAKULA DATA SET/consumers_data.csv")
df_sellers = pd.read_csv("/content/drive/MyDrive/CHAKULA DATA SET/sellers_data.csv")

def load_data(food_path, sales_path, consumers_path=None, sellers_path=None):
    df_food = pd.read_csv(food_path)
    df_sales = pd.read_csv(sales_path)
    df_consumers = pd.read_csv(consumers_path) if consumers_path else None
    df_sellers = pd.read_csv(sellers_path) if sellers_path else None
    return df_food, df_sales, df_consumers, df_sellers

def prepare_sales_data(df_sales, df_food):
    df_sales['sale_date'] = pd.to_datetime(df_sales['sale_date'])
    df_sales['year'] = df_sales['sale_date'].dt.year
    df_sales['month'] = df_sales['sale_date'].dt.month
    sales_with_price = df_sales.merge(df_food[['food_id', 'seller_id', 'price']], on='food_id', how='left')
    sales_with_price['total_price'] = sales_with_price['quantity_sold'] * sales_with_price['price']
    return sales_with_price

def top_products(df_sales, n=10):
    return df_sales.groupby('food_id')['quantity_sold'].sum().sort_values(ascending=False).head(n)

def category_demand(df_sales, df_food):
    merged = df_sales.merge(df_food[['food_id', 'category']], on='food_id', how='left')
    return merged.groupby('category')['quantity_sold'].sum().sort_values(ascending=False)

def seller_revenue(df_sales, df_food):
    merged = df_sales.merge(df_food[['food_id', 'seller_id', 'price']], on='food_id', how='left')
    merged['total_price'] = merged['quantity_sold'] * merged['price']
    return merged.groupby('seller_id')['total_price'].sum().reset_index()

def prepare_ml_data(sales_with_price):
    # Monthly
    monthly_seller_revenue = sales_with_price.groupby(['year', 'month', 'seller_id'])['total_price'].sum().reset_index()
    monthly_seller_revenue['month_index'] = range(1, len(monthly_seller_revenue)+1)
    monthly_seller_ml_data = {}
    for seller_id in monthly_seller_revenue['seller_id'].unique():
        seller_data = monthly_seller_revenue[monthly_seller_revenue['seller_id']==seller_id]
        X_seller = seller_data[['month_index']]
        y_seller = seller_data['total_price']
        if len(X_seller) > 1:
            X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_seller, y_seller, test_size=0.2, shuffle=False)
        else:
            X_train_s = X_test_s = X_seller
            y_train_s = y_test_s = y_seller
        monthly_seller_ml_data[seller_id] = {'X_train': X_train_s, 'X_test': X_test_s,
                                             'y_train': y_train_s, 'y_test': y_test_s}
    # Daily
    daily_seller_revenue = sales_with_price.groupby(['sale_date','seller_id'])['total_price'].sum().reset_index()
    daily_seller_revenue['day_index'] = range(1,len(daily_seller_revenue)+1)
    daily_seller_ml_data = {}
    for seller_id in daily_seller_revenue['seller_id'].unique():
        seller_data = daily_seller_revenue[daily_seller_revenue['seller_id']==seller_id]
        X_seller = seller_data[['day_index']]
        y_seller = seller_data['total_price']
        if len(X_seller) > 1:
            X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_seller, y_seller, test_size=0.2, shuffle=False)
        else:
            X_train_s = X_test_s = X_seller
            y_train_s = y_test_s = y_seller
        daily_seller_ml_data[seller_id] = {'X_train': X_train_s, 'X_test': X_test_s,
                                           'y_train': y_train_s, 'y_test': y_test_s}
    return monthly_seller_ml_data, daily_seller_ml_data

def train_forecast(data, index_col, forecast_days=1):
    model = LinearRegression()
    model.fit(data['X_train'], data['y_train'])
    y_pred = model.predict(data['X_test'])
    mae = mean_absolute_error(data['y_test'], y_pred)
    last_index = data['X_train'][index_col].max() if len(data['X_train'])>0 else 0
    future = pd.DataFrame({index_col: range(last_index+1,last_index+1+forecast_days)})
    forecast = model.predict(future)
    return model, mae, forecast

def forecast_all_sellers(monthly_seller_ml_data, daily_seller_ml_data):
    monthly_forecasts = {}
    daily_forecasts = {}
    for seller_id, data in monthly_seller_ml_data.items():
        model, mae, forecast = train_forecast(data,'month_index',forecast_days=1)
        monthly_forecasts[seller_id] = {'model':model,'mae':mae,'next_month_forecast':forecast}
    for seller_id, data in daily_seller_ml_data.items():
        model, mae, forecast = train_forecast(data,'day_index',forecast_days=7)
        daily_forecasts[seller_id] = {'model':model,'mae':mae,'next_7_days_forecast':forecast}
    return monthly_forecasts,daily_forecasts

def forecast_seller(monthly_forecasts,daily_forecasts,seller_id):
    if seller_id not in monthly_forecasts or seller_id not in daily_forecasts:
        print(f"Seller {seller_id} data not available.")
        return None
    return {'monthly_forecast':monthly_forecasts[seller_id]['next_month_forecast'],
            'monthly_mae':monthly_forecasts[seller_id]['mae'],
            'daily_forecast':daily_forecasts[seller_id]['next_7_days_forecast'],
            'daily_mae':daily_forecasts[seller_id]['mae']}
