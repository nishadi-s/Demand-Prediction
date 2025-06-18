import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import uuid

def load_data():
    """Generate synthetic sales data."""
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    data = {
        'date': dates,
        'sales_quantity': 100 + np.random.normal(0, 10, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 20,
        'price': 10 + np.random.normal(0, 1, len(dates)),
        'temperature': 20 + np.random.normal(0, 5, len(dates)),
        'promotion': np.random.choice([0, 1], len(dates), p=[0.8, 0.2]),
        'holiday': np.random.choice([0, 1], len(dates), p=[0.95, 0.05])
    }
    df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    """Preprocess the dataset."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['sales_lag_1'] = df['sales_quantity'].shift(1).fillna(df['sales_quantity'].mean())
    df['sales_lag_7'] = df['sales_quantity'].shift(7).fillna(df['sales_quantity'].mean())
    return df

def select_features(df):
    """Select features and target."""
    features = ['price', 'temperature', 'promotion', 'holiday', 'day_of_week', 'month', 'year', 'is_weekend', 'quarter', 'sales_lag_1', 'sales_lag_7']
    X = df[features]
    y = df['sales_quantity']
    return X, y, features

def split_and_scale_data(X, y):
    """Split and scale data."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train_scaled, y_train):
    """Train Random Forest model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model

def evaluate_model(model, X_test_scaled, y_test):
    """Evaluate model."""
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared Score: {r2:.2f}")
    return y_pred, mse, r2

def predict_new_input(model, scaler, features, input_data):
    """Predict sales quantity for new input."""
    input_df = pd.DataFrame([input_data], columns=features)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    return prediction

def train_and_save_model():
    """Train model and return model, scaler, features."""
    df = load_data()
    df = preprocess_data(df)
    X, y, features = select_features(df)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(X, y)
    model = train_model(X_train_scaled, y_train)
    y_pred, mse, r2 = evaluate_model(model, X_test_scaled, y_test)
    return model, scaler, features