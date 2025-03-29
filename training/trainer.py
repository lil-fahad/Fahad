import argparse
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

def generate_features(df):
    df['MA_7'] = df['Close'].rolling(7).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).mean()))
    df['Volatility'] = df['Close'].pct_change().rolling(30).std()
    df.dropna(inplace=True)
    return df

def train_xgb(X_train, y_train, output_dir):
    preprocessor = ColumnTransformer([
        ('scaler', StandardScaler(), X_train.columns.tolist())
    ])
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb_model)
    ])
    param_grid = {
        'model__n_estimators': [100],
        'model__max_depth': [3, 5],
        'model__learning_rate': [0.01, 0.1]
    }
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    joblib.dump(best_model, os.path.join(output_dir, "omnimarket_xgb.pkl"))
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.pkl"))
    return best_model

def train_lstm(X_train, y_train, output_dir):
    sequence_length = 30
    X_seq = np.array([X_train[i-sequence_length:i] for i in range(sequence_length, len(X_train))])
    y_seq = y_train[sequence_length:]

    model = Sequential([
        LSTM(64, input_shape=(X_seq.shape[1], X_seq.shape[2]), return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(
        X_seq, y_seq,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=5)],
        verbose=0
    )
    model.save(os.path.join(output_dir, "omnimarket_lstm.h5"))
    return model

def main():
    parser = argparse.ArgumentParser(description="OmniMarket Trainer")
    parser.add_argument("--csv", required=True, help="CSV file path")
    parser.add_argument("--model", choices=["xgb", "lstm", "all"], default="all")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if not set(['Open', 'High', 'Low', 'Close', 'Volume']).issubset(df.columns):
        raise ValueError("CSV must contain: Open, High, Low, Close, Volume")

    df = generate_features(df)
    X = df[['Open', 'High', 'Low', 'Volume', 'MA_7', 'RSI', 'Volatility']]
    y = df['Close']

    # تقسيم البيانات 80/20
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    os.makedirs("models", exist_ok=True)

    if args.model in ["xgb", "all"]:
        print("✅ تدريب XGBoost...")
        model = train_xgb(X_train, y_train, "models")
        preds = model.predict(X_test)
        mae = np.mean(np.abs(preds - y_test))
        print(f"📊 MAE (XGB) على مجموعة الاختبار: {mae:.4f}")

    if args.model in ["lstm", "all"]:
        print("✅ تدريب LSTM...")
        model = train_lstm(X_train.values, y_train.values, "models")
        seq_len = 30
        X_test_seq = np.array([X_test.values[i-seq_len:i] for i in range(seq_len, len(X_test))])
        y_test_seq = y_test[seq_len:]
        preds = model.predict(X_test_seq).flatten()
        mae = np.mean(np.abs(preds - y_test_seq))
        print(f"📊 MAE (LSTM) على مجموعة الاختبار: {mae:.4f}")

    print("✅ تم حفظ النماذج في مجلد models/")

if __name__ == "__main__":
    main()
