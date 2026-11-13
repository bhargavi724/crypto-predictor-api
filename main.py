from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "Crypto Predictor API is running. Use /predict/BTC to get predictions."}

@app.get("/predict/{ticker}")
def predict_crypto(ticker: str):
    try:
        symbol = ticker.upper() if "-" in ticker else f"{ticker.upper()}-USD"
        df = yf.download(symbol, period="2y", interval="1d", progress=False)
        
        if df.empty or len(df) < 30:
            raise HTTPException(status_code=404, detail="Cryptocurrency symbol not found.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df["Target"] = df["Close"].shift(-1)
        features = ["Open", "High", "Low", "Close", "Volume"]
        
        latest_data = df[features].iloc[-1:]
        train_df = df.dropna()

        X = train_df[features]
        y = train_df["Target"]

        # 1. Split chronologically (80% train, 20% test) to evaluate accuracy
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        eval_model = RandomForestRegressor(n_estimators=100, random_state=42)
        eval_model.fit(X_train, y_train)
        test_preds = eval_model.predict(X_test)

        r2 = float(r2_score(y_test, test_preds))
        mae = float(mean_absolute_error(y_test, test_preds))
        mape = float(mean_absolute_percentage_error(y_test, test_preds)) * 100
        directional_accuracy = max(0.0, min(100.0, 100.0 - mape))

        # 2. Train final model on 100% of available data for tomorrow's prediction
        final_model = RandomForestRegressor(n_estimators=100, random_state=42)
        final_model.fit(X, y)

        # Extract scalar values safely using [0] and .item()
        predicted_price = float(final_model.predict(latest_data)[0])
        current_price = float(df["Close"].iloc[-1].item())

        # 3. Prepare last 30 days for the frontend chart
        recent_df = df.tail(30)
        history = [
            {"date": idx.strftime("%Y-%m-%d"), "price": round(float(val), 4)}
            for idx, val in zip(recent_df.index, recent_df["Close"])
        ]

        return {
            "symbol": symbol,
            "current_price": round(current_price, 4),
            "predicted_next_close": round(predicted_price, 4),
            "trend": "BULLISH" if predicted_price > current_price else "BEARISH",
            "metrics": {
                "r2_score": round(r2, 4),
                "mae": round(mae, 2),
                "mape_percent": round(mape, 2),
                "accuracy_percent": round(directional_accuracy, 2)
            },
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
