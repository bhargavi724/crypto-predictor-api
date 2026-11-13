from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        predicted_price = float(model.predict(latest_data))
        current_price = float(df["Close"].iloc[-1])

        # Prepare the last 30 days of prices for the frontend chart
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
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
