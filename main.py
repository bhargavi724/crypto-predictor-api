from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app = FastAPI()

# Allow your Lovable website to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict/{ticker}")
def predict_crypto(ticker: str):
    try:
        # 1. Fetch live historical data from Yahoo Finance (e.g., BTC-USD)
        symbol = ticker.upper() if "-" in ticker else f"{ticker.upper()}-USD"
        df = yf.download(symbol, period="2y", interval="1d", progress=False)
        
        if df.empty or len(df) < 30:
            raise HTTPException(status_code=404, detail="Cryptocurrency symbol not found.")

        # Flatten multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 2. Prepare features (Today's Open, High, Low, Close, Volume -> Predict Tomorrow's Close)
        df["Target"] = df["Close"].shift(-1)
        features = ["Open", "High", "Low", "Close", "Volume"]
        
        latest_data = df[features].iloc[-1:] # Today's live data for prediction
        train_df = df.dropna()

        X = train_df[features]
        y = train_df["Target"]

        # 3. Train Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # 4. Predict tomorrow's price
        predicted_price = float(model.predict(latest_data)[0])
        current_price = float(df["Close"].iloc[-1])

        return {
            "symbol": symbol,
            "current_price": round(current_price, 4),
            "predicted_next_close": round(predicted_price, 4),
            "trend": "BULLISH" if predicted_price > current_price else "BEARISH"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))