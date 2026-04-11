def load_stock_data(symbol="AAPL", period="2y"):
    import yfinance as yf

    df = yf.download(symbol, period=period)

    # ✅ FIX MultiIndex columns (IMPORTANT)
    df.columns = df.columns.get_level_values(0)
    df.columns = [col.lower() for col in df.columns]

    df = df.reset_index()

    df = df.rename(columns={
        "date": "date",
        "open": "price_open",
        "high": "price_high",
        "low": "price_low",
        "close": "price_close",
        "volume": "volume"
    })

    df["symbol"] = symbol

    return df