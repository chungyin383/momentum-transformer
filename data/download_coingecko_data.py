import requests
from datetime import datetime
import pandas as pd
import os
from settings.default import TICKERS

MARKET_CHART_API = "https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
MARKET_CHART_PARAMS = {
    "vs_currency": "usd",
    "days": "max",
}


def main():

    if not os.path.exists(os.path.join("data", "raw_data")):
        os.mkdir(os.path.join("data", "raw_data"))

    # Fetch data for each ticker and save it as a separate CSV file
    for ticker in TICKERS:
        print(ticker)
        try:
            response = requests.get(MARKET_CHART_API.format(coin=ticker), params=MARKET_CHART_PARAMS)
            prices = response.json()["prices"]
            volumes = response.json()["total_volumes"]
            data = [(
                datetime.fromtimestamp(price[0]/1000).strftime("%Y-%m-%d"),
                price[1],
                volumes[i][1]
            ) for i, price in enumerate(prices)]
            df = pd.DataFrame(data, columns=["Date", "Settle", "Volume"])
            df.set_index("Date", inplace=True)
            df.to_csv(
                os.path.join("data", "raw_data", f"{ticker}.csv")
            )
        except Exception as ex:
            print(ex)
        

if __name__ == "__main__":
    main()
