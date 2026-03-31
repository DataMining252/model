from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from datetime import date, timedelta
import pandas as pd

from .predictor import predict_forecast

from .models import ForecastResponse, HistoricalDataResponse, HistoricalRow, ForecastRow
from .utils import load_historical_data_from_db, get_next_business_days

app = FastAPI(title="Historical + Forecast API with DB")

@app.get("/historical_data", response_model=HistoricalDataResponse)
def get_historical_data(
    period: Optional[str] = Query("week", description="week | month | 3months | custom"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    columns: Optional[List[str]] = Query(None)
):
    """
    Lấy dữ liệu lịch sử từ DB, join gold_price + feature.
    - period: 'week', 'month', '3months', 'custom'
    - Nếu period=custom, phải truyền start_date và end_date
    """
    today = date.today()

    if period == "week":
        start_date = (today - timedelta(days=7)).isoformat()
        end_date = today.isoformat()
    elif period == "month":
        first_day_this_month = today.replace(day=1)
        last_day_prev_month = first_day_this_month - pd.Timedelta(days=1)
        start_date = (last_day_prev_month.replace(day=1)).isoformat()
        end_date = last_day_prev_month.isoformat()
    elif period == "3months":
        first_day_this_month = today.replace(day=1)
        last_day_prev_month = first_day_this_month - pd.Timedelta(days=1)
        three_months_ago = (last_day_prev_month - pd.DateOffset(months=2)).replace(day=1)
        start_date = three_months_ago.date().isoformat()
        end_date = last_day_prev_month.isoformat()
    elif period == "custom":
        if not start_date or not end_date:
            raise HTTPException(status_code=400, detail="start_date and end_date must be provided for custom period")
    else:
        raise HTTPException(status_code=400, detail="Invalid period. Use week | month | 3months | custom")

    # Lấy dữ liệu từ DB
    df = load_historical_data_from_db(start_date, end_date, columns)

    if df.empty:
        raise HTTPException(status_code=400, detail="No historical data found in DB")

    historical = [
        HistoricalRow(
            date=str(row.date),
            data={col: row[col] for col in df.columns if col != 'date'}
        )
        for _, row in df.iterrows()
    ]

    return HistoricalDataResponse(historical=historical)

@app.get("/predict")
def predict(n_forecast_days: int = 7):

    hist_df = load_historical_data_from_db(limit=100)
    hist_df = hist_df.rename(columns={
        'dxy': 'DXY',
        'sp500': 'SP500',
        'oil': 'OIL',
        'interest_rate': 'INTEREST_RATE',
        'cpi': 'CPI'
    })

    hist_df['date'] = pd.to_datetime(hist_df['date'])

    last_date = hist_df['date'].max().date()

    forecast_dates = get_next_business_days(
        n_forecast_days,
        from_date=last_date
    )

    forecast = predict_forecast(
        hist_df,
        forecast_dates,
        n_forecast_days
    )

    return {
        "forecast": forecast
    }