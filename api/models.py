from typing import List

from pydantic import BaseModel

class ForecastRequest(BaseModel):
    start_date: str
    end_date: str
    n_forecast_days: int

class HistoricalRow(BaseModel):
    date: str
    value: float

class ForecastRow(BaseModel):
    date: str
    prediction: float

class ForecastResponse(BaseModel):
    historical: list[HistoricalRow]
    forecast: list[ForecastRow]

class HistoricalRow(BaseModel):
    date: str
    data: dict 

class HistoricalDataResponse(BaseModel):
    historical: List[HistoricalRow]