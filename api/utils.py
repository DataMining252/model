from datetime import date, timedelta
import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONN_STR = os.getenv("DB_URL")
TABLE_NAME = os.getenv("TABLE_NAME", "historical_data")

def get_next_business_days(n_days: int, from_date: date = None):
    if from_date is None:
        from_date = date.today()
    days = []
    delta = 1
    while len(days) < n_days:
        next_day = from_date + timedelta(days=delta)
        if next_day.weekday() < 6 and next_day.weekday() > 0:  # Mon=1, Fri=5
            days.append(next_day)
        delta += 1
    return days

SCHEMA = "predict_gold_price"

def load_historical_data_from_db(
    start_date: str = None,
    end_date: str = None,
    columns: list = None,
    limit: int = None
):
    if DB_CONN_STR is None:
        raise ValueError("DB_URL not set in environment")

    engine = create_engine(DB_CONN_STR)

    gold_cols = ['open','high','low','close','volume']
    feature_cols = ['dxy','sp500','oil','interest_rate','cpi']

    # -------- SELECT COLUMNS --------
    if columns:
        selected_cols = []
        for c in columns:
            if c in gold_cols:
                selected_cols.append(f"g.{c}")
            elif c in feature_cols:
                selected_cols.append(f"f.{c}")
            else:
                raise ValueError(f"Column {c} not found")
    else:
        selected_cols = [f"g.{c}" for c in gold_cols] + [f"f.{c}" for c in feature_cols]

    select_clause = ", ".join(["d.date"] + selected_cols)

    # -------- BASE QUERY --------
    query_str = f"""
        SELECT {select_clause}
        FROM {SCHEMA}.gold_price g
        JOIN {SCHEMA}.dim_date d ON g.date_id = d.id
        LEFT JOIN {SCHEMA}.feature f ON g.date_id = f.date_id
        WHERE 1=1
    """

    # -------- FILTER --------
    params = {}

    if start_date:
        query_str += " AND d.date >= :start_date"
        params["start_date"] = start_date

    if end_date:
        query_str += " AND d.date <= :end_date"
        params["end_date"] = end_date

    # -------- ORDER + LIMIT --------
    query_str += " ORDER BY d.date DESC"

    if limit:
        query_str += " LIMIT :limit"
        params["limit"] = limit

    # -------- EXECUTE --------
    query = text(query_str)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    # ⚠️ cực quan trọng: sort lại ascending cho LSTM
    df = df.sort_values("date")

    return df