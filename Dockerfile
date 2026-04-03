# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy code
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r api/requirements.txt

# Train models khi build image (optional)
# RUN python model/lstm/train_lstm.py
# RUN python model/xgboost/train_xgb.py

# Expose port cho Render
EXPOSE 8000

# Run API
CMD ["uvicorn", "model.api.main:app", "--host", "0.0.0.0", "--port", "8000"]