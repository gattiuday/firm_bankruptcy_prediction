FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application into the expected directory structure
COPY . ./firm_bankruptcy_prediction

# Expose Streamlit port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "firm_bankruptcy_prediction/app.py", "--server.address=0.0.0.0"]
