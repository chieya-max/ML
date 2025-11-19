FROM python:3.13-slim

# install system dependencies required for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# working directory
WORKDIR /app

# install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# copy app files
COPY . .

# expose port (railway handles actual routing)
EXPOSE 8000

# start FastAPI via entrypoint script
CMD ["python", "ml_api.py"]
