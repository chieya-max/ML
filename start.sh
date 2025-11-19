#/bin/sh
# default to 8000 if PORT is not set
PORT=${PORT:-8000}
exec uvicorn ml_api:app --host 0.0.0.0 --port $PORT
