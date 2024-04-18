FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]