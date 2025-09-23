# Use official Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and models
COPY app/ ./app/
COPY models/ ./models/

# Create uploads folder
RUN mkdir /app/uploads

# Set environment variables
ENV FLASK_APP=app/main.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5600

# Expose port
EXPOSE 5000

# Run Flask
CMD ["flask", "run"]
