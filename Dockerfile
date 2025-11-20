# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy backend code
COPY backend/ ./

# Copy models directory
COPY backend/models ./models

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Expose port 8080 (standard for Flask)
EXPOSE 8080

# Set environment variables (optional but useful)
ENV PYTHONUNBUFFERED=1 \
    PORT=8080

# Run the Flask app
CMD ["python", "app.py"]
