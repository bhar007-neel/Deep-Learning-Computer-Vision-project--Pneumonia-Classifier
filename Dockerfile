# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose Flask port
EXPOSE 10000
# Run the Flask app
CMD ["/bin/sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} app.app:app"]