# Use the Python 3.10 slim base image
FROM python:3.10-slim

# Install required system dependencies without recommended packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libhdf5-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Set the working directory
WORKDIR /app

# Copy the requirements file and install Python dependencies without caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Create a non-root user
RUN useradd -m appuser

# Change ownership of /app to the non-root user
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Expose port 8080 for the application (adjust if your app uses a different port)
EXPOSE 8080

# Specify the command to run the application (update 'app.py' to your app's entry point)
CMD ["python", "app.py"]