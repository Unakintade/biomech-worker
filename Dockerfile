# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV MUJOCO_GL osmesa

# Install system dependencies for MuJoCo and Mesh processing
# libgl1-mesa-glx and libosmesa6 are required for headless MuJoCo rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libosmesa6 \
    mesa-utils \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Headless Rendering Config
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

# Set the working directory
WORKDIR /code

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary application directories
RUN mkdir -p app/data app/models

# Copy the application code
COPY ./app ./app

# Set permissions for the data folder (important for Render's ephemeral storage)
RUN chmod -R 777 /code/app/data

# Expose the port Render uses (10000 by default)
EXPOSE 10000

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
