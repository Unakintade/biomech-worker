# Use an official Python runtime with GL support
FROM python:3.10-slim-bullseye

# Install system dependencies for MuJoCo and OpenGL
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libosmesa6 \
    libglew-dev \
    libglfw3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Environment Variables for MuJoCo headless rendering
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

# Start FastAPI using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
