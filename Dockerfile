# Use an official Python runtime with GL support
FROM python:3.10-slim-bullseye

# Install Native C libraries for MuJoCo
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libosmesa6 \
    libglew-dev \
    libglfw3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Headless Rendering Config
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

# Start the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
