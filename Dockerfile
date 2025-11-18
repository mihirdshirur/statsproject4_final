# 1. Base Image: Python image for the required linux/amd64 platform
FROM python:3.11-bookworm

# 2. Set environment variables
# Ensures Python output is sent directly to the terminal/logs
ENV PYTHONUNBUFFERED 1

# 3. Set the working directory inside the container
# All subsequent commands will run relative to this directory.
WORKDIR /app

# 4. Copy requirements file and install dependencies
# This leverages Docker's build cache.
COPY requirements.text .
RUN pip install --no-cache-dir -r requirements.text

# 4. Install 'make' utility
# We use a single RUN command to keep the image layers clean:
# - apt-get update: Refreshes the package list
# - apt-get install -y make: Installs the utility
# - rm -rf /var/lib/apt/lists/*: Cleans up the cache immediately to keep the image small
#RUN apt-get update && \
#    apt-get install -y make && \
#    rm -rf /var/lib/apt/lists/*

RUN mkdir -p models data/weather data/complete_dfs data/pjm

# 5. Copy the rest of your application code and data
# This single command copies everything from your project root (on the host)
# into the /app directory (in the container).
# This includes: main.py, my_utils.py, the 'data' directory, etc.
# IMPORTANT: You must use a .dockerignore file (see section below)
# to exclude unnecessary files like .git, __pycache__, and venvs.
COPY . /app

# 6. Define the command to run your application
# This is the entry point when the container starts.
#CMD ["make"]
