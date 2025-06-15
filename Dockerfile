# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies that might be required by Python packages
# For example, if you had packages that needed C compilers or other libs.
# RUN apt-get update && apt-get install -y --no-install-recommends gcc

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and artifacts into the container
# This assumes you have already trained the model and have the artifacts ready.
COPY ./backend /app/backend
COPY ./app /app/app
COPY ./artifacts /app/artifacts
COPY ./data /app/data

# Expose the port the app runs on
EXPOSE 8888

# Run the application using a production-grade server like Gunicorn
# The command should be: gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app -b 0.0.0.0:8888
# Breaking it down for the CMD instruction:
# ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "-b", "0.0.0.0:8888"]
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8888", "app.main:app"] 