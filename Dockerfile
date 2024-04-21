# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy .env file into the container
RUN cp /home/realhack/secret/llm/.env .

# Copy the FastAPI application code into the container
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8081

# Command to run the application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081"]
