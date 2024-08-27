# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files to disc
# and to ensure that Python output is sent straight to the terminal (e.g., for logs)
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Create and set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app/

# Install any needed packages specified in requirements.txt
# Ensure pip is upgraded before installing dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 8501

# Define environment variable for Streamlit
ENV STREAMLIT_SERVER_PORT 8501

# Run Streamlit app when the container launches
CMD ["streamlit", "run", "test1.py"]
