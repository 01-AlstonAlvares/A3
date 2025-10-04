# Use official Python image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy entire app folder (including deploy_assets)
COPY app/ ./ 
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Expose port
EXPOSE 5000

# Define environment variables for MLflow credentials.
# These will be used by the test script inside the container to authenticate.
# The GitHub Actions workflow will pass the secrets to these variables.
ENV MLFLOW_TRACKING_USERNAME=${MLFLOW_TRACKING_USERNAME}
ENV MLFLOW_TRACKING_PASSWORD=${MLFLOW_TRACKING_PASSWORD}

# --- The CMD instruction is the command that runs when the container starts ---
# Instead of running a web app, we now tell it to run the test script.
CMD ["python", "app.py"]