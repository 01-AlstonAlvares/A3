import unittest
import pandas as pd
import numpy as np
import os
import mlflow

# --- MLflow Configuration ---
# Set credentials for the MLflow server.
# For a CI/CD environment like GitHub Actions, these should be set as secrets.
MLFLOW_TRACKING_USERNAME = os.environ.get("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD")

# --- Fallback for Local Testing ---
# If environment variables are not found, use the credentials here.
if not MLFLOW_TRACKING_USERNAME:
    MLFLOW_TRACKING_USERNAME = 'admin' # Your MLflow username
if not MLFLOW_TRACKING_PASSWORD:
    MLFLOW_TRACKING_PASSWORD = 'password' # Your MLflow password

# Check if credentials are provided
if not MLFLOW_TRACKING_USERNAME or not MLFLOW_TRACKING_PASSWORD:
    raise ValueError("MLflow credentials are not set. Please set them as environment variables or directly in the script for local testing.")

os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD

# Set the remote server URI and model alias
mlflow.set_tracking_uri("http://mlflow.ml.brain.cs.ait.ac.th/")
MODEL_NAME = "st126488-a3-model"
MODEL_ALIAS = "Staging"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


class TestDeployedModel(unittest.TestCase):
    """
    Unit test suite for the deployed car price prediction model.
    """

    @classmethod
    def setUpClass(cls):
        """
        This method is called once before any tests are run.
        It loads the Staging model from the MLflow Model Registry.
        """
        try:
            print(f"--- Loading model '{MODEL_NAME}' with alias '{MODEL_ALIAS}' from MLflow ---")
            cls.model = mlflow.pyfunc.load_model(MODEL_URI)
            print("✅ Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from MLflow. "
                f"Ensure a model version has the '{MODEL_ALIAS}' alias set. Error: {e}"
            )
        
        # Define the exact feature names the model expects
        cls.expected_features = ['car_age', 'km_driven', 'mileage', 'owner', 'brand', 'km_per_year']

    def test_model_takes_expected_input(self):
        """
        Test 1: Verifies that the model successfully processes a valid input DataFrame.
        """
        print("\nRunning test 1: Model takes expected input...")
        # Create a sample DataFrame representing a single valid car
        sample_data = {
            'car_age': [5],
            'km_driven': [50000],
            'mileage': [20.0],
            'owner': [1],
            'brand': ['Maruti'],
            'km_per_year': [10000.0]
        }
        sample_df = pd.DataFrame(sample_data, columns=self.expected_features)
        
        try:
            prediction = self.model.predict(sample_df)
            self.assertIsNotNone(prediction, "Prediction should not be None.")
            print("  -> PASSED")
        except Exception as e:
            self.fail(f"Model prediction failed with a valid input. Error: {e}")

    def test_output_has_expected_shape(self):
        """
        Test 2: Verifies that the model's output has the correct shape for batch predictions.
        """
        print("\nRunning test 2: Output has expected shape...")
        # Create a sample DataFrame with multiple rows (3 cars)
        sample_data = {
            'car_age': [5, 10, 2],
            'km_driven': [50000, 120000, 15000],
            'mileage': [20.0, 18.5, 15.0],
            'owner': [1, 2, 1],
            'brand': ['Maruti', 'Hyundai', 'BMW'],
            'km_per_year': [8333.3, 10909.1, 5000.0]
        }
        sample_df = pd.DataFrame(sample_data, columns=self.expected_features)
        
        # Make predictions on the batch of data
        predictions = self.model.predict(sample_df)
        
        # 1. Check that the output is a numpy array
        self.assertIsInstance(predictions, np.ndarray, "Model output should be a numpy array.")
        
        # 2. Check that the shape is correct (a 1D array with length matching input rows)
        self.assertEqual(predictions.shape, (3,), f"Output shape is incorrect. Expected (3,), but got {predictions.shape}.")
        print("  -> PASSED")

# This allows the script to be run directly from the command line
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

