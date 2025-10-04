import os
import mlflow
from mlflow.tracking import MlflowClient

# --- MLflow Configuration ---
print("--- Configuring MLflow Connection ---")
# Credentials are read from environment variables set by the GitHub Action
MLFLOW_TRACKING_USERNAME = os.environ.get("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD")

if not MLFLOW_TRACKING_USERNAME or not MLFLOW_TRACKING_PASSWORD:
    raise ValueError("MLflow credentials must be set as environment variables.")

mlflow.set_tracking_uri("http://mlflow.ml.brain.cs.ait.ac.th/")

# Define the model name and the aliases we will be working with
MODEL_NAME = "st126488-a3-model"
STAGING_ALIAS = "Staging"
PRODUCTION_ALIAS = "Production"

def promote_staging_to_production():
    """
    Finds the model version with the 'Staging' alias and promotes it
    by assigning the 'Production' alias.
    """
    client = MlflowClient()
    
    try:
        # Find the model version currently in "Staging"
        print(f"Searching for model version with alias '{STAGING_ALIAS}'...")
        staging_version_info = client.get_model_version_by_alias(name=MODEL_NAME, alias=STAGING_ALIAS)
        
        if not staging_version_info:
            print(f"🚨 No model version found with the '{STAGING_ALIAS}' alias. Nothing to promote.")
            return

        version_to_promote = staging_version_info.version
        print(f"✅ Found Version {version_to_promote} in '{STAGING_ALIAS}'.")

        # Set the "Production" alias for this version
        print(f"Promoting Version {version_to_promote} by setting alias '{PRODUCTION_ALIAS}'...")
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=PRODUCTION_ALIAS,
            version=version_to_promote
        )
        print(f"🎉 Successfully promoted Version {version_to_promote} to '{PRODUCTION_ALIAS}'.")
        
        # Optional: Remove the "Staging" alias from the newly promoted version
        print(f"Removing '{STAGING_ALIAS}' alias from Version {version_to_promote}...")
        client.delete_registered_model_alias(
            name=MODEL_NAME,
            alias=STAGING_ALIAS
        )
        print("✅ Alias cleanup complete.")

    except Exception as e:
        print(f"🚨 An error occurred during the promotion process: {e}")
        # Exit with a non-zero status code to fail the GitHub Action
        exit(1)

if __name__ == '__main__':
    promote_staging_to_production()
