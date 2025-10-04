import os
import json
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.pyfunc import PyFuncModel

# -------------------------------
# MLflow Configuration
# -------------------------------
print("--- Connecting to MLflow server ---")
# Set credentials for the MLflow server from environment variables for security
os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'password'
mlflow.set_tracking_uri("http://mlflow.ml.brain.cs.ait.ac.th/")

# Define the model to be loaded from the registry using its alias
model_name = "st126488-a3-model"
model_alias = "Staging"
model_uri = f"models:/{model_name}@{model_alias}"

# -------------------------------
# Load Model and Assets from MLflow
# -------------------------------
try:
    # Step 1: Load the model pipeline from the MLflow Model Registry.
    print(f"Loading model '{model_name}' with alias '{model_alias}' from MLflow...")
    model_pipeline: PyFuncModel = mlflow.pyfunc.load_model(model_uri)
    print("✅ Model loaded successfully from MLflow.")

    # --- FIX: Explicitly download the assets artifact using the model's run_id ---
    print("Downloading associated assets from MLflow...")
    # Get the run_id from the loaded model's metadata
    run_id = model_pipeline.metadata.get_model_info().run_id
    
    client = MlflowClient()
    
    # Step 2: Download the assets directory to a temporary local path
    local_assets_dir = client.download_artifacts(run_id, "model_assets", ".")
    
    # Construct the full path to the assets.json file
    assets_path = os.path.join(local_assets_dir, "assets.json")
    
    if not os.path.exists(assets_path):
        raise FileNotFoundError(f"assets.json not found in the downloaded artifacts at {local_assets_dir}")

    with open(assets_path, "r") as f:
        assets = json.load(f)
        num_cols = assets["num_cols"]
        cat_cols = assets["cat_cols"]
        price_bin_edges = assets["price_bin_edges"]
        class_names = assets["class_names"]
    
    # This is the full list of features the model pipeline expects
    features = num_cols + cat_cols
    print("✅ Assets loaded successfully from MLflow artifacts.")

except Exception as e:
    print(f"🚨 Failed to load model or assets from MLflow: {e}")
    # This fallback ensures the app can start and display an error
    model_pipeline = None
    assets = None


# -------------------------------
# Initialize Dash App
# -------------------------------
app = dash.Dash(__name__)
app.title = "Car Price Category Predictor"


# -------------------------------
# App Layout
# -------------------------------
# If the model fails to load, display an error message. Otherwise, show the app.
if model_pipeline is None or assets is None:
    app.layout = html.Div([
        html.H1("🚨 Error: Could not load model from MLflow", style={"color": "red", "textAlign": "center"}),
        html.P("Please ensure a model version has the 'Staging' alias in the MLflow UI and that the app has the correct credentials.", style={"textAlign": "center"})
    ])
else:
    # Define Dropdown Options
    owner_options = [
        {'label': 'First Owner', 'value': 1}, {'label': 'Second Owner', 'value': 2},
        {'label': 'Third Owner', 'value': 3}, {'label': 'Fourth & Above Owner', 'value': 4}
    ]
    brand_options = [
        'Maruti', 'Hyundai', 'Mahindra', 'Tata', 'Honda', 'Ford', 'Toyota', 
        'Chevrolet', 'Renault', 'Volkswagen', 'Nissan', 'Skoda', 'BMW', 'Mercedes-Benz'
    ]

    # Helper function to generate input fields statically
    def create_input_fields():
        # The UI will ask for user-friendly inputs like 'year'
        display_features = ['year', 'km_driven', 'mileage', 'owner', 'brand']
        
        fields = []
        for col in display_features:
            field_style = {"marginBottom": "10px"}
            label_style = {"display": "block", "marginBottom": "5px"}
            
            if col == 'year':
                 fields.append(html.Div([
                    html.Label("Year:", style=label_style),
                    dcc.Input(id="input-year", type="number", placeholder="e.g., 2015", style={"width": "95%", "padding": "8px"})
                ], style=field_style))
            elif col == 'km_driven' or col == 'mileage':
                fields.append(html.Div([
                    html.Label(f"{col.replace('_', ' ').title()}:", style=label_style),
                    dcc.Input(id=f"input-{col}", type="number", placeholder="e.g., 70000", style={"width": "95%", "padding": "8px"})
                ], style=field_style))
            elif col == 'owner':
                fields.append(html.Div([
                    html.Label("Owner:", style=label_style),
                    dcc.Dropdown(id="input-owner", options=owner_options, value=1, clearable=False, style={"width": "100%"})
                ], style=field_style))
            elif col == 'brand':
                 fields.append(html.Div([
                    html.Label("Brand:", style=label_style),
                    dcc.Dropdown(id="input-brand", options=brand_options, value='Maruti', clearable=False, style={"width": "100%"})
                ], style=field_style))
        return fields

    app.layout = html.Div([
        html.Div([
            html.H1("🚗 Car Price Category Predictor", style={"margin": "0"}),
            html.P("A machine learning app to classify used cars into price brackets.", style={"margin": "0"})
        ], style={"backgroundColor": "#007BFF", "color": "white", "padding": "20px", "textAlign": "center"}),
        html.Div([
            html.Div([
                html.H3("How It Works"),
                dcc.Markdown("""
                    This app uses a **Multinomial Logistic Regression** model loaded directly from the MLflow Model Registry.
                    1.  Enter the car's details on the right.
                    2.  Click the **'Predict Category'** button.
                    3.  The model will output the predicted class and its estimated price range.
                """),
                html.Hr(),
                html.H3("Model Features"),
                 dcc.Markdown("""
                    - **Dynamic Loading**: Loads the latest 'Staging' model on startup.
                    - **Feature Engineering**: Uses `car_age` and `km_per_year` for better predictions.
                    - **Regularization**: Uses an L2 (Ridge) penalty to prevent overfitting.
                """)
            ], style={"flex": "1", "padding": "20px", "backgroundColor": "#f8f9fa"}),
            html.Div([
                html.H3("Enter Car Features"),
                html.Div(create_input_fields()),
                html.Button("Predict Category", id="predict-button", n_clicks=0, 
                            style={"marginTop": "20px", "padding": "10px 20px", "fontSize": "16px", "cursor": "pointer"}),
                html.Hr(),
                html.Div(id="prediction-output", style={"fontSize": "22px", "fontWeight": "bold", "marginTop": "20px"})
            ], style={"flex": "2", "padding": "20px"})
        ], style={"display": "flex", "flexDirection": "row"})
    ], style={"fontFamily": "Arial, sans-serif"})


    # Callback for prediction
    @app.callback(
        Output("prediction-output", "children"),
        Input("predict-button", "n_clicks"),
        [
            State("input-year", "value"),
            State("input-km_driven", "value"),
            State("input-mileage", "value"),
            State("input-owner", "value"),
            State("input-brand", "value"),
        ]
    )
    def predict_price_category(n_clicks, year, km_driven, mileage, owner, brand):
        if n_clicks == 0:
            return ""
        try:
            all_values = [year, km_driven, mileage, owner, brand]
            if any(v is None for v in all_values):
                return html.Div("Please fill in all feature fields before predicting.", style={"color": "red"})

            # --- Feature Engineering: Convert user inputs to model inputs ---
            current_year = datetime.now().year
            car_age = current_year - year
            # Add 1 to car_age to prevent division by zero for new cars
            km_per_year = km_driven / (car_age + 1)

            # Create the DataFrame with the exact columns the model expects
            user_input = {
                'car_age': car_age, 'km_driven': km_driven, 'mileage': mileage,
                'owner': owner, 'brand': brand, 'km_per_year': km_per_year
            }
            df_input = pd.DataFrame([user_input])
            
            # --- FIX: Explicitly cast data types to match the model's schema ---
            # The schema requires specific integer and float types.
            df_input['car_age'] = df_input['car_age'].astype(np.int64)
            df_input['km_driven'] = df_input['km_driven'].astype(np.int64)
            df_input['owner'] = df_input['owner'].astype(np.int64)
            df_input['mileage'] = df_input['mileage'].astype(np.float64)
            df_input['km_per_year'] = df_input['km_per_year'].astype(np.float64)

            # Reorder columns to be certain they match the model's expectation
            df_input = df_input[features]
            
            print("\n[DEBUG] Data sent to model:")
            print(df_input.to_string())
            print(df_input.dtypes)

            # The loaded model is a pyfunc model, so we predict on the DataFrame
            predicted_class = model_pipeline.predict(df_input)[0]
            
            # Get the price range from the loaded assets
            lower_bound = price_bin_edges[predicted_class]
            upper_bound = price_bin_edges[predicted_class + 1]
            
            category_name = class_names[predicted_class]
            price_range_str = f"₹{lower_bound:,.0f} - ₹{upper_bound:,.0f}"
            
            return html.Div([
                html.Span(f"Predicted Category: "),
                html.Span(f"{category_name}", style={"color": "#007BFF"}),
                html.Br(),
                html.Span(f"Estimated Price Range: "),
                html.Span(f"{price_range_str}", style={"color": "#28a745"})
            ])
        except Exception as e:
            return f"An error occurred during prediction: {str(e)}"

# -------------------------------
# Run Dash Server
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)

