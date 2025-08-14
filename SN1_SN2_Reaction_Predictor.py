import pandas as pd
from IPython.display import Image, display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import sys

# --- Constants for encoding and decoding model inputs/outputs ---
# Mapping nucleophile strength from string values to numerical values for training
NUCLEOPHILE_MAP = {'strong': 2, 'moderate': 1, 'weak': 0}
# Reverse mapping for model predictions: 1 = SN1, 0 = SN2
DECODE_MAP = {1: 'SN1', 0: 'SN2'}

# Valid values for each input feature; used to validate user input
ALLOWED_VALUES = {
    'substrate_degree': ['methyl', 'primary', 'secondary', 'tertiary'],
    'special_environment': ['none', 'allylic', 'benzylic', 'neopentyl'],
    'nucleophile_strength': ['strong', 'moderate', 'weak'],
    'solvent': ['protic', 'aprotic'],
    'leaving_group': ['h2o', 'br', 'cl', 'i', 'ots', 'oms', 'otf', 'f', 'tso']
}

# Prompts for collecting input from the user
FIELD_PROMPTS = {
    'substrate_degree': "Substrate degree (methyl, primary, secondary, tertiary): ",
    'special_environment': "Special environment (none, allylic, benzylic, neopentyl): ",
    'nucleophile_strength': "Nucleophile strength (strong, moderate, weak): ",
    'solvent': "Solvent type (protic, aprotic): ",
    'leaving_group': "Leaving group: "
}

# Get the absolute path to the folder where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Data Preprocessing ---
def preprocess_data(csv_path):
    """
    Reads the dataset from CSV, encodes categorical variables, 
    and prepares it for training.
    """
    df = pd.read_csv(csv_path)
    # Map nucleophile strength to numerical values
    df['nucleophile_strength'] = df['nucleophile_strength'].map(NUCLEOPHILE_MAP)
    # Map reaction mechanism to binary values (1=SN1, 0=SN2)
    df['mechanism'] = df['mechanism'].map({'SN1': 1, 'SN2': 0})
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['substrate_degree', 'special_environment', 'solvent', 'leaving_group'])
    # Remove rows with missing values to avoid training errors
    df.dropna(inplace=True)
    return df


# --- Training and Saving Model ---
def train_and_save_model(csv_path, model_path, features_path):
    """
    Trains a RandomForest model on the dataset and saves both the model 
    and the list of feature names to disk.
    """
    print("Training model...")
    df = preprocess_data(csv_path)

    # Separate features and target
    X = df.drop(columns=['mechanism', 'source'])
    y = df['mechanism']

    # Train-test split for model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save the model and feature list
    joblib.dump(model, model_path)
    joblib.dump(X_train.columns.tolist(), features_path)

    # Print the test accuracy for reference
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Training complete — Test Accuracy: {acc:.2%}")
    
    return model, X_train.columns.tolist()


# --- Model Loading (with Auto-Training if Missing) ---
def load_model_and_features(csv_path, model_path, features_path):
    """
    Loads the trained model and features list from disk.
    If they do not exist, trains a new model automatically.
    """
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        return train_and_save_model(csv_path, model_path, features_path)

    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features


# --- Collect User Input ---
def get_user_input():
    """
    Prompts the user for each reaction parameter and validates the input 
    against the allowed values.
    """
    input_data = {}
    for key, prompt in FIELD_PROMPTS.items():
        while True:
            conditions = input(prompt).strip().lower()
            if conditions in ALLOWED_VALUES[key]:
                input_data[key] = conditions
                # Special note if user picks Fluorine as leaving group
                if key == 'leaving_group' and conditions == 'f':
                    print("Note: Fluorine is a poor leaving group.")
                break
            else:
                print(f"Invalid input for {key}. Allowed reaction conditions: {ALLOWED_VALUES[key]}")
    return input_data


# --- Prepare Input for Model Prediction ---
def prepare_input_dataframe(input_data, model_features):
    """
    Encodes a single reaction input into the same format used 
    for training so the model can predict correctly.
    """
    # Encode nucleophile strength numerically
    input_data_encoded = input_data.copy()
    input_data_encoded['nucleophile_strength'] = NUCLEOPHILE_MAP[input_data['nucleophile_strength']]

    # Convert input to DataFrame
    df_input = pd.DataFrame([input_data_encoded])

    # One-hot encode categorical variables
    df_input_encoded = pd.get_dummies(df_input, columns=['substrate_degree', 'special_environment', 'solvent', 'leaving_group'])

    # Ensure all expected features are present, fill missing with 0
    for col in model_features:
        if col not in df_input_encoded.columns:
            df_input_encoded[col] = 0

    # Arrange columns in the same order as during training
    df_input_encoded = df_input_encoded[model_features]
    return df_input_encoded


# --- Generate Explanation ---
def generate_explanation(predicted_label, substrate, nucleophile_strength, solvent, leaving_group):
    """
    Returns a human-readable explanation for why the prediction was made.
    """
    if predicted_label == "SN2":
        return (f"Since {leaving_group} is a good leaving group and the substrate is {substrate} "
                f"with a {nucleophile_strength} nucleophile in a {solvent} solvent, this favors SN2.")
    elif predicted_label == "SN1":
        return (f"The {substrate} substrate and {solvent} solvent favor carbocation formation, "
                f"and with {leaving_group} as the leaving group, this reaction proceeds via SN1.")
    else:
        return "Prediction unknown."


# --- Prediction Workflow ---
def predict_mechanism(model, model_features):
    """
    Orchestrates the input collection, encoding, prediction, and output display.
    """
    # Get validated user input
    input_data = get_user_input()

    # Encode input in model-ready format
    df_input_encoded = prepare_input_dataframe(input_data, model_features)

    # Make prediction
    prediction_encoded = model.predict(df_input_encoded)[0]
    predicted_label = DECODE_MAP[prediction_encoded]

    # Display prediction
    print(f"\nPredicted mechanism: {predicted_label}")
    print(generate_explanation(predicted_label, input_data['substrate_degree'], input_data['nucleophile_strength'], input_data['solvent'], input_data['leaving_group']))

    # Show mechanism diagram if PNG is available in the same folder
    image_file = os.path.join(BASE_DIR, f"{predicted_label}.png")
    if os.path.exists(image_file):
        display(Image(filename=image_file))
        print("McMurry, Organic Chemistry: A Tenth Edition")


# --- Script Entry Point ---
if __name__ == "__main__":
    # Define file paths for dataset, model, and features file
    CSV_PATH = os.path.join(BASE_DIR, "SN1_SN2_Mechanisms.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "reaction-mechanism-predictor.joblib")
    FEATURES_PATH = os.path.join(BASE_DIR, "model_features.joblib")

    # Ensure dataset exists before continuing
    if not os.path.exists(CSV_PATH):
        print(f"Dataset not found at {CSV_PATH}")
        sys.exit(1)

    # Load existing model or train a new one automatically
    model, model_features = load_model_and_features(CSV_PATH, MODEL_PATH, FEATURES_PATH)

    # Start prediction workflow
    predict_mechanism(model, model_features)
