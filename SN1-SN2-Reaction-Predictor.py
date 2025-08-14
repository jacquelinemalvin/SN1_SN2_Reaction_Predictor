
import pandas as pd
from IPython.display import Image, display
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import sys

# --- Constants ---
NUCLEOPHILE_MAP = {'strong': 2, 'moderate': 1, 'weak': 0}
DECODE_MAP = {1: 'SN1', 0: 'SN2'}

ALLOWED_VALUES = {
    'substrate_degree': ['methyl', 'primary', 'secondary', 'tertiary'],
    'special_environment': ['none', 'allylic', 'benzylic', 'neopentyl'],
    'nucleophile_strength': ['strong', 'moderate', 'weak'],
    'solvent': ['protic', 'aprotic'],
    'leaving_group': ['h2o', 'br', 'cl', 'i', 'ots', 'oms', 'otf', 'f', 'tso']
}

FIELDS_PROMPTS = {
    'substrate_degree': "Substrate degree (methyl, primary, secondary, tertiary): ",
    'special_environment': "Special environment (none, allylic, benzylic, neopentyl): ",
    'nucleophile_strength': "Nucleophile strength (strong, moderate, weak): ",
    'solvent': "Solvent type (protic, aprotic): ",
    'leaving_group': "Leaving group: "
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "SN1_SN2_Mechanisms.csv")
MODEL_PATH = os.path.join(BASE_DIR, "reaction-mechanism-predictor.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "model_features.joblib")


def preprocess_data(csv_path):
    """
    Load and preprocess the dataset for training.
    """
    df = pd.read_csv(csv_path)

    # Map nucleophile strength
    df['nucleophile_strength'] = df['nucleophile_strength'].map(NUCLEOPHILE_MAP)

    # Map mechanisms to binary
    df['mechanism'] = df['mechanism'].map({'SN1': 1, 'SN2': 0})

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=['substrate_degree', 'special_environment', 'solvent', 'leaving_group'])

    # Drop any rows with missing values
    df.dropna()

    return df


def train_and_save_model(data_path, model_path, features_path):
    """
    # Train RandomForest model and save the model and feature columns.
    # """
    # df = preprocess_data(data_path)

    # X = df.drop(columns=['mechanism', 'source'])
    # y = df['mechanism']

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # model = RandomForestClassifier(random_state=42)
    # model.fit(X_train, y_train)

    # # Save model and features
    # joblib.dump(model, model_path)
    # joblib.dump(X_train.columns.tolist(), features_path)

    # # Evaluate
    # predictions = model.predict(X_test)
    # acc = accuracy_score(y_test, predictions)
    # print(f"Test set accuracy: {acc:.2%}")

    # scores = cross_val_score(model, X, y, cv=5)
    # print(f"Cross-validation accuracy: {scores.mean():.2%} ± {scores.std():.2%}")

    # corrs = df.corr(numeric_only=True)['mechanism'].sort_values(ascending=False)
    # print("\nCorrelation with mechanism:")
    # print(corrs)
    print("Training model...")
    df = preprocess_data(CSV_PATH)
    X = df.drop(columns=['mechanism', 'source'])
    y = df['mechanism']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(X_train.columns.tolist(), FEATURES_PATH)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Training complete — Test Accuracy: {acc:.2%}")

def load_model_and_features(model_path, features_path):
    """
    Load saved model and feature list.
    """
    # model = joblib.load(model_path)
    # features = joblib.load(features_path)
    # return model, features
    if not (os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH)):
        train_and_save_model()
    return joblib.load(MODEL_PATH), joblib.load(FEATURES_PATH)


def get_user_input():
    """
    Prompt user for inputs with validation.
    """
    input_data = {}
    while True:
        for key, prompt in FIELDS_PROMPTS.items():
            while True:
                value = input(prompt).strip().lower()
                if value in ALLOWED_VALUES[key]:
                    input_data[key] = value
                    if key == 'leaving_group' and value == 'f':
                        print("Note: Fluorine is a poor leaving group due to its electrophilic properties.")
                    break
                else:
                    print(f"Invalid input for {key}. Allowed values: {ALLOWED_VALUES[key]}")
        if input_data:
            break
    return input_data


def prepare_input_dataframe(input_data, model_features):
    """
    Convert input dictionary into a properly encoded DataFrame matching training features.
    """
    input_data_encoded = input_data.copy()

    # Map nucleophile strength to numeric
    input_data_encoded['nucleophile_strength'] = NUCLEOPHILE_MAP[input_data['nucleophile_strength']]

    df_input = pd.DataFrame([input_data_encoded])

    df_input_encoded = pd.get_dummies(df_input, columns=['substrate_degree', 'special_environment', 'solvent', 'leaving_group'])

    # Add any missing columns that the model expects
    for col in model_features:
        if col not in df_input_encoded.columns:
            df_input_encoded[col] = 0

    # Keep only model features columns in the correct order
    df_input_encoded = df_input_encoded[model_features]

    return df_input_encoded


def generate_explanation(predicted_label, substrate, nucleophile_strength, solvent, leaving_group):
    """
    Generate a human-readable explanation of the prediction.
    """
    if predicted_label == "SN2":
        return (f"Since {leaving_group} is a good leaving group and the substrate is {substrate} "
                f"with a {nucleophile_strength} nucleophile in a {solvent} solvent, this favors SN2.")
    elif predicted_label == "SN1":
        return (f"The {substrate} substrate and {solvent} solvent favor carbocation formation, "
                f"and with {leaving_group} as the leaving group, this reaction proceeds via SN1.")
    else:
        return "Prediction unknown, unable to generate explanation."


def predict_mechanism(model, model_features):
    """
    # Run the user input, encode, predict, and display results.
    # """
    # input_data = get_user_input()
    # df_input_encoded = prepare_input_dataframe(input_data, model_features)
    # prediction_num = model.predict(df_input_encoded)[0]
    # predicted_label = DECODE_MAP[prediction_num]

    # explanation = generate_explanation(
    #     predicted_label,
    #     input_data['substrate_degree'],
    #     input_data['nucleophile_strength'],
    #     input_data['solvent'],
    #     input_data['leaving_group']
    # )
    input_data = get_user_input()
    df_input_encoded = prepare_input_dataframe(input_data, model_features)
    prediction_num = model.predict(df_input_encoded)[0]
    predicted_label = DECODE_MAP[prediction_num]
    print(f"\nPredicted mechanism: {predicted_label}")
    print(generate_explanation(predicted_label, input_data['substrate_degree'], input_data['nucleophile_strength'], input_data['solvent'], input_data['leaving_group']))

    if predicted_label == "SN2":
        display(Image(filename=os.path.join(BASE_DIR, "SN2.png")))
        print("McMurry, Organic Chemistry: A Tenth Edition")
    elif predicted_label == "SN1":
        display(Image(filename=os.path.join(BASE_DIR, "SN1.png")))
        print("McMurry, Organic Chemistry: A Tenth Edition")
   

if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description="SN1 vs SN2 Reaction Mechanism Predictor")
    # parser.add_argument('--train', action='store_true', help='Train model from dataset')
    # parser.add_argument('--data', default='SN1_SN2_Mechanisms.csv', help='Path to training CSV file')
    # parser.add_argument('--model', default='reaction-mechanism-predictor.joblib', help='Path to save/load model')
    # parser.add_argument('--features', default='model_features.joblib', help='Path to save/load model features')

    # args = parser.parse_args()

    # if args.train:
    #     print("Training model...")
    #     train_and_save_model(args.data, args.model, args.features)
    #     print("Training complete.")
    # else:
    #     print("Loading model...")
    #     model, features = load_model_and_features(args.model, args.features)
    #     predict_mechanism(model, features)

    if not os.path.exists(CSV_PATH):
        print(f"Dataset not found at {CSV_PATH}")
        sys.exit(1)
    model, model_features = load_model_and_features(MODEL_PATH, FEATURES_PATH)

    predict_mechanism(model, model_features)