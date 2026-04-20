"""
Car Evaluation AI System using Naive Bayes
CS3431 - Machine Problem 2
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# ── 1. Column names and valid attribute values ──────────────────────────────
COLUMNS = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

VALID_VALUES = {
    'buying':   ['low', 'med', 'high', 'vhigh'],
    'maint':    ['low', 'med', 'high', 'vhigh'],
    'doors':    ['2', '3', '4', '5more'],
    'persons':  ['2', '4', 'more'],
    'lug_boot': ['small', 'med', 'big'],
    'safety':   ['low', 'med', 'high'],
}

FEATURE_COLS = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

MODEL_FILE = 'car_model.pkl'


# ── 2. Load and encode data ──────────────────────────────────────────────────
def load_data(filepath='car.data'):
    df = pd.read_csv(filepath, header=None, names=COLUMNS)
    return df


def build_encoder():
    """Build an OrdinalEncoder using the known categories."""
    categories = [VALID_VALUES[c] for c in FEATURE_COLS]
    enc = OrdinalEncoder(categories=categories,
                         handle_unknown='use_encoded_value',
                         unknown_value=-1)
    return enc


def train_model(filepath='car.data'):
    print("=" * 55)
    print("  Training Naive Bayes model on car.data …")
    print("=" * 55)

    df = load_data(filepath)
    X = df[FEATURE_COLS]
    y = df['class']

    # Encode features
    encoder = build_encoder()
    X_enc = encoder.fit_transform(X)

    # Train / test split  (80 / 20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"  Training samples : {len(X_train)}")
    print(f"  Test samples     : {len(X_test)}")

    # Categorical Naive Bayes (correct choice for discrete/categorical data)
    model = CategoricalNB()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy : {acc * 100:.2f}%\n")
    print("  Classification Report:")
    print(classification_report(y_test, y_pred))

    # Persist model + encoder
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({'model': model, 'encoder': encoder}, f)
    print(f"  Model saved to '{MODEL_FILE}'")
    return model, encoder


def load_saved_model():
    with open(MODEL_FILE, 'rb') as f:
        bundle = pickle.load(f)
    return bundle['model'], bundle['encoder']


# ── 3. Interactive prediction ────────────────────────────────────────────────
def get_input(prompt, valid):
    opts = ' / '.join(valid)
    while True:
        val = input(f"  {prompt} [{opts}]: ").strip().lower()
        if val in valid:
            return val
        print(f"    ✗ Invalid. Choose from: {opts}")


def predict_car(model, encoder):
    print("\n" + "=" * 55)
    print("  Enter the car attributes to evaluate:")
    print("=" * 55)

    user_input = {
        'buying':   get_input("Buying price   ", VALID_VALUES['buying']),
        'maint':    get_input("Maintenance    ", VALID_VALUES['maint']),
        'doors':    get_input("No. of doors   ", VALID_VALUES['doors']),
        'persons':  get_input("Persons (cap.) ", VALID_VALUES['persons']),
        'lug_boot': get_input("Luggage boot   ", VALID_VALUES['lug_boot']),
        'safety':   get_input("Safety         ", VALID_VALUES['safety']),
    }

    row = pd.DataFrame([user_input], columns=FEATURE_COLS)
    X_enc = encoder.transform(row)
    prediction = model.predict(X_enc)[0]
    proba = model.predict_proba(X_enc)[0]
    classes = model.classes_

    label_map = {
        'unacc': 'Unacceptable',
        'acc':   'Acceptable',
        'good':  'Good',
        'vgood': 'Very Good',
    }

    print("\n" + "=" * 55)
    print(f"  ▶  Car Condition : {label_map.get(prediction, prediction).upper()}")
    print("=" * 55)
    print("  Probability breakdown:")
    for cls, prob in sorted(zip(classes, proba), key=lambda x: -x[1]):
        bar = '█' * int(prob * 30)
        print(f"    {label_map.get(cls, cls):12s}  {prob * 100:5.1f}%  {bar}")
    print()


# ── 4. Main ──────────────────────────────────────────────────────────────────
def main():
    data_file = 'car.data'

    # Train if no saved model, or if data file present alongside script
    if not os.path.exists(MODEL_FILE):
        if not os.path.exists(data_file):
            print(f"ERROR: '{data_file}' not found. Place car.data in the same folder.")
            return
        model, encoder = train_model(data_file)
    else:
        print(f"Loading saved model from '{MODEL_FILE}' …")
        model, encoder = load_saved_model()

    while True:
        predict_car(model, encoder)
        again = input("  Evaluate another car? (yes/no): ").strip().lower()
        if again not in ('yes', 'y'):
            print("\n  Goodbye!\n")
            break


if __name__ == '__main__':
    main()
