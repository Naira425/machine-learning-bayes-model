
# Car Advisor

A simple AI system that evaluates car acceptability using machine learning.

## What You Need

- Python 3.8 or higher
- Libraries: scikit-learn, pandas, numpy

## Installation

Install the required libraries:

```bash
pip install scikit-learn pandas numpy
```

## How to Run

1. Place `car.data` in the same folder as `car_advisor.py`
2. Run the program:

```bash
python car_advisor.py
```

The system will train the model on first run (takes a few seconds), then prompt you to enter car details for evaluation.

## Model Training

The car advisor uses a Categorical Naive Bayes classifier from scikit-learn to evaluate car acceptability based on categorical features. Here's how the model is trained:

### Training Process

1. **Data Loading**: The system loads the `car.data` file, which contains 1728 car examples with 6 categorical features and a target class (unacc, acc, good, vgood).

2. **Feature Encoding**: Categorical features are encoded using scikit-learn's `OrdinalEncoder` with predefined categories for each feature:
   - Buying price: low, med, high, vhigh
   - Maintenance: low, med, high, vhigh
   - Number of doors: 2, 3, 4, 5more
   - Persons capacity: 2, 4, more
   - Luggage boot: small, med, big
   - Safety: low, med, high

3. **Data Splitting**: The dataset is split into training and test sets using an 80/20 stratified split to maintain class distribution.

4. **Model Training**: A Categorical Naive Bayes model is trained on the encoded training data. This algorithm is suitable for categorical features as it calculates probabilities for each class given the feature values.

5. **Model Evaluation**: The trained model is evaluated on the test set, reporting accuracy and a detailed classification report showing precision, recall, and F1-score for each class.

6. **Model Persistence**: The trained model and encoder are saved to `car_model.pkl` for future use, avoiding retraining on subsequent runs.

### Model Performance

The model typically achieves around 85-90% accuracy on the test set, with good performance across all classes. The system provides probability estimates for each prediction, showing the confidence level for each possible car rating.

## Example

```
Buying price   [low / med / high / vhigh]: low
Maintenance    [low / med / high / vhigh]: med
No. of doors   [2 / 3 / 4 / 5more]: 4
Persons (cap.) [2 / 4 / more]: more
Luggage boot   [small / med / big]: big
Safety         [low / med / high]: high

▶  Car Condition : ACCEPTABLE
```

## Files

- `car_advisor.py` - Main program
- `car.data` - Training data (1728 car examples)
- `README.md` - This file
