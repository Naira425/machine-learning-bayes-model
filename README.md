
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
