# Quick Start Guide - Gearbox Neural Network

## Installation

1. Install required packages:
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

## Training a New Model

Run the training script:
```bash
python train_gearbox_nn.py
```

This will:
- Generate 5000 training samples
- Train the neural network
- Save the model to `gearbox_nn_model.keras`
- Create training visualizations

Expected time: 2-5 minutes

## Using the Trained Model

### Option 1: Interactive Mode
```bash
python train_gearbox_nn.py --predict
```

Then enter your values when prompted.

### Option 2: Run Examples
```bash
python example_usage.py
```

This shows comprehensive examples with stress validation.

### Option 3: Use in Your Code
```python
import train_gearbox_nn as tg

# Make a prediction
params = tg.predict_gearbox_parameters(
    input_rpm=2000,
    output_rpm=250,
    power_hp=10
)

print(params)
# Output: {'n1': 4.84, 'Pdn1': 7, 'Np1': 54, ...}
```

## What You Get

**Input Parameters:**
- Input RPM
- Output RPM  
- Power (HP)

**Output Parameters:**
- n1: Stage 1 gear ratio
- Pdn1: Stage 1 diametral pitch
- Np1: Stage 1 pinion teeth
- Helix1: Stage 1 helix angle
- Pdn2: Stage 2 diametral pitch
- Np2: Stage 2 pinion teeth
- Helix2: Stage 2 helix angle

## Files

- `train_gearbox_nn.py` - Main training script (extensively commented)
- `example_usage.py` - Usage examples
- `README_NN_TRAINING.md` - Full documentation
- `gearbox_nn_model.keras` - Trained model (generated)
- `gearbox_scaler.npz` - Data scalers (generated)

## Tips

- The model was trained on valid gearbox designs (stresses within limits)
- Predictions should be validated with stress calculations
- More training data = better accuracy
- See `README_NN_TRAINING.md` for advanced usage

## Need Help?

- Check `README_NN_TRAINING.md` for detailed documentation
- Run `example_usage.py` to see working examples
- The code has 500+ lines of beginner-friendly comments
