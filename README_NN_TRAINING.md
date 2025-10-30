# Neural Network Training Script for Gearbox Parameter Prediction

## Overview

This neural network model predicts optimal gearbox parameters for a fixed two-stage gearbox based on operating conditions.

## What the Model Does

**Inputs (What you provide):**
- Input RPM (wp): Input shaft rotational speed
- Output RPM (wf): Desired output shaft speed
- Power (P): Power transmitted through gearbox in HP

**Outputs (What the model predicts):**
- n1: Stage 1 input gear ratio
- Pdn1: Stage 1 Normal Diametral Pitch (teeth/inch)
- Np1: Number of teeth in stage 1 pinion
- Helix1: Helix angle for stage 1 gears (degrees)
- Pdn2: Stage 2 Normal Diametral Pitch (teeth/inch)
- Np2: Number of teeth in stage 2 pinion
- Helix2: Helix angle for stage 2 gears (degrees)

## Files

- `train_gearbox_nn.py` - Main training script with extensive comments
- `gearbox_nn_model.keras` - Trained neural network model (created after training)
- `gearbox_scaler.npz` - Data scalers for normalization (created after training)
- `training_data.csv` - Generated training dataset
- `training_history.png` - Plot showing model training progress

## Requirements

Install the required packages:
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

The script also requires the existing gearbox calculation modules:
- `calculations.py` - Contains stress calculation functions
- `functions.py` - Contains utility functions
- `constants.py` - Contains gearbox constants
- `lookupTables.py` - Contains lookup tables

## Usage

### Training the Model

To train a new model from scratch:

```bash
python train_gearbox_nn.py
```

This will:
1. Generate training data (5000 samples by default)
2. Build and train a neural network
3. Validate the model using stress calculations
4. Save the trained model and scalers
5. Generate visualizations of training progress

### Using the Trained Model for Predictions

After training, you can use the model to predict gearbox parameters:

```bash
python train_gearbox_nn.py --predict
```

This will prompt you to enter:
- Input RPM
- Output RPM
- Power in HP

And it will output the predicted gearbox parameters.

### Using in Your Own Code

```python
import train_gearbox_nn as tg

# Make a prediction
params = tg.predict_gearbox_parameters(
    input_rpm=2000,
    output_rpm=250,
    power_hp=10
)

print(params)
# Output: {'n1': 4.84, 'Pdn1': 7, 'Np1': 54, 'Helix1': 20, 
#          'Pdn2': 6, 'Np2': 61, 'Helix2': 20}
```

## How It Works

### 1. Data Generation
The script generates training data by:
- Randomly sampling gearbox parameters within valid ranges
- Calculating operating conditions (RPM, power)
- Using existing stress calculation functions to validate designs
- Only keeping designs where stresses are within allowable limits

### 2. Neural Network Architecture
The model uses a deep neural network with:
- Input layer: 3 features (Input RPM, Output RPM, Power)
- Hidden layers: 4 layers with 128, 256, 128, and 64 neurons
- Dropout layers: To prevent overfitting
- Output layer: 7 parameters (gear ratios, teeth, angles, etc.)
- Activation: ReLU for hidden layers, linear for output

### 3. Training Process
- The model learns from 80% of data (training set)
- Validates on 20% of data (validation set)
- Uses Mean Squared Error (MSE) as loss function
- Implements early stopping to prevent overfitting
- Reduces learning rate when progress plateaus
- Validates predictions using stress calculations every 10 epochs

### 4. Model Validation
After each epoch, the model checks if predicted parameters result in:
- Bending stress < 36.84 ksi
- Contact stress < 129.24 ksi

This ensures the model learns to predict practical, usable designs.

## Configuration

You can modify these parameters in `train_gearbox_nn.py`:

```python
EPOCHS = 100              # Number of training iterations
BATCH_SIZE = 32           # Samples per batch
VALIDATION_SPLIT = 0.2    # Validation data percentage
LEARNING_RATE = 0.001     # Learning rate
```

To generate more/fewer training samples, modify the `main()` function:
```python
X_data, y_data = generate_training_data(num_samples=10000)  # Change 10000
```

## Understanding the Output

### Training Progress
During training, you'll see:
- Epoch number and progress bar
- Loss (MSE): Lower is better
- MAE (Mean Absolute Error): Average prediction error
- Validation metrics: Performance on unseen data
- Stress validation: % of predictions that are valid designs

### Final Evaluation
After training completes, the script shows:
- Prediction accuracy for each parameter (MAE and MAPE)
- Training history plots (loss and MAE over time)
- Example predictions vs actual values

## Tips for Beginners

1. **What is a Neural Network?**
   - Think of it as a "function learner" - it learns the pattern between inputs and outputs from examples

2. **What is Training?**
   - The process where the network adjusts its internal parameters to minimize prediction errors

3. **What is an Epoch?**
   - One complete pass through all training data

4. **What is Validation?**
   - Testing the model on data it hasn't seen during training to ensure it generalizes well

5. **What is Overfitting?**
   - When the model memorizes training data instead of learning patterns (prevented by dropout and early stopping)

## Troubleshooting

**Error: "Model not found"**
- Train the model first by running `python train_gearbox_nn.py`

**Error: "divide by zero" during training**
- Some parameter combinations cause division by zero in stress calculations
- These are automatically skipped during data generation

**Poor prediction accuracy**
- Try generating more training data (increase `num_samples`)
- Train for more epochs (increase `EPOCHS`)
- The model may need more data for certain parameter ranges

## Advanced Usage

### Custom Training Configuration

```python
import train_gearbox_nn as tg

# Generate custom dataset
X_data, y_data = tg.generate_training_data(
    num_samples=20000,  # More data = better model
    save_to_csv=True
)

# Preprocess
X_train, X_val, y_train, y_val, scaler_X, scaler_y = tg.preprocess_data(X_data, y_data)

# Build model with custom architecture
# (You can modify the build_model function)
model = tg.build_model(input_dim=3, output_dim=7)

# Train with custom settings
history = tg.train_model(model, X_train, X_val, y_train, y_val, scaler_X, scaler_y)

# Save
tg.save_model_and_scalers(model, scaler_X, scaler_y)
```

## References

- TensorFlow/Keras documentation: https://www.tensorflow.org/
- Gearbox stress calculations based on AGMA standards
- Neural network architecture based on regression best practices
