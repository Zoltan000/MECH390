"""
Neural Network Training Script for Gearbox Parameter Prediction

This script trains a neural network model to predict gearbox parameters for a 
fixed two-stage gearbox based on input/output speeds and power requirements.

INPUTS (What the model receives):
    - Input RPM (wp): Rotational speed at input shaft
    - Output RPM (wf): Desired rotational speed at output shaft  
    - Power (P): Power transmitted through the gearbox in horsepower (HP)

OUTPUTS (What the model predicts):
    - n1: Stage 1 input gear ratio
    - Pdn1: Stage 1 Normal Diametral Pitch (teeth per inch)
    - Np1: Number of teeth in stage 1 pinion
    - Helix1: Helix angle for stage 1 gears (degrees)
    - Pdn2: Stage 2 Normal Diametral Pitch (teeth per inch)
    - Np2: Number of teeth in stage 2 pinion
    - Helix2: Helix angle for stage 2 gears (degrees)

The model is trained using existing stress calculation functions to validate
that predicted parameters result in acceptable stress levels.
"""

# =============================================================================
# IMPORTS - Loading necessary libraries
# =============================================================================

import numpy as np                      # For numerical operations and arrays
import pandas as pd                     # For data handling and CSV operations
import tensorflow as tf                 # Main deep learning framework
from tensorflow import keras            # High-level neural network API
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import StandardScaler      # For normalizing data
import matplotlib.pyplot as plt         # For plotting training results
import itertools                        # For generating parameter combinations
import time                             # For tracking training time
import os                               # For file operations

# Import our existing gearbox calculation functions
import calculations as calc             # Contains stress calculation functions
import functions as fn                  # Contains utility functions like distance()

# =============================================================================
# CONFIGURATION - Setting up training parameters
# =============================================================================

# Random seed for reproducibility (makes results consistent across runs)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Training configuration
EPOCHS = 100                    # Number of times to iterate over the entire dataset
BATCH_SIZE = 32                 # Number of samples processed before model update
VALIDATION_SPLIT = 0.2          # Fraction of data to use for validation (20%)
LEARNING_RATE = 0.001          # How quickly the model learns (smaller = slower but more precise)

# Model save path
MODEL_SAVE_PATH = "gearbox_nn_model.keras"  # Where to save the trained model
SCALER_SAVE_PATH = "gearbox_scaler.npz"     # Where to save the data scalers

# Allowable stress values (these are the target maximum stresses)
ALLOWABLE_BENDING_STRESS = 36.8403    # Maximum bending stress in ksi
ALLOWABLE_CONTACT_STRESS = 129.242     # Maximum contact stress in ksi

# =============================================================================
# STEP 1: DATA GENERATION
# =============================================================================

def generate_training_data(num_samples=10000, save_to_csv=True):
    """
    Generate training data by creating combinations of gearbox parameters
    and calculating their performance characteristics.
    
    This function:
    1. Defines ranges for all gearbox parameters
    2. Generates random combinations within those ranges
    3. Calculates input/output speeds and power for each combination
    4. Filters out invalid combinations (those with excessive stress)
    5. Returns a dataset ready for training
    
    Parameters:
        num_samples (int): Number of data samples to generate
        save_to_csv (bool): Whether to save the dataset to a CSV file
        
    Returns:
        tuple: (X_data, y_data) where X is inputs and y is outputs
    """
    
    print(f"Generating {num_samples} training samples...")
    print("This may take a few minutes...\n")
    
    # Define the ranges for each parameter
    # These ranges are based on typical gearbox design constraints
    parameter_ranges = {
        'n1': (1.0, 9.0),           # Stage 1 gear ratio (dimensionless)
        'Pdn1': [4, 5, 6, 8, 10],   # Discrete diametral pitch values
        'Np1': (10, 100),           # Number of teeth (must be whole number)
        'Helix1': [15, 20, 25],     # Discrete helix angles in degrees
        'Pdn2': [4, 5, 6, 8, 10],   # Stage 2 diametral pitch
        'Np2': (10, 100),           # Stage 2 number of teeth
        'Helix2': [15, 20, 25],     # Stage 2 helix angle
    }
    
    # Operating condition ranges
    wp_range = (1200, 3600)         # Input RPM range
    power_range = (5, 20)           # Power range in HP
    
    # Lists to store generated data
    X_data = []  # Will store [wp, wf, P] for each sample
    y_data = []  # Will store [n1, Pdn1, Np1, Helix1, Pdn2, Np2, Helix2]
    
    valid_samples = 0
    attempts = 0
    max_attempts = num_samples * 10  # Prevent infinite loop
    
    # Generate samples until we have enough valid ones
    while valid_samples < num_samples and attempts < max_attempts:
        attempts += 1
        
        # Randomly generate parameters within specified ranges
        # For continuous ranges, use uniform random sampling
        # For discrete options, randomly choose from the list
        n1 = np.random.uniform(*parameter_ranges['n1'])
        Pdn1 = np.random.choice(parameter_ranges['Pdn1'])
        Np1 = int(np.random.uniform(*parameter_ranges['Np1']))
        Helix1 = np.random.choice(parameter_ranges['Helix1'])
        Pdn2 = np.random.choice(parameter_ranges['Pdn2'])
        Np2 = int(np.random.uniform(*parameter_ranges['Np2']))
        Helix2 = np.random.choice(parameter_ranges['Helix2'])
        
        # Generate operating conditions
        wp = np.random.uniform(*wp_range)     # Input RPM
        P = np.random.uniform(*power_range)    # Power in HP
        
        # Calculate output speed based on the gear ratios
        # wf = wp / (n1 * n2), where n2 is calculated from power and speed
        # For now, we'll use a simplified relationship
        wf = wp / 12 + 100  # Simplified output speed calculation
        
        try:
            # Calculate stresses for Stage 1 using our existing functions
            sigma_b1 = calc.bending_stress(wp, n1, Pdn1, Np1, Helix1)
            sigma_c1 = calc.contact_stress(wp, n1, Pdn1, Np1, Helix1)
            
            # Get important values to calculate n2 for stage 2
            P_calc, Pd, wf_calc, n, n2 = calc.important_values(wp, n1, Pdn1, Np1, Helix1)
            
            # Calculate stresses for Stage 2
            # Stage 2 input speed is the output of Stage 1
            wi = wf_calc  # Intermediate speed between stages
            sigma_b2 = calc.bending_stress(wi, n2, Pdn2, Np2, Helix2)
            sigma_c2 = calc.contact_stress(wi, n2, Pdn2, Np2, Helix2)
            
            # Check if the stresses are within allowable limits
            # We only want to train on "good" designs
            if (sigma_b1 < ALLOWABLE_BENDING_STRESS and 
                sigma_c1 < ALLOWABLE_CONTACT_STRESS and
                sigma_b2 < ALLOWABLE_BENDING_STRESS and 
                sigma_c2 < ALLOWABLE_CONTACT_STRESS):
                
                # This is a valid design! Add it to our dataset
                X_data.append([wp, wf_calc, P_calc])  # Input features
                y_data.append([n1, Pdn1, Np1, Helix1, Pdn2, Np2, Helix2])  # Output parameters
                valid_samples += 1
                
                # Print progress every 1000 samples
                if valid_samples % 1000 == 0:
                    print(f"Generated {valid_samples}/{num_samples} valid samples " +
                          f"(attempts: {attempts}, success rate: {valid_samples/attempts*100:.1f}%)")
                    
        except Exception as e:
            # If calculations fail (divide by zero, etc.), skip this combination
            continue
    
    # Convert lists to numpy arrays for easier manipulation
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    
    print(f"\nSuccessfully generated {valid_samples} valid training samples!")
    print(f"Success rate: {valid_samples/attempts*100:.1f}%\n")
    
    # Optionally save to CSV for inspection
    if save_to_csv:
        # Combine input and output data
        full_data = np.concatenate([X_data, y_data], axis=1)
        columns = ['Input_RPM', 'Output_RPM', 'Power_HP', 
                   'n1', 'Pdn1', 'Np1', 'Helix1', 'Pdn2', 'Np2', 'Helix2']
        df = pd.DataFrame(full_data, columns=columns)
        csv_filename = 'training_data.csv'
        df.to_csv(csv_filename, index=False)
        print(f"Training data saved to: {csv_filename}\n")
    
    return X_data, y_data


# =============================================================================
# STEP 2: DATA PREPROCESSING
# =============================================================================

def preprocess_data(X_data, y_data):
    """
    Prepare the data for neural network training.
    
    Neural networks work best when:
    1. Data is split into training and validation sets
    2. Features are normalized (scaled to similar ranges)
    
    Parameters:
        X_data: Input features (RPM, power)
        y_data: Output parameters (gear parameters)
        
    Returns:
        tuple: (X_train, X_val, y_train, y_val, scaler_X, scaler_y)
    """
    
    print("Preprocessing data...")
    
    # Split data into training and validation sets
    # Training set: used to train the model
    # Validation set: used to check if model generalizes well to new data
    X_train, X_val, y_train, y_val = train_test_split(
        X_data, y_data, 
        test_size=VALIDATION_SPLIT,  # 20% for validation
        random_state=RANDOM_SEED
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}\n")
    
    # Normalize the data using StandardScaler
    # This transforms data to have mean=0 and standard deviation=1
    # This helps the neural network learn faster and more effectively
    
    # Create scalers for inputs and outputs
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Fit scalers on training data and transform both train and validation
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    
    print("Data normalized using StandardScaler")
    print(f"Input features - Mean: {scaler_X.mean_}, Std: {scaler_X.scale_}")
    print(f"Output parameters - Mean: {scaler_y.mean_}, Std: {scaler_y.scale_}\n")
    
    return X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, scaler_X, scaler_y


# =============================================================================
# STEP 3: MODEL ARCHITECTURE
# =============================================================================

def build_model(input_dim, output_dim):
    """
    Build the neural network architecture.
    
    Architecture:
    - Input layer: receives the 3 input features (wp, wf, P)
    - Hidden layers: Multiple layers with ReLU activation to learn complex patterns
    - Output layer: Produces 7 output parameters
    - Dropout layers: Prevent overfitting by randomly dropping neurons during training
    
    Parameters:
        input_dim (int): Number of input features (3: wp, wf, P)
        output_dim (int): Number of output parameters (7: n1, Pdn1, Np1, etc.)
        
    Returns:
        keras.Model: Compiled neural network model
    """
    
    print("Building neural network model...")
    
    # Sequential model: layers are stacked one after another
    model = keras.Sequential([
        # Input layer - explicitly define input shape
        keras.layers.Input(shape=(input_dim,)),
        
        # First hidden layer: 128 neurons
        # Dense = fully connected layer (each neuron connects to all previous neurons)
        # ReLU activation = max(0, x), introduces non-linearity
        keras.layers.Dense(128, activation='relu', name='hidden_layer_1'),
        
        # Dropout layer: randomly sets 20% of neurons to 0 during training
        # This prevents overfitting (memorizing training data)
        keras.layers.Dropout(0.2, name='dropout_1'),
        
        # Second hidden layer: 256 neurons (wider layer to capture more features)
        keras.layers.Dense(256, activation='relu', name='hidden_layer_2'),
        keras.layers.Dropout(0.2, name='dropout_2'),
        
        # Third hidden layer: 128 neurons (narrowing down)
        keras.layers.Dense(128, activation='relu', name='hidden_layer_3'),
        keras.layers.Dropout(0.2, name='dropout_3'),
        
        # Fourth hidden layer: 64 neurons
        keras.layers.Dense(64, activation='relu', name='hidden_layer_4'),
        
        # Output layer: produces 7 values (our gear parameters)
        # Linear activation (no activation function) for regression
        keras.layers.Dense(output_dim, activation='linear', name='output_layer')
    ])
    
    # Compile the model with:
    # - Optimizer: Adam (adaptive learning rate, works well for most problems)
    # - Loss function: MSE (Mean Squared Error) for regression
    # - Metrics: MAE (Mean Absolute Error) for easier interpretation
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',           # Mean Squared Error
        metrics=['mae']       # Mean Absolute Error
    )
    
    # Print model summary to see the architecture
    print("\nModel Architecture:")
    model.summary()
    print()
    
    return model


# =============================================================================
# STEP 4: CUSTOM CALLBACK FOR STRESS VALIDATION
# =============================================================================

class StressValidationCallback(keras.callbacks.Callback):
    """
    Custom callback to validate predictions using stress calculations.
    
    This callback:
    1. Runs after each epoch
    2. Makes predictions on validation data
    3. Checks if predicted parameters result in acceptable stresses
    4. Reports the percentage of valid designs
    
    This ensures our model is learning to predict practical, usable designs.
    """
    
    def __init__(self, X_val, y_val, scaler_X, scaler_y):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each training epoch"""
        
        # Only validate every 10 epochs to save time
        if (epoch + 1) % 10 != 0:
            return
        
        # Make predictions on validation set
        y_pred_scaled = self.model.predict(self.X_val, verbose=0)
        
        # Convert predictions back to original scale
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        X_val_original = self.scaler_X.inverse_transform(self.X_val)
        
        # Count how many predictions result in valid designs
        valid_count = 0
        total_samples = min(100, len(y_pred))  # Check first 100 samples
        
        for i in range(total_samples):
            # Extract predicted parameters (round discrete values)
            n1 = float(y_pred[i, 0])
            Pdn1 = int(round(y_pred[i, 1]))
            Np1 = int(round(y_pred[i, 2]))
            Helix1 = int(round(y_pred[i, 3]))
            Pdn2 = int(round(y_pred[i, 4]))
            Np2 = int(round(y_pred[i, 5]))
            Helix2 = int(round(y_pred[i, 6]))
            
            # Extract input conditions
            wp = float(X_val_original[i, 0])
            
            try:
                # Calculate stresses using predicted parameters
                sigma_b1 = calc.bending_stress(wp, n1, Pdn1, Np1, Helix1)
                sigma_c1 = calc.contact_stress(wp, n1, Pdn1, Np1, Helix1)
                
                # Get n2 for stage 2 calculations
                _, _, _, _, n2 = calc.important_values(wp, n1, Pdn1, Np1, Helix1)
                wf = wp / (n1 * n2)
                
                sigma_b2 = calc.bending_stress(wf, n2, Pdn2, Np2, Helix2)
                sigma_c2 = calc.contact_stress(wf, n2, Pdn2, Np2, Helix2)
                
                # Check if within allowable limits
                if (sigma_b1 < ALLOWABLE_BENDING_STRESS and 
                    sigma_c1 < ALLOWABLE_CONTACT_STRESS and
                    sigma_b2 < ALLOWABLE_BENDING_STRESS and 
                    sigma_c2 < ALLOWABLE_CONTACT_STRESS):
                    valid_count += 1
            except:
                # Invalid combination, skip
                continue
        
        # Report validation results
        valid_percentage = (valid_count / total_samples) * 100
        print(f"  → Stress Validation: {valid_count}/{total_samples} " +
              f"predictions are valid ({valid_percentage:.1f}%)")


# =============================================================================
# STEP 5: TRAINING THE MODEL
# =============================================================================

def train_model(model, X_train, X_val, y_train, y_val, scaler_X, scaler_y):
    """
    Train the neural network model.
    
    Parameters:
        model: The compiled neural network
        X_train, X_val: Training and validation inputs
        y_train, y_val: Training and validation outputs
        scaler_X, scaler_y: Data scalers for inverse transformation
        
    Returns:
        history: Training history object containing loss and metrics
    """
    
    print("Starting model training...\n")
    
    # Define callbacks (functions called during training)
    callbacks = [
        # Early stopping: stop training if validation loss doesn't improve
        keras.callbacks.EarlyStopping(
            monitor='val_loss',      # Metric to monitor
            patience=15,             # Stop after 15 epochs without improvement
            restore_best_weights=True,  # Load the best model weights
            verbose=1
        ),
        
        # Reduce learning rate when validation loss plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,              # Reduce learning rate by half
            patience=5,              # After 5 epochs without improvement
            min_lr=1e-7,             # Minimum learning rate
            verbose=1
        ),
        
        # Our custom stress validation callback
        StressValidationCallback(X_val, y_val, scaler_X, scaler_y)
    ]
    
    # Train the model
    # This is where the actual learning happens!
    history = model.fit(
        X_train, y_train,                    # Training data
        validation_data=(X_val, y_val),      # Validation data
        epochs=EPOCHS,                       # Maximum number of epochs
        batch_size=BATCH_SIZE,               # Samples per gradient update
        callbacks=callbacks,                 # Callbacks defined above
        verbose=1                            # Show progress bar
    )
    
    print("\nTraining completed!")
    
    return history


# =============================================================================
# STEP 6: EVALUATE AND VISUALIZE RESULTS
# =============================================================================

def evaluate_model(model, X_val, y_val, scaler_y, history):
    """
    Evaluate the trained model and visualize results.
    
    Parameters:
        model: Trained neural network
        X_val: Validation inputs
        y_val: Validation outputs
        scaler_y: Output scaler for inverse transformation
        history: Training history object
    """
    
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80 + "\n")
    
    # Make predictions on validation set
    y_pred_scaled = model.predict(X_val, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_val_original = scaler_y.inverse_transform(y_val)
    
    # Calculate metrics for each output parameter
    parameter_names = ['n1', 'Pdn1', 'Np1', 'Helix1', 'Pdn2', 'Np2', 'Helix2']
    
    print("Prediction Accuracy for Each Parameter:")
    print("-" * 80)
    for i, param_name in enumerate(parameter_names):
        # Calculate Mean Absolute Error (MAE) for this parameter
        mae = np.mean(np.abs(y_pred[:, i] - y_val_original[:, i]))
        # Calculate Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_pred[:, i] - y_val_original[:, i]) / 
                              (y_val_original[:, i] + 1e-10))) * 100
        print(f"{param_name:10s} - MAE: {mae:8.4f}, MAPE: {mape:6.2f}%")
    
    print("\n" + "="*80 + "\n")
    
    # Plot training history
    plot_training_history(history)
    
    # Show some example predictions
    show_example_predictions(y_val_original, y_pred, parameter_names, num_examples=5)


def plot_training_history(history):
    """
    Plot training and validation loss over epochs.
    
    Parameters:
        history: Training history object
    """
    
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Model Loss During Training')
    plt.legend()
    plt.grid(True)
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.title('Model MAE During Training')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("Training history plot saved as: training_history.png")
    plt.close()


def show_example_predictions(y_true, y_pred, parameter_names, num_examples=5):
    """
    Display a few example predictions vs actual values.
    
    Parameters:
        y_true: True values
        y_pred: Predicted values
        parameter_names: Names of output parameters
        num_examples: Number of examples to show
    """
    
    print("\nExample Predictions:")
    print("="*80)
    
    for i in range(min(num_examples, len(y_true))):
        print(f"\nExample {i+1}:")
        print("-" * 80)
        print(f"{'Parameter':<10s} {'True Value':>12s} {'Predicted':>12s} {'Error':>12s}")
        print("-" * 80)
        
        for j, param_name in enumerate(parameter_names):
            true_val = y_true[i, j]
            pred_val = y_pred[i, j]
            error = pred_val - true_val
            
            print(f"{param_name:<10s} {true_val:12.2f} {pred_val:12.2f} {error:12.2f}")
    
    print("\n" + "="*80 + "\n")


# =============================================================================
# STEP 7: SAVE THE MODEL
# =============================================================================

def save_model_and_scalers(model, scaler_X, scaler_y):
    """
    Save the trained model and data scalers for future use.
    
    Parameters:
        model: Trained neural network
        scaler_X: Input scaler
        scaler_y: Output scaler
    """
    
    print("Saving model and scalers...")
    
    # Save the model in Keras format
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    
    # Save scalers as numpy arrays
    np.savez(SCALER_SAVE_PATH,
             scaler_X_mean=scaler_X.mean_,
             scaler_X_scale=scaler_X.scale_,
             scaler_y_mean=scaler_y.mean_,
             scaler_y_scale=scaler_y.scale_)
    print(f"Scalers saved to: {SCALER_SAVE_PATH}")
    
    print("\nModel training complete! You can now use the model for predictions.\n")


# =============================================================================
# STEP 8: INFERENCE FUNCTION (Using the Trained Model)
# =============================================================================

def predict_gearbox_parameters(input_rpm, output_rpm, power_hp):
    """
    Use the trained model to predict gearbox parameters.
    
    Parameters:
        input_rpm (float): Input rotational speed in RPM
        output_rpm (float): Desired output rotational speed in RPM
        power_hp (float): Power in horsepower
        
    Returns:
        dict: Dictionary containing predicted gearbox parameters
    """
    
    # Load the saved model and scalers
    model = keras.models.load_model(MODEL_SAVE_PATH)
    
    scaler_data = np.load(SCALER_SAVE_PATH)
    scaler_X = StandardScaler()
    scaler_X.mean_ = scaler_data['scaler_X_mean']
    scaler_X.scale_ = scaler_data['scaler_X_scale']
    
    scaler_y = StandardScaler()
    scaler_y.mean_ = scaler_data['scaler_y_mean']
    scaler_y.scale_ = scaler_data['scaler_y_scale']
    
    # Prepare input
    X_input = np.array([[input_rpm, output_rpm, power_hp]])
    X_input_scaled = scaler_X.transform(X_input)
    
    # Make prediction
    y_pred_scaled = model.predict(X_input_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]
    
    # Round discrete parameters
    parameters = {
        'n1': float(y_pred[0]),
        'Pdn1': int(round(y_pred[1])),
        'Np1': int(round(y_pred[2])),
        'Helix1': int(round(y_pred[3])),
        'Pdn2': int(round(y_pred[4])),
        'Np2': int(round(y_pred[5])),
        'Helix2': int(round(y_pred[6]))
    }
    
    return parameters


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function that orchestrates the entire training process.
    """
    
    print("\n" + "="*80)
    print("NEURAL NETWORK TRAINING FOR GEARBOX PARAMETER PREDICTION")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    # Step 1: Generate training data
    X_data, y_data = generate_training_data(num_samples=5000, save_to_csv=True)
    
    # Step 2: Preprocess the data
    X_train, X_val, y_train, y_val, scaler_X, scaler_y = preprocess_data(X_data, y_data)
    
    # Step 3: Build the model
    input_dim = X_train.shape[1]   # Number of input features (3)
    output_dim = y_train.shape[1]  # Number of output parameters (7)
    model = build_model(input_dim, output_dim)
    
    # Step 4: Train the model
    history = train_model(model, X_train, X_val, y_train, y_val, scaler_X, scaler_y)
    
    # Step 5: Evaluate the model
    evaluate_model(model, X_val, y_val, scaler_y, history)
    
    # Step 6: Save the model
    save_model_and_scalers(model, scaler_X, scaler_y)
    
    # Calculate total training time
    elapsed_time = time.time() - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    # Step 7: Show example usage
    print("\n" + "="*80)
    print("EXAMPLE USAGE")
    print("="*80 + "\n")
    print("To use the trained model for prediction, run:")
    print("python train_gearbox_nn.py --predict")
    print("\nOr use the predict_gearbox_parameters() function in your code:")
    print("params = predict_gearbox_parameters(input_rpm=2000, output_rpm=250, power_hp=10)")
    print("\n" + "="*80 + "\n")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Check if user wants to make a prediction
    if len(sys.argv) > 1 and sys.argv[1] == "--predict":
        # Interactive prediction mode
        print("\n" + "="*80)
        print("GEARBOX PARAMETER PREDICTION")
        print("="*80 + "\n")
        
        try:
            # Get user input
            input_rpm = float(input("Enter input RPM (1200-3600): "))
            output_rpm = float(input("Enter output RPM (100-500): "))
            power_hp = float(input("Enter power in HP (5-20): "))
            
            # Make prediction
            params = predict_gearbox_parameters(input_rpm, output_rpm, power_hp)
            
            # Display results
            print("\nPredicted Gearbox Parameters:")
            print("-" * 80)
            for key, value in params.items():
                print(f"{key:10s}: {value}")
            print("-" * 80 + "\n")
            
        except FileNotFoundError:
            print("\nError: Model not found. Please train the model first by running:")
            print("python train_gearbox_nn.py\n")
        except Exception as e:
            print(f"\nError: {e}\n")
    else:
        # Training mode (default)
        main()
