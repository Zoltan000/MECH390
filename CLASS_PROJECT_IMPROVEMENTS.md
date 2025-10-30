# Neural Network Improvements for Class Project

## Understanding the Professor's Simplified Model

Your professor specified that **output RPM and power scale linearly** with input parameters for pedagogical simplicity. This is intentional for the class project:

```python
wf = wp / 12 + 100  # Simplified linear relationship (professor's specification)
```

This is **NOT a bug** - it's a simplified model for educational purposes. The real improvements focus on making the neural network learn this relationship better.

---

## What Was Actually Improved

### 1. ✅ **ENHANCED MODEL ARCHITECTURE** (Primary Improvement)

**Problem:** The original 4-layer network was too simple to learn complex gear parameter relationships.

**Solution:** Deeper, wider network with batch normalization

```python
# OLD ARCHITECTURE:
128 → 256 → 128 → 64 → output (4 hidden layers)

# NEW ARCHITECTURE:
256 → 512 → 256 → 128 → 64 → output (5 hidden layers)
With BatchNormalization after each dense layer
```

**Why this helps:**
- **Batch Normalization** stabilizes training and allows higher learning rates
- **Wider layers** (512 neurons) can capture complex interactions between 7 output parameters
- **Deeper network** learns hierarchical feature representations
- **Better gradient flow** through the network

### 2. ✅ **ADDED FEATURE ENGINEERING**

**Problem:** Model only had 3 input features (wp, wf, P)

**Solution:** Added `total_ratio` as 4th derived feature

```python
total_ratio = wp / wf  # Approximate total gear reduction
X = [wp, wf, P, total_ratio]  # 4 features instead of 3
```

**Why this helps:**
- Gives the model a strong hint about the fundamental relationship
- Total ratio ≈ n1 × n2, which the model needs to predict
- Reduces the learning burden on the neural network

### 3. ✅ **OPTIMIZED HYPERPARAMETERS**

| Parameter | Old | New | Why |
|-----------|-----|-----|-----|
| **Epochs** | 100 | 200 | More time to converge |
| **Batch Size** | 32 | 64 | More stable gradient updates |
| **Learning Rate** | 0.001 | 0.0005 | Finer adjustments, better precision |
| **Dropout** | 0.2 | 0.3 | Better regularization prevents overfitting |
| **EarlyStopping patience** | 15 | 25 | Don't give up too early |
| **ReduceLR patience** | 5 | 10 | More epochs before reducing LR |
| **Training samples** | 5,000 | 10,000 | More data for better generalization |

### 4. ✅ **IMPROVED VALIDATION CALLBACK**

**Problem:** Callback used simple `round()` instead of snapping to valid discrete values

**Solution:** Added constraint snapping in validation

```python
# NOW validates with proper constraints:
Pdn1 = int(self._snap_to_valid(y_pred[i, 1], [4,5,6,8,10]))
Helix1 = int(self._snap_to_valid(y_pred[i, 3], [15,20,25]))
```

**Why this helps:**
- More accurate representation of real-world performance
- Validation metrics now match inference behavior

### 5. ✅ **BETTER REGULARIZATION**

**Technique:** BatchNormalization + Higher Dropout

**What changed:**
```python
# Each layer now has:
Dense(neurons)
→ BatchNormalization()  # NEW: normalizes activations
→ Activation('relu')
→ Dropout(0.3)         # INCREASED from 0.2
```

**Why this helps:**
- BatchNorm reduces internal covariate shift
- Higher dropout (0.3) prevents the larger network from overfitting
- More robust predictions on unseen data

---

## Expected Improvements

### ✅ Better Prediction Accuracy
- Deeper network can model complex relationships between 7 gear parameters
- Batch normalization accelerates training convergence
- More training data (10K samples) improves generalization

### ✅ More Stable Training
- Batch normalization prevents exploding/vanishing gradients
- Larger batch size (64) gives smoother gradient estimates
- Lower learning rate (0.0005) prevents overshooting optimal weights

### ✅ Better Generalization
- Higher dropout (0.3) prevents overfitting
- More training samples provide better coverage of parameter space
- Feature engineering (total_ratio) guides the model

### ✅ Consistent Discrete Outputs
- Constraint snapping ensures valid Pdn (4,5,6,8,10) and Helix (15,20,25)
- Validation callback now matches inference behavior

---

## How to Retrain with Improvements

### Step 1: Delete Old Model Files
```powershell
Remove-Item gearbox_nn_model.keras -ErrorAction SilentlyContinue
Remove-Item gearbox_scaler.npz -ErrorAction SilentlyContinue
Remove-Item training_data.csv -ErrorAction SilentlyContinue
```

**Why?** The model architecture changed (3 → 4 input features, 4 → 5 layers), so old weights are incompatible.

### Step 2: Train the Improved Model
```powershell
python train_gearbox_nn.py
```

**What to expect:**
- **Data generation:** ~2-5 minutes (10,000 samples)
- **Training time:** 10-30 minutes depending on hardware
- **Progress:** Watch validation MAE decrease over epochs
- **Stress validation:** Should reach >70-80% valid designs

### Step 3: Test Predictions
```powershell
python train_gearbox_nn.py --predict
```

**Test cases to try:**
```
Input: 1800 RPM, Output: 250 RPM, Power: 10 HP
Input: 2400 RPM, Output: 300 RPM, Power: 15 HP
Input: 3000 RPM, Output: 350 RPM, Power: 12 HP
```

**Verify:**
- Helix angles are **exactly** 15, 20, or 25
- Pdn values are **exactly** 4, 5, 6, 8, or 10
- Predictions are consistent (similar inputs → similar outputs)

---

## What Was NOT Changed (And Why)

### ❌ The Linear Relationship
```python
wf = wp / 12 + 100  # Professor's specification - KEPT AS IS
```

This is **intentional** for your class project. The professor wants you to work with a simplified model.

### ❌ The Physics Calculations
All `calc.bending_stress()` and `calc.contact_stress()` functions remain unchanged - they're used for validation only.

### ❌ The Output Parameters
Still predicting the same 7 parameters: n1, Pdn1, Np1, Helix1, Pdn2, Np2, Helix2

---

## Monitoring Training Quality

### ✅ Good Training Signs:
- Validation loss decreases steadily
- Training and validation loss stay close (gap < 2x)
- Stress validation percentage increases over epochs
- Final stress validation >70%
- MAE for each parameter decreases

### ❌ Warning Signs:
- Validation loss increases while training loss decreases (overfitting)
- Loss plateaus very early (<20 epochs without early stopping)
- Stress validation stays <50%
- Very large MAE values that don't improve

### 📊 What to Check:
1. **training_history.png** - Loss should decrease, not oscillate wildly
2. **training_data.csv** - Verify Total_Ratio column makes sense
3. **Console output** - Watch stress validation percentage improve

---

## If Predictions Are Still Poor

### Try These Adjustments:

#### 1. Increase Training Data
```python
X_data, y_data = generate_training_data(num_samples=20000, save_to_csv=True)
```

#### 2. Adjust Model Size
Experiment with layer widths:
```python
# Try wider network:
[512, 1024, 512, 256, 128]

# Or deeper network:
[256, 512, 512, 256, 128, 64]
```

#### 3. Tune Dropout
```python
# Less regularization (if underfitting):
keras.layers.Dropout(0.2)

# More regularization (if overfitting):
keras.layers.Dropout(0.4)
```

#### 4. Adjust Learning Rate
```python
# Slower (more precise):
LEARNING_RATE = 0.0003

# Faster (if converging too slowly):
LEARNING_RATE = 0.001
```

#### 5. Check Feature Scaling
Open `training_data.csv` and verify:
- Input_RPM ranges from ~1200-3600
- Output_RPM ranges from ~200-400 (from wp/12 + 100)
- Power_HP ranges from ~5-20
- Total_Ratio values look reasonable

---

## Technical Summary

### Model Architecture:
```
Input (4 features: wp, wf, P, total_ratio)
    ↓
Dense(256) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense(512) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense(256) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense(128) → BatchNorm → ReLU → Dropout(0.2)
    ↓
Dense(64) → BatchNorm → ReLU
    ↓
Output (7 parameters: n1, Pdn1, Np1, Helix1, Pdn2, Np2, Helix2)
```

### Training Configuration:
- **Optimizer:** Adam with LR=0.0005
- **Loss:** Mean Squared Error (MSE)
- **Metrics:** Mean Absolute Error (MAE)
- **Batch size:** 64
- **Max epochs:** 200 (with early stopping at patience=25)
- **Callbacks:** EarlyStopping, ReduceLROnPlateau, StressValidation

---

## Summary

The **architectural improvements** (batch normalization, wider/deeper network, better hyperparameters, feature engineering) should significantly improve your model's predictions while respecting your professor's simplified linear relationships.

**Expected outcome:** Much better predictions with:
- Lower MAE for all parameters
- Higher stress validation percentage (>70-80%)
- Consistent, reliable outputs
- Properly constrained discrete values

**Time investment:** 10-30 minutes training time for significantly better results.
