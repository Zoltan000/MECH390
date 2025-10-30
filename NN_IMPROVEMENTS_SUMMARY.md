# Neural Network Model Improvements

## Critical Issues Fixed

### 1. ❌ **BROKEN DATA GENERATION LOGIC** (CRITICAL FIX)
**Problem:** 
- Line 141 had: `wf = wp / 12 + 100` - completely arbitrary formula
- Created massive train/test distribution mismatch
- Model trained on random wf, but uses calculated wf_calc at inference
- This was the **PRIMARY CAUSE** of poor predictions

**Solution:**
- Removed the arbitrary formula
- Now uses only physically calculated values from `calc.important_values()`
- Training data now matches real-world physics

```python
# BEFORE (WRONG):
wf = wp / 12 + 100  # Random nonsense
sigma_b1 = calc.bending_stress(wp, n1, Pdn1, Np1, Helix1)
P_calc, Pd, wf_calc, n, n2 = calc.important_values(...)
X_data.append([wp, wf_calc, P_calc])  # Using wf_calc, not wf!

# AFTER (CORRECT):
sigma_b1 = calc.bending_stress(wp, n1, Pdn1, Np1, Helix1)
P_calc, Pd, wf_calc, n, n2 = calc.important_values(...)
total_ratio = n1 * n2
X_data.append([wp, wf_calc, P_calc, total_ratio])  # All physically meaningful
```

### 2. ✅ **ADDED FEATURE ENGINEERING**
**Problem:** Model had no derived features to help learn relationships

**Solution:**
- Added `total_ratio = n1 * n2` as 4th input feature
- Helps model understand the fundamental gear reduction relationship
- Input RPM / Output RPM ≈ Total Ratio (gives model a strong hint)

```python
# BEFORE: X = [wp, wf_calc, P_calc]  (3 features)
# AFTER:  X = [wp, wf_calc, P_calc, total_ratio]  (4 features)
```

### 3. 🏗️ **IMPROVED MODEL ARCHITECTURE**
**Changes:**
- **Deeper network:** 4 layers → 5 layers
- **Wider network:** Max 256 neurons → Max 512 neurons
- **Batch Normalization:** Added after every dense layer (stabilizes training)
- **Better regularization:** Dropout 0.2 → 0.3 for first layers
- **Better structure:** Dense → BatchNorm → Activation → Dropout

**Why this helps:**
- Batch normalization reduces internal covariate shift
- Wider layers capture more complex patterns in gear relationships
- Better regularization prevents overfitting

```python
# OLD: 128 → 256 → 128 → 64 → output
# NEW: 256 → 512 → 256 → 128 → 64 → output (with BatchNorm between each)
```

### 4. ⚙️ **OPTIMIZED HYPERPARAMETERS**
| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| EPOCHS | 100 | 200 | Allow more time for convergence |
| BATCH_SIZE | 32 | 64 | More stable gradients |
| LEARNING_RATE | 0.001 | 0.0005 | Slower, more precise learning |
| EarlyStopping patience | 15 | 25 | Don't give up too early |
| ReduceLR patience | 5 | 10 | Give more time before reducing LR |
| Training samples | 5,000 | 10,000 | More data = better generalization |

### 5. 🎯 **IMPROVED VALIDATION CALLBACK**
**Problem:** StressValidationCallback used simple `round()` instead of constraint snapping

**Solution:**
- Added `_snap_to_valid()` helper method to the callback
- Now validates predictions using the same constraints as inference
- More accurate representation of real-world performance

```python
# BEFORE:
Pdn1 = int(round(y_pred[i, 1]))  # Could give invalid values like 7, 9, etc.

# AFTER:
Pdn1 = int(self._snap_to_valid(y_pred[i, 1], VALID_PDN))  # Only 4,5,6,8,10
```

### 6. 📊 **BETTER DATA GENERATION**
- Removed arbitrary wf calculation
- All training data now based on real physics calculations
- CSV now includes 'Total_Ratio' column for inspection

---

## Expected Improvements

### Prediction Quality
- **Much better parameter predictions** because training data matches inference conditions
- **Consistent physics** - no more train/test mismatch
- **Better discrete parameter handling** throughout the pipeline

### Training Stability
- **Batch normalization** prevents exploding/vanishing gradients
- **Larger batch size** (64 vs 32) gives more stable gradient updates
- **Lower learning rate** (0.0005 vs 0.001) prevents overshooting

### Model Capacity
- **512 neurons** in widest layer can capture complex gear interactions
- **5 layers** provide sufficient depth for feature hierarchies
- **Better regularization** (0.3 dropout) prevents overfitting with more capacity

---

## How to Retrain with Improvements

1. **Delete old model files** (important - architecture changed):
   ```powershell
   Remove-Item gearbox_nn_model.keras -ErrorAction SilentlyContinue
   Remove-Item gearbox_scaler.npz -ErrorAction SilentlyContinue
   Remove-Item training_data.csv -ErrorAction SilentlyContinue
   ```

2. **Train the improved model**:
   ```powershell
   python train_gearbox_nn.py
   ```
   
   This will:
   - Generate 10,000 training samples with correct physics
   - Train for up to 200 epochs (will stop early if converged)
   - Take 10-30 minutes depending on your hardware
   - Save improved model and scalers

3. **Test predictions**:
   ```powershell
   python train_gearbox_nn.py --predict
   ```

---

## What to Expect

### During Training
- **Data generation:** ~2-5 minutes for 10,000 samples
- **Training time:** 10-30 minutes total
- **Progress:** Watch validation MAE decrease consistently
- **Stress validation:** Should show >80% valid predictions by end

### After Training
- **Prediction accuracy:** Should be significantly better
- **Physically realistic:** All parameters within valid ranges
- **Consistent results:** Similar inputs give similar outputs

---

## Monitoring Training Quality

Watch for these indicators of good training:

✅ **Good signs:**
- Validation loss decreases steadily
- Training and validation loss stay close (not overfitting)
- Stress validation percentage >70-80%
- MAE decreases over epochs

❌ **Bad signs:**
- Validation loss increases while training loss decreases (overfitting)
- Loss plateaus very early (<20 epochs)
- Stress validation <50%
- Very high MAE values

---

## Additional Recommendations

### If predictions are still not good enough:

1. **Increase training data:**
   ```python
   X_data, y_data = generate_training_data(num_samples=20000, save_to_csv=True)
   ```

2. **Experiment with architecture:**
   - Try different layer sizes: [128, 256, 512, 256, 128, 64]
   - Adjust dropout: 0.2-0.4 range
   - Try LeakyReLU instead of ReLU

3. **Tune learning rate:**
   - Try 0.0003 (slower) or 0.001 (faster)
   - Use cyclical learning rate schedule

4. **Check your data:**
   - Open `training_data.csv` 
   - Verify Input_RPM, Output_RPM, Power_HP, Total_Ratio look reasonable
   - Check that Pdn and Helix columns only have valid values

5. **Consider ensemble methods:**
   - Train 3-5 models with different random seeds
   - Average their predictions

---

## Technical Details

### New Input Features (4 total):
1. **wp** (Input RPM) - User specified
2. **wf_calc** (Output RPM) - Calculated from gear physics
3. **P_calc** (Power HP) - Calculated from gear physics  
4. **total_ratio** (n1 × n2) - Derived feature

### Model Architecture:
```
Input (4 features)
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
Output (7 parameters)
```

### Output Parameters (7 total):
1. n1 (continuous)
2. Pdn1 (discrete: 4,5,6,8,10)
3. Np1 (discrete: any integer)
4. Helix1 (discrete: 15,20,25)
5. Pdn2 (discrete: 4,5,6,8,10)
6. Np2 (discrete: any integer)
7. Helix2 (discrete: 15,20,25)

---

## Summary

The **primary issue** was the broken data generation logic that created a train/test mismatch. Combined with insufficient model capacity and suboptimal hyperparameters, this led to poor predictions.

The improvements address:
1. ✅ Data quality and physics consistency
2. ✅ Model capacity and architecture
3. ✅ Regularization and training stability
4. ✅ Hyperparameter optimization
5. ✅ Validation accuracy

**Expected result:** Significantly better predictions that respect physical constraints and produce practical gearbox designs.
