# Gearbox NN Prediction Constraint Fix - Summary

## Problem
The neural network was predicting values outside the allowable discrete ranges:
- **Helix angles**: Model predicted values like 22°, but only [15, 20, 25] are valid
- **Diametral pitch (Pdn)**: Model predicted arbitrary integers, but only [4, 5, 6, 8, 10] are valid

The original code used simple `round()` which doesn't constrain to specific discrete options.

## Solution
Added a constraint function that snaps predicted continuous values to the nearest valid discrete option:

### 1. New Helper Function: `snap_to_valid_values()`
```python
def snap_to_valid_values(predicted_value, valid_options):
    """
    Snap a predicted continuous value to the nearest valid discrete option.
    Finds the closest valid option to the model's prediction.
    """
    valid_options = np.array(valid_options)
    distances = np.abs(valid_options - predicted_value)
    closest_idx = np.argmin(distances)
    return valid_options[closest_idx]
```

### 2. Updated `predict_gearbox_parameters()` Function
**Before:**
```python
parameters = {
    'n1': float(y_pred[0]),
    'Pdn1': int(round(y_pred[1])),      # ❌ Could produce any integer
    'Np1': int(round(y_pred[2])),
    'Helix1': int(round(y_pred[3])),    # ❌ Could produce any integer
    'Pdn2': int(round(y_pred[4])),      # ❌ Could produce any integer
    'Np2': int(round(y_pred[5])),
    'Helix2': int(round(y_pred[6]))     # ❌ Could produce any integer
}
```

**After:**
```python
# Define valid discrete values
VALID_PDN = [4, 5, 6, 8, 10]      # Valid diametral pitch options
VALID_HELIX = [15, 20, 25]         # Valid helix angle options (degrees)

# Constrain discrete parameters to valid values
parameters = {
    'n1': float(y_pred[0]),
    'Pdn1': int(snap_to_valid_values(y_pred[1], VALID_PDN)),      # ✓ Constrained
    'Np1': int(round(y_pred[2])),
    'Helix1': int(snap_to_valid_values(y_pred[3], VALID_HELIX)),  # ✓ Constrained
    'Pdn2': int(snap_to_valid_values(y_pred[4], VALID_PDN)),      # ✓ Constrained
    'Np2': int(round(y_pred[5])),
    'Helix2': int(snap_to_valid_values(y_pred[6], VALID_HELIX))   # ✓ Constrained
}
```

## How It Works
The `snap_to_valid_values()` function:
1. Takes the model's continuous prediction (e.g., 22.3 for helix angle)
2. Calculates the distance to each valid option: |22.3 - 15| = 7.3, |22.3 - 20| = 2.3, |22.3 - 25| = 2.7
3. Returns the closest valid option: 20° (smallest distance)

## Test Results
Created `test_constraints.py` to verify the function works correctly:

**Helix Angle Examples:**
- Prediction: 14 → Output: 15 ✓
- Prediction: 17 → Output: 15 ✓
- Prediction: 22 → Output: 20 ✓ (fixes your reported issue!)
- Prediction: 24 → Output: 25 ✓

**Diametral Pitch Examples:**
- Prediction: 4.2 → Output: 4 ✓
- Prediction: 7.0 → Output: 6 ✓
- Prediction: 9.0 → Output: 8 ✓

## Impact on Model
- ✓ **No retraining required** - this is a post-processing constraint
- ✓ Training data already uses these discrete values, so the model learns the correct patterns
- ✓ The constraint simply ensures the output respects physical/manufacturing limitations
- ✓ `Np1` and `Np2` (number of teeth) still use `round()` since any integer is valid

## Files Modified
1. **train_gearbox_nn.py** (lines 615-680)
   - Added `snap_to_valid_values()` helper function
   - Updated `predict_gearbox_parameters()` to use constraints

## Next Steps
The model will now **always** predict helix angles of exactly 15°, 20°, or 25°, and diametral pitch values of exactly 4, 5, 6, 8, or 10. No more out-of-range values!

You can test this immediately by running:
```powershell
python train_gearbox_nn.py --predict
```

If you haven't trained the model yet, train it first with:
```powershell
python train_gearbox_nn.py
```
