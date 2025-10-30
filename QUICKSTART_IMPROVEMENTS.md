# Quick Start: Improved Neural Network Model

## What Changed?

Your professor's simplified linear relationship (`wf = wp/12 + 100`) is **preserved**.

The improvements are **architectural**:
1. ✅ **Deeper network:** 4 layers → 5 layers  
2. ✅ **Wider network:** Max 256 → Max 512 neurons
3. ✅ **Batch normalization:** Added to stabilize training
4. ✅ **Feature engineering:** Added `total_ratio` as 4th input
5. ✅ **Better hyperparameters:** Lower LR, higher dropout, more epochs
6. ✅ **More training data:** 5K → 10K samples

**Result:** Same simplified physics model, but neural network learns it **much better**.

---

## Steps to Get Better Predictions

### 1. Delete Old Model (Required!)
```powershell
Remove-Item gearbox_nn_model.keras, gearbox_scaler.npz, training_data.csv -ErrorAction SilentlyContinue
```

**Why?** Model architecture changed (3→4 inputs, 4→5 layers). Old weights won't work.

### 2. Retrain
```powershell
python train_gearbox_nn.py
```

**Time:** 10-30 minutes  
**Watch for:** Stress validation percentage increasing over epochs (should reach >70%)

### 3. Test
```powershell
python train_gearbox_nn.py --predict
```

**Try:**
- Input: 1800 RPM, Output: 250 RPM, Power: 10 HP
- Input: 2400 RPM, Output: 300 RPM, Power: 15 HP

**Verify:** Helix = exactly 15, 20, or 25; Pdn = exactly 4, 5, 6, 8, or 10

---

## What to Expect

### During Training:
- ✅ Validation loss decreases steadily
- ✅ Stress validation % increases (target: >70-80%)
- ✅ MAE decreases for all 7 parameters
- ⏱️ Takes 10-30 min (depends on CPU/GPU)

### After Training:
- ✅ Much better predictions
- ✅ All discrete values properly constrained
- ✅ Consistent results for similar inputs

---

## Key Files

| File | Purpose |
|------|---------|
| `train_gearbox_nn.py` | Main training script (IMPROVED) |
| `CLASS_PROJECT_IMPROVEMENTS.md` | Detailed explanation of changes |
| `gearbox_nn_model.keras` | Trained model (generated) |
| `gearbox_scaler.npz` | Data scalers (generated) |
| `training_data.csv` | Training dataset (generated) |
| `training_history.png` | Loss curves (generated) |

---

## Still Getting Poor Predictions?

### Check These:

1. **Training data looks reasonable?**
   - Open `training_data.csv`
   - Verify Total_Ratio column makes sense
   - Check Output_RPM follows the linear formula

2. **Model converging?**
   - Open `training_history.png`
   - Loss should decrease, not plateau early
   - Training and validation loss shouldn't diverge too much

3. **Stress validation percentage?**
   - Should reach >70% by end of training
   - If <50%, may need more samples or different architecture

4. **Try adjusting:**
   - Increase samples to 20,000
   - Try learning rate 0.0003 (slower) or 0.001 (faster)
   - Adjust dropout between 0.2-0.4

---

## Summary

The improvements focus on making the neural network **learn better**, not changing the physics model. Your professor's simplified relationships are preserved. The deeper, wider architecture with batch normalization and feature engineering should give you **significantly better predictions** for your class project.

**Expected improvement:** From poor/inconsistent predictions → Accurate, reliable, constrained outputs.
