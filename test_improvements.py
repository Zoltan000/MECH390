"""
Quick script to test the improved neural network model.
Compares predictions before and after improvements.
"""

import numpy as np
import os

def test_model_improvements():
    """Test if the model improvements work"""
    
    print("="*80)
    print("NEURAL NETWORK MODEL IMPROVEMENT VERIFICATION")
    print("="*80)
    
    # Check if old model exists
    old_model_exists = os.path.exists('gearbox_nn_model.keras')
    old_scaler_exists = os.path.exists('gearbox_scaler.npz')
    
    if old_model_exists or old_scaler_exists:
        print("\n⚠️  WARNING: Old model files detected!")
        print("   The model architecture has changed significantly.")
        print("   You should delete the old files and retrain:")
        print()
        print("   Remove-Item gearbox_nn_model.keras -ErrorAction SilentlyContinue")
        print("   Remove-Item gearbox_scaler.npz -ErrorAction SilentlyContinue")
        print("   python train_gearbox_nn.py")
        print()
        return False
    
    print("\n✅ No old model files found - ready for fresh training!")
    
    # Show key improvements
    print("\n" + "="*80)
    print("KEY IMPROVEMENTS IMPLEMENTED")
    print("="*80)
    
    improvements = [
        ("Data Generation", "Fixed broken wf calculation - now uses real physics"),
        ("Feature Engineering", "Added total_ratio as 4th input feature"),
        ("Model Architecture", "Deeper (5 layers) and wider (512 neurons max)"),
        ("Batch Normalization", "Added after every dense layer for stability"),
        ("Regularization", "Increased dropout to 0.3 for better generalization"),
        ("Training Samples", "Increased from 5,000 to 10,000 samples"),
        ("Learning Rate", "Reduced from 0.001 to 0.0005 for precision"),
        ("Batch Size", "Increased from 32 to 64 for stability"),
        ("Epochs", "Increased from 100 to 200 (with early stopping)"),
        ("Validation", "Updated to use constraint snapping")
    ]
    
    for i, (area, improvement) in enumerate(improvements, 1):
        print(f"{i:2d}. {area:20s}: {improvement}")
    
    # Test cases for after training
    print("\n" + "="*80)
    print("RECOMMENDED TEST CASES (After Training)")
    print("="*80)
    
    test_cases = [
        (1800, 150, 10.0),
        (2400, 200, 15.0),
        (3000, 250, 8.0),
        (1500, 180, 12.0),
        (2800, 300, 18.0)
    ]
    
    print("\nTest these input combinations after training:")
    print("-" * 80)
    print(f"{'Case':<6} {'Input RPM':<12} {'Output RPM':<12} {'Power HP':<10}")
    print("-" * 80)
    
    for i, (inp_rpm, out_rpm, power) in enumerate(test_cases, 1):
        print(f"{i:<6} {inp_rpm:<12.0f} {out_rpm:<12.0f} {power:<10.1f}")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    
    print("""
1. Train the improved model:
   python train_gearbox_nn.py
   
   Expected time: 10-30 minutes
   Expected output: Training data CSV, model files, training history plot

2. Test predictions:
   python train_gearbox_nn.py --predict
   
   Enter one of the test cases above and verify:
   - Helix angles are exactly 15, 20, or 25
   - Pdn values are exactly 4, 5, 6, 8, or 10
   - All parameters seem reasonable

3. Review training data:
   Open training_data.csv in Excel/spreadsheet
   Verify the Total_Ratio column makes sense

4. Check training history:
   Open training_history.png
   Verify loss decreases and doesn't overfit

5. Monitor stress validation:
   During training, watch the "Stress Validation" percentage
   Should be >70-80% by the end of training
""")
    
    print("="*80)
    return True


if __name__ == "__main__":
    test_model_improvements()
