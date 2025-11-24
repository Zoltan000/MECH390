"""
Example Usage of the Trained Gearbox Neural Network Model

This script demonstrates how to use the trained neural network to predict
gearbox parameters and validate them using stress calculations.

Before running this script, you must first train the model by running:
    python train_gearbox_nn.py
"""

# Import necessary modules
import train_gearbox_nn as tg
import calculations as calc
import functions as fn

# =============================================================================
# EXAMPLE 1: Basic Prediction
# =============================================================================

print("="*80)
print("EXAMPLE 1: Basic Gearbox Parameter Prediction")
print("="*80 + "\n")

# Define the operating conditions you want to design for
input_rpm = 2000      # Input shaft speed in RPM

# Derive output RPM and power from input_rpm (same linear relations used in training)
output_rpm = input_rpm / 12 + 100
power_hp = input_rpm / 240

print(f"Input Conditions:")
print(f"  - Input RPM:  {input_rpm}")
print(f"  - Output RPM: {output_rpm}")
print(f"  - Power:      {power_hp} HP")
print()

# Use the trained model to predict gearbox parameters
try:
    predicted_params = tg.predict_gearbox_parameters(input_rpm)
    
    print("Predicted Gearbox Parameters:")
    print("-" * 80)
    print(f"  Stage 1 Gear Ratio (n1):        {predicted_params['n1']:.2f}")
    print(f"  Stage 1 Diametral Pitch (Pd1): {predicted_params['Pd1']} teeth/inch")
    print(f"  Stage 1 Pinion Teeth (Np1):     {predicted_params['Np1']} teeth")
    print(f"  Stage 1 Helix Angle (Helix1):   {predicted_params['Helix1']}°")
    print(f"  Stage 2 Diametral Pitch (Pd2): {predicted_params['Pd2']} teeth/inch")
    print(f"  Stage 2 Pinion Teeth (Np2):     {predicted_params['Np2']} teeth")
    print(f"  Stage 2 Helix Angle (Helix2):   {predicted_params['Helix2']}°")
    print("-" * 80)
    
except FileNotFoundError:
    print("\n❌ Error: Model files not found!")
    print("Please train the model first by running:")
    print("    python train_gearbox_nn.py\n")
    exit(1)

# =============================================================================
# EXAMPLE 2: Validate Predictions with Stress Calculations
# =============================================================================

print("\n" + "="*80)
print("EXAMPLE 2: Validating Predictions with Stress Calculations")
print("="*80 + "\n")

# Extract the predicted parameters for Stage 1
n1 = predicted_params['n1']
Pd1 = predicted_params['Pd1']
Np1 = predicted_params['Np1']
Helix1 = predicted_params['Helix1']

try:
    # Use validation info provided by predict_gearbox_parameters (if available)
    sigma_b1 = predicted_params.get('sigma_b1')
    sigma_c1 = predicted_params.get('sigma_c1')
    sigma_b2 = predicted_params.get('sigma_b2')
    sigma_c2 = predicted_params.get('sigma_c2')
    wf = predicted_params.get('wf')
    n2 = predicted_params.get('n2')

    print("Stage 1 Stress Analysis:")
    print("-" * 80)
    print(f"  Bending Stress:  {sigma_b1:.2f} ksi")
    print(f"  Allowable:       {tg.ALLOWABLE_BENDING_STRESS} ksi")
    print(f"  Contact Stress:  {sigma_c1:.2f} ksi")
    print(f"  Allowable:       {tg.ALLOWABLE_CONTACT_STRESS} ksi")
    print("-" * 80)

    print("Stage 2 Stress Analysis:")
    print("-" * 80)
    print(f"  Intermediate Speed: {wf:.1f} RPM")
    print(f"  Stage 2 Ratio (n2): {n2:.2f}")
    print(f"  Bending Stress:  {sigma_b2:.2f} ksi")
    print(f"  Contact Stress:  {sigma_c2:.2f} ksi")
    print("-" * 80)

    # Overall validation
    print("\nOverall Design Validation:")
    print("-" * 80)
    if predicted_params.get('valid'):
        print("  ✓✓✓ ALL STRESSES ARE WITHIN ALLOWABLE LIMITS ✓✓✓")
        print("  This is a VALID gearbox design!")
    else:
        print("  ✗✗✗ SOME STRESSES EXCEED ALLOWABLE LIMITS ✗✗✗")
        print("  This design may need modification.")
    print("-" * 80)

except Exception as e:
    print(f"\n❌ Error during stress display: {e}")
    print("This may happen if the predicted parameters are missing validation info.")

# =============================================================================
# EXAMPLE 3: Multiple Predictions
# =============================================================================

print("\n" + "="*80)
print("EXAMPLE 3: Comparing Multiple Design Scenarios")
print("="*80 + "\n")

# Define multiple scenarios to test
scenarios = [
    {"name": "Low Speed, Low Power", "input_rpm": 1500, "output_rpm": 200, "power_hp": 5},
    {"name": "Medium Speed, Medium Power", "input_rpm": 2000, "output_rpm": 250, "power_hp": 10},
    {"name": "High Speed, High Power", "input_rpm": 3000, "output_rpm": 300, "power_hp": 15},
]

print("Comparing different operating conditions:\n")

for i, scenario in enumerate(scenarios, 1):
    print(f"Scenario {i}: {scenario['name']}")
    print("-" * 80)
    
    try:
        params = tg.predict_gearbox_parameters(scenario['input_rpm'])

        derived_wf = scenario['input_rpm'] / 12 + 100
        derived_P = scenario['input_rpm'] / 240
        print(f"  Input:  {scenario['input_rpm']} RPM, {derived_wf:.1f} RPM, {derived_P:.1f} HP")
        print(f"  n1:     {params['n1']:.2f}")
        print(f"  Pd1:   {params['Pd1']}, Np1: {params['Np1']}, Helix1: {params['Helix1']}°")
        print(f"  Pd2:   {params['Pd2']}, Np2: {params['Np2']}, Helix2: {params['Helix2']}°")
        
        # Quick validation
        n1 = params['n1']
        bending = calc.bending_stress(scenario['input_rpm'], n1, params['Pd1'], 
                           params['Np1'], params['Helix1'])
        contact = calc.contact_stress(scenario['input_rpm'], n1, params['Pd1'], 
                          params['Np1'], params['Helix1'])
        
        safe = (bending < tg.ALLOWABLE_BENDING_STRESS and 
                contact < tg.ALLOWABLE_CONTACT_STRESS)
        print(f"  Status: {'✓ Valid design' if safe else '✗ May need revision'}")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    print()

# =============================================================================
# SUMMARY
# =============================================================================

print("="*80)
print("SUMMARY")
print("="*80)
print("""
This example script demonstrated:

1. How to use the trained model to predict gearbox parameters
2. How to validate predictions using stress calculations
3. How to compare multiple design scenarios

Key Points:
- The model predicts 7 parameters for a two-stage gearbox
- Predictions should be validated using stress calculations
- The model aims to predict designs within allowable stress limits
- You can use this for rapid design iteration and optimization

Next Steps:
- Try different input conditions in your own code
- Integrate the model into your design workflow
- Retrain with more data for improved accuracy
- Modify the model architecture for specific needs

For more information, see README_NN_TRAINING.md
""")
print("="*80)
