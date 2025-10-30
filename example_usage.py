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
output_rpm = 250      # Desired output shaft speed in RPM
power_hp = 10         # Power in horsepower

print(f"Input Conditions:")
print(f"  - Input RPM:  {input_rpm}")
print(f"  - Output RPM: {output_rpm}")
print(f"  - Power:      {power_hp} HP")
print()

# Use the trained model to predict gearbox parameters
try:
    predicted_params = tg.predict_gearbox_parameters(input_rpm, output_rpm, power_hp)
    
    print("Predicted Gearbox Parameters:")
    print("-" * 80)
    print(f"  Stage 1 Gear Ratio (n1):        {predicted_params['n1']:.2f}")
    print(f"  Stage 1 Diametral Pitch (Pdn1): {predicted_params['Pdn1']} teeth/inch")
    print(f"  Stage 1 Pinion Teeth (Np1):     {predicted_params['Np1']} teeth")
    print(f"  Stage 1 Helix Angle (Helix1):   {predicted_params['Helix1']}°")
    print(f"  Stage 2 Diametral Pitch (Pdn2): {predicted_params['Pdn2']} teeth/inch")
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
Pdn1 = predicted_params['Pdn1']
Np1 = predicted_params['Np1']
Helix1 = predicted_params['Helix1']

try:
    # Calculate stresses for Stage 1
    bending_stress_1 = calc.bending_stress(input_rpm, n1, Pdn1, Np1, Helix1)
    contact_stress_1 = calc.contact_stress(input_rpm, n1, Pdn1, Np1, Helix1)
    
    print("Stage 1 Stress Analysis:")
    print("-" * 80)
    print(f"  Bending Stress:  {bending_stress_1:.2f} ksi")
    print(f"  Allowable:       {tg.ALLOWABLE_BENDING_STRESS} ksi")
    
    # Calculate percentage difference from allowable
    bending_pct = ((bending_stress_1 - tg.ALLOWABLE_BENDING_STRESS) / tg.ALLOWABLE_BENDING_STRESS) * 100
    if bending_stress_1 < tg.ALLOWABLE_BENDING_STRESS:
        print(f"  Status:          ✓ SAFE ({abs(bending_pct):.1f}% below allowable)")
    else:
        print(f"  Status:          ✗ UNSAFE ({bending_pct:.1f}% above allowable)")
    
    print()
    print(f"  Contact Stress:  {contact_stress_1:.2f} ksi")
    print(f"  Allowable:       {tg.ALLOWABLE_CONTACT_STRESS} ksi")
    
    contact_pct = ((contact_stress_1 - tg.ALLOWABLE_CONTACT_STRESS) / tg.ALLOWABLE_CONTACT_STRESS) * 100
    if contact_stress_1 < tg.ALLOWABLE_CONTACT_STRESS:
        print(f"  Status:          ✓ SAFE ({abs(contact_pct):.1f}% below allowable)")
    else:
        print(f"  Status:          ✗ UNSAFE ({contact_pct:.1f}% above allowable)")
    print("-" * 80)
    
    # Calculate Stage 2 parameters
    P, Pd, wf, n, n2 = calc.important_values(input_rpm, n1, Pdn1, Np1, Helix1)
    
    # Stage 2 validation
    Pdn2 = predicted_params['Pdn2']
    Np2 = predicted_params['Np2']
    Helix2 = predicted_params['Helix2']
    
    bending_stress_2 = calc.bending_stress(wf, n2, Pdn2, Np2, Helix2)
    contact_stress_2 = calc.contact_stress(wf, n2, Pdn2, Np2, Helix2)
    
    print("\nStage 2 Stress Analysis:")
    print("-" * 80)
    print(f"  Intermediate Speed: {wf:.1f} RPM")
    print(f"  Stage 2 Ratio (n2): {n2:.2f}")
    print(f"  Overall Ratio (n):  {n:.2f}")
    print()
    print(f"  Bending Stress:  {bending_stress_2:.2f} ksi")
    print(f"  Allowable:       {tg.ALLOWABLE_BENDING_STRESS} ksi")
    
    bending_pct_2 = ((bending_stress_2 - tg.ALLOWABLE_BENDING_STRESS) / tg.ALLOWABLE_BENDING_STRESS) * 100
    if bending_stress_2 < tg.ALLOWABLE_BENDING_STRESS:
        print(f"  Status:          ✓ SAFE ({abs(bending_pct_2):.1f}% below allowable)")
    else:
        print(f"  Status:          ✗ UNSAFE ({bending_pct_2:.1f}% above allowable)")
    
    print()
    print(f"  Contact Stress:  {contact_stress_2:.2f} ksi")
    print(f"  Allowable:       {tg.ALLOWABLE_CONTACT_STRESS} ksi")
    
    contact_pct_2 = ((contact_stress_2 - tg.ALLOWABLE_CONTACT_STRESS) / tg.ALLOWABLE_CONTACT_STRESS) * 100
    if contact_stress_2 < tg.ALLOWABLE_CONTACT_STRESS:
        print(f"  Status:          ✓ SAFE ({abs(contact_pct_2):.1f}% below allowable)")
    else:
        print(f"  Status:          ✗ UNSAFE ({contact_pct_2:.1f}% above allowable)")
    print("-" * 80)
    
    # Overall validation
    print("\nOverall Design Validation:")
    print("-" * 80)
    all_safe = (bending_stress_1 < tg.ALLOWABLE_BENDING_STRESS and 
                contact_stress_1 < tg.ALLOWABLE_CONTACT_STRESS and
                bending_stress_2 < tg.ALLOWABLE_BENDING_STRESS and 
                contact_stress_2 < tg.ALLOWABLE_CONTACT_STRESS)
    
    if all_safe:
        print("  ✓✓✓ ALL STRESSES ARE WITHIN ALLOWABLE LIMITS ✓✓✓")
        print("  This is a VALID gearbox design!")
    else:
        print("  ✗✗✗ SOME STRESSES EXCEED ALLOWABLE LIMITS ✗✗✗")
        print("  This design may need modification.")
    print("-" * 80)
    
except Exception as e:
    print(f"\n❌ Error during stress calculation: {e}")
    print("This may happen if the predicted parameters are outside valid ranges.")

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
        params = tg.predict_gearbox_parameters(
            scenario['input_rpm'], 
            scenario['output_rpm'], 
            scenario['power_hp']
        )
        
        print(f"  Input:  {scenario['input_rpm']} RPM, {scenario['output_rpm']} RPM, {scenario['power_hp']} HP")
        print(f"  n1:     {params['n1']:.2f}")
        print(f"  Pdn1:   {params['Pdn1']}, Np1: {params['Np1']}, Helix1: {params['Helix1']}°")
        print(f"  Pdn2:   {params['Pdn2']}, Np2: {params['Np2']}, Helix2: {params['Helix2']}°")
        
        # Quick validation
        n1 = params['n1']
        bending = calc.bending_stress(scenario['input_rpm'], n1, params['Pdn1'], 
                                       params['Np1'], params['Helix1'])
        contact = calc.contact_stress(scenario['input_rpm'], n1, params['Pdn1'], 
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
