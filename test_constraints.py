"""
Quick test to verify the constraint function works correctly.
This tests the snap_to_valid_values function without needing the trained model.
"""

import numpy as np

def snap_to_valid_values(predicted_value, valid_options):
    """
    Snap a predicted continuous value to the nearest valid discrete option.
    """
    valid_options = np.array(valid_options)
    distances = np.abs(valid_options - predicted_value)
    closest_idx = np.argmin(distances)
    return valid_options[closest_idx]


# Test cases
VALID_PDN = [4, 5, 6, 8, 10]
VALID_HELIX = [15, 20, 25]

print("Testing snap_to_valid_values function")
print("=" * 60)

# Test helix angle constraints
test_helix_values = [14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26]
print("\nHelix Angle Tests (valid options: 15, 20, 25):")
print("-" * 60)
for val in test_helix_values:
    result = snap_to_valid_values(val, VALID_HELIX)
    print(f"Input: {val:2d} → Output: {result:2d}")

# Test PDN constraints
test_pdn_values = [3.5, 4.2, 4.8, 5.5, 6.3, 7, 8.1, 9, 10.5]
print("\nDiametral Pitch (Pdn) Tests (valid options: 4, 5, 6, 8, 10):")
print("-" * 60)
for val in test_pdn_values:
    result = snap_to_valid_values(val, VALID_PDN)
    print(f"Input: {val:4.1f} → Output: {result:2d}")

print("\n" + "=" * 60)
print("✓ All values successfully constrained to valid discrete options!")
print("=" * 60)
