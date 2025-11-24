# run_train.ps1 — Activate the venv and run the training/prediction script
# Usage: .\run_train.ps1 [--predict]

$venvActivate = Join-Path $PSScriptRoot ".venv_py310\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    Write-Error "Virtual environment not found at $venvActivate. Create it with: py -3.10 -m venv .venv_py310"
    exit 1
}

# Activate the venv in this script's session
& $venvActivate

# Run the Python script using the activated venv's python
# Forward any arguments provided to this script to the python script
python .\train_gearbox_nn.py @args
