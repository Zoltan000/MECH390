@echo off
setlocal EnableDelayedExpansion

REM ============================
REM CONFIGURATION
REM ============================
REM Number of workers you want
set "NUM_WORKERS=24"

REM Global wp range you want to cover
set "WP_MIN=1200"
set "WP_MAX=3600"

REM Python + script name
set "PYTHON=python"
set "SCRIPT=data_generator.py"

REM Any flags common to all workers (optional, leave empty if not needed)
REM e.g. set "COMMON_FLAGS=--estimate"
set "COMMON_FLAGS="

REM ============================
REM CALCULATE RANGE PER WORKER
REM ============================
set /A RANGE=WP_MAX - WP_MIN
set /A CHUNK=RANGE / NUM_WORKERS

echo Total wp range: %WP_MIN% to %WP_MAX%
echo Num workers: %NUM_WORKERS%
echo Chunk size (approx): %CHUNK%
echo.

REM ============================
REM LAUNCH WORKERS
REM ============================
for /L %%i in (0,1,%NUM_WORKERS%-1) do (
    REM compute start for this worker
    set /A START=WP_MIN + CHUNK*%%i

    REM last worker takes us exactly up to WP_MAX
    if %%i EQU %NUM_WORKERS%-1 (
        set "STOP=%WP_MAX%"
    ) else (
        set /A STOP=START + CHUNK
    )

    set "OUT_CSV=wp_!START!-!STOP!.csv"

    echo Launching worker %%i: wp from !START! to !STOP!, output !OUT_CSV!

    REM Each worker runs in its own PowerShell window
    start "worker %%i" powershell -NoExit -Command ^
        "%PYTHON% %SCRIPT% --wp-start !START! --wp-stop !STOP! --out-csv !OUT_CSV! %COMMON_FLAGS%"
)

echo All workers launched.
endlocal
