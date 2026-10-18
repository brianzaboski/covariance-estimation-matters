@echo off
REM Run every analysis script in order, then the two checks.
REM
REM   run_all.bat              full run, needs the registry data in data\
REM   run_all.bat synthetic    Study 1 and the three Study 2 scripts against the
REM                            synthetic file; 01, 02 and 02b are skipped, since
REM                            they need the item-level export
REM
REM Stops at the first failure rather than carrying on with missing inputs.

setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if defined CONDA_DEFAULT_ENV (
    echo Conda environment: %CONDA_DEFAULT_ENV%
    if /i "%CONDA_DEFAULT_ENV%"=="base" (
        echo.
        echo WARNING: this is the base environment. Create a dedicated one:
        echo    conda create -n covest python=3.11.8 -y
        echo    conda activate covest
        echo    pip install -r env\requirements.txt
        echo.
    )
) else (
    echo No virtual environment detected. Using whatever python is on PATH.
    echo Create one with either:
    echo    conda create -n covest python=3.11.8 -y ^&^& conda activate covest
    echo    python -m venv .venv ^&^& .venv\Scripts\activate
    echo then:  pip install -r env\requirements.txt
    echo.
)

REM conda environments ship "python" but not the "py" launcher, so prefer python
where python >nul 2>nul && (set PY=python) || (set PY=py)

REM Show which interpreter is actually being used, so a wrong environment is
REM visible before the analysis scripts run against it.
%PY% -c "import sys; print('python', sys.version.split()[0]); print('from   ', sys.executable)"

%PY% -c "import sklearn, sys; v = sklearn.__version__; print('scikit-learn', v); sys.exit(0 if v == '1.8.0' else 1)"
if errorlevel 1 (
    echo.
    echo WARNING: scikit-learn is not 1.8.0. MinCovDet changed across releases,
    echo so every MCD number will differ from the manuscript. Continuing anyway.
    echo.
)

set DATA=
if /i "%~1"=="synthetic" (
    set DATA=--data data\synthetic\synthetic_clinical_data.csv
    echo Running against the synthetic dataset. Numbers will not match the paper.
    echo.
    goto :study1
)

REM `exit /b` inside :run ends the subroutine, not this script, so every call
REM has to test the errorlevel itself. Later scripts read what earlier ones
REM write, and stale files in outputs\ will hide a failure otherwise.
call :run 01_clean_and_score.py
if errorlevel 1 goto :aborted
call :run 02_descriptives_octcdq_bfas.py
if errorlevel 1 goto :aborted
call :run 02b_feature_reliability.py
if errorlevel 1 goto :aborted

:study1
call :run 03_phase2_analyses.py %DATA%
if errorlevel 1 goto :aborted
call :run 04_invert_r.py
if errorlevel 1 goto :aborted
call :run 05_study1_simulation.py
if errorlevel 1 goto :aborted
call :run 05b_figure1.py
if errorlevel 1 goto :aborted
call :run 05c_calibration_ranges.py
if errorlevel 1 goto :aborted
call :run 06_bootstrap_sample_size.py %DATA%
if errorlevel 1 goto :aborted
call :run 07_support_fraction_sweep.py %DATA%
if errorlevel 1 goto :aborted
call :run 08_fig_mcd_discarded.py
if errorlevel 1 goto :aborted

echo.
echo === Checks ===
%PY% code\check_reproduction.py
echo.
%PY% code\check_outputs.py --quiet
echo.
echo Done. Read the two check reports above.
goto :eof

:aborted
echo.
echo Stopped. Nothing after the failing script was run, so anything now in
echo outputs\ from a later step is left over from an earlier run.
exit /b 1

:run
echo.
echo === %* ===
%PY% code\%*
if errorlevel 1 (
    echo.
    echo FAILED: %1 stopped with an error. Fix it before continuing;
    echo later scripts read what the earlier ones write.
    exit /b 1
)
goto :eof
