@echo off
rem run_pipeline.bat — one-shot wrapper for scripts\run_pipeline.py
rem
rem Usage:
rem   run_pipeline.bat my_voice
rem   run_pipeline.bat my_voice --epochs 300 --sr 48k
rem
rem Override defaults via env vars before calling, e.g.:
rem   set EPOCHS=300 && run_pipeline.bat my_voice

setlocal enabledelayedexpansion

rem --- Defaults (override with env vars) ---
if "%MODEL_NAME%"=="" set MODEL_NAME=my_voice
if "%SR%"==""         set SR=40k
if "%EPOCHS%"==""     set EPOCHS=50
if "%SAVE_EPOCH%"=="" set SAVE_EPOCH=10
if "%BATCH_SIZE%"=="" set BATCH_SIZE=8
if "%GPUS%"==""       set GPUS=0
if "%F0METHOD%"==""   set F0METHOD=rmvpe
if "%N_PROCESSES%"="" set N_PROCESSES=4

rem --- First positional arg overrides MODEL_NAME ---
if not "%~1"=="" (
    echo %~1 | findstr /b /c:"--" >nul || (
        set MODEL_NAME=%~1
        shift
    )
)

rem --- Forward remaining args verbatim ---
set EXTRA_ARGS=
:collect_args
if "%~1"=="" goto run
set EXTRA_ARGS=%EXTRA_ARGS% %1
shift
goto collect_args

:run
set SCRIPT_DIR=%~dp0

python "%SCRIPT_DIR%scripts\run_pipeline.py" ^
    --model_name  "%MODEL_NAME%" ^
    --sr          "%SR%" ^
    --epochs      %EPOCHS% ^
    --save_epoch  %SAVE_EPOCH% ^
    --batch_size  %BATCH_SIZE% ^
    --gpus        "%GPUS%" ^
    --f0method    "%F0METHOD%" ^
    --n_processes %N_PROCESSES% ^
    %EXTRA_ARGS%

if errorlevel 1 (
    echo.
    echo Pipeline failed. Check output above.
    exit /b 1
)
