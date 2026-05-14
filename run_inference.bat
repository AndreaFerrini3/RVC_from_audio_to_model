@echo off
rem run_inference.bat — one-shot wrapper for scripts\run_inference.py
rem
rem Usage:
rem   run_inference.bat my_voice                       (reads inference_dataset\inf_dataset_1\)
rem   run_inference.bat my_voice inf_dataset_2         (reads inference_dataset\inf_dataset_2\)
rem   run_inference.bat my_voice inf_dataset_1 --transpose 2
rem   run_inference.bat my_voice --input path\to\file.wav

setlocal enabledelayedexpansion

rem --- Defaults (override with env vars) ---
if "%MODEL_NAME%"==""        set MODEL_NAME=my_voice
if "%INFERENCE_DATASET%"=="" set INFERENCE_DATASET=inf_dataset_1
if "%TRANSPOSE%"=="" set TRANSPOSE=0
if "%F0METHOD%"=""  set F0METHOD=rmvpe
if "%INDEX_RATE%"="" set INDEX_RATE=0.66
if "%DEVICE%"=""    set DEVICE=cuda:0
if "%IS_HALF%"=""   set IS_HALF=true

rem --- First positional: MODEL_NAME (unless starts with --) ---
if not "%~1"=="" (
    echo %~1 | findstr /b /c:"--" >nul || (
        set MODEL_NAME=%~1
        shift
    )
)

rem --- Second positional: INFERENCE_DATASET (unless starts with --) ---
if not "%~1"=="" (
    echo %~1 | findstr /b /c:"--" >nul || (
        set INFERENCE_DATASET=%~1
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

python "%SCRIPT_DIR%scripts\run_inference.py" ^
    --model_name        "%MODEL_NAME%" ^
    --inference_dataset "%INFERENCE_DATASET%" ^
    --transpose         %TRANSPOSE% ^
    --f0method          %F0METHOD% ^
    --index_rate        %INDEX_RATE% ^
    --device            %DEVICE% ^
    --is_half           %IS_HALF% ^
    %EXTRA_ARGS%

if errorlevel 1 (
    echo.
    echo Inference failed. Check output above.
    exit /b 1
)
