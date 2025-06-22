@echo off
setlocal enabledelayedexpansion

REM =============================================================================
REM Gender Classification System - Task A Complete Runner (Windows)
REM =============================================================================
REM This batch file provides a comprehensive way to run the complete Task A system
REM with automatic environment setup, dependency management, and GPU optimization.
REM
REM Author: AI Assistant
REM Date: 2024
REM
REM Usage:
REM   run_task_a.bat                    REM Run with default settings
REM   run_task_a.bat --quick            REM Quick training (reduced epochs)
REM   run_task_a.bat --gpu-only         REM Skip if no GPU available
REM   run_task_a.bat --custom-config    REM Use custom configuration
REM =============================================================================

REM Configuration
set SCRIPT_DIR=%~dp0
set VENV_DIR=%SCRIPT_DIR%venv
set LOG_FILE=%SCRIPT_DIR%setup_and_training.log
set PYTHON_MIN_VERSION=3.8

REM Default parameters
set EPOCHS=50
set BATCH_SIZE=32
set LEARNING_RATE=1e-4
set OUTPUT_DIR=output
set SKIP_ENSEMBLE=false
set SKIP_OPTIMIZATION=false
set GPU_ONLY=false
set QUICK_MODE=false
set CUSTOM_CONFIG=false
set CONFIG_FILE=config_template.json

REM Colors (limited in Windows batch)
set "RED=[31m"
set "GREEN=[32m"
set "YELLOW=[33m"
set "BLUE=[34m"
set "PURPLE=[35m"
set "CYAN=[36m"
set "NC=[0m"

REM =============================================================================
REM Utility Functions
REM =============================================================================

:print_header
echo ===============================================================================
echo 🚀 Gender Classification System - Task A Complete Training
echo ===============================================================================
echo.
goto :eof

:print_section
echo 📋 %~1
echo %date% %time% - %~1 >> "%LOG_FILE%"
goto :eof

:print_success
echo ✅ %~1
echo %date% %time% - SUCCESS: %~1 >> "%LOG_FILE%"
goto :eof

:print_warning
echo ⚠️  %~1
echo %date% %time% - WARNING: %~1 >> "%LOG_FILE%"
goto :eof

:print_error
echo ❌ %~1
echo %date% %time% - ERROR: %~1 >> "%LOG_FILE%"
goto :eof

:print_info
echo ℹ️  %~1
echo %date% %time% - INFO: %~1 >> "%LOG_FILE%"
goto :eof

REM =============================================================================
REM Argument Parsing
REM =============================================================================

:parse_arguments
if "%~1"=="" goto :done_parsing

if "%~1"=="--epochs" (
    set EPOCHS=%~2
    shift & shift
    goto :parse_arguments
)

if "%~1"=="--batch-size" (
    set BATCH_SIZE=%~2
    shift & shift
    goto :parse_arguments
)

if "%~1"=="--lr" (
    set LEARNING_RATE=%~2
    shift & shift
    goto :parse_arguments
)

if "%~1"=="--learning-rate" (
    set LEARNING_RATE=%~2
    shift & shift
    goto :parse_arguments
)

if "%~1"=="--output-dir" (
    set OUTPUT_DIR=%~2
    shift & shift
    goto :parse_arguments
)

if "%~1"=="--skip-ensemble" (
    set SKIP_ENSEMBLE=true
    shift
    goto :parse_arguments
)

if "%~1"=="--skip-optimization" (
    set SKIP_OPTIMIZATION=true
    shift
    goto :parse_arguments
)

if "%~1"=="--gpu-only" (
    set GPU_ONLY=true
    shift
    goto :parse_arguments
)

if "%~1"=="--quick" (
    set QUICK_MODE=true
    set EPOCHS=20
    set BATCH_SIZE=64
    shift
    goto :parse_arguments
)

if "%~1"=="--custom-config" (
    set CUSTOM_CONFIG=true
    set CONFIG_FILE=%~2
    shift & shift
    goto :parse_arguments
)

if "%~1"=="--config" (
    set CONFIG_FILE=%~2
    shift & shift
    goto :parse_arguments
)

if "%~1"=="--help" goto :show_help
if "%~1"=="-h" goto :show_help

call :print_error "Unknown option: %~1"
goto :show_help

:done_parsing
goto :eof

:show_help
echo Gender Classification System - Task A Runner
echo.
echo Usage: %~nx0 [OPTIONS]
echo.
echo Options:
echo   --epochs N              Number of training epochs (default: 50)
echo   --batch-size N          Batch size for training (default: 32)
echo   --lr, --learning-rate   Learning rate (default: 1e-4)
echo   --output-dir DIR        Output directory (default: output)
echo   --skip-ensemble         Skip ensemble training
echo   --skip-optimization     Skip model optimization
echo   --gpu-only              Exit if no GPU available
echo   --quick                 Quick mode (20 epochs, batch size 64)
echo   --custom-config FILE    Use custom configuration file
echo   --config FILE           Configuration file to use
echo   --help, -h              Show this help message
echo.
echo Examples:
echo   %~nx0                                    REM Default training
echo   %~nx0 --quick                           REM Quick training
echo   %~nx0 --epochs 100 --batch-size 64     REM Custom parameters
echo   %~nx0 --gpu-only --skip-ensemble       REM GPU-only, no ensemble
exit /b 0

REM =============================================================================
REM System Checks
REM =============================================================================

:check_python_version
call :print_section "Checking Python Version"

python --version >nul 2>&1
if errorlevel 1 (
    call :print_error "Python is not installed or not in PATH"
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
call :print_info "Found Python %PYTHON_VERSION%"

REM Simple version check (assuming 3.x.x format)
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

if %MAJOR% LSS 3 (
    call :print_error "Python 3 is required. Found: %PYTHON_VERSION%"
    exit /b 1
)

if %MAJOR% EQU 3 if %MINOR% LSS 8 (
    call :print_error "Python 3.8 or higher is required. Found: %PYTHON_VERSION%"
    exit /b 1
)

call :print_success "Python version is compatible"
goto :eof

:check_gpu_availability
call :print_section "Checking GPU Availability"

nvidia-smi >nul 2>&1
if errorlevel 1 (
    call :print_warning "No NVIDIA GPU detected or nvidia-smi not available"
    if "%GPU_ONLY%"=="true" (
        call :print_error "GPU-only mode requested but no GPU available"
        exit /b 1
    )
    call :print_info "Training will proceed on CPU (much slower)"
    goto :eof
)

call :print_success "NVIDIA GPU detected"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>nul
goto :eof

:check_disk_space
call :print_section "Checking Disk Space"

for /f "tokens=3" %%a in ('dir /-c "%SCRIPT_DIR%" ^| find "bytes free"') do set FREE_BYTES=%%a
set /a FREE_GB=!FREE_BYTES!/1024/1024/1024

call :print_info "Available disk space: %FREE_GB%GB"

if %FREE_GB% LSS 5 (
    call :print_warning "Low disk space detected. Training may fail if space runs out."
) else (
    call :print_success "Sufficient disk space available"
)
goto :eof

:check_memory
call :print_section "Checking System Memory"

for /f "skip=1" %%p in ('wmic computersystem get TotalPhysicalMemory') do (
    set TOTAL_MEM=%%p
    goto :memory_done
)
:memory_done

set /a TOTAL_MEM_GB=!TOTAL_MEM!/1024/1024/1024
call :print_info "Total system memory: %TOTAL_MEM_GB%GB"

if %TOTAL_MEM_GB% LSS 8 (
    call :print_warning "Low system memory. Consider reducing batch size."
    set BATCH_SIZE=16
    call :print_info "Automatically reduced batch size to %BATCH_SIZE%"
) else (
    call :print_success "Sufficient system memory available"
)
goto :eof

REM =============================================================================
REM Environment Setup Functions
REM =============================================================================

:setup_virtual_environment
call :print_section "Setting Up Virtual Environment"

if exist "%VENV_DIR%" (
    call :print_info "Virtual environment already exists"
) else (
    call :print_info "Creating virtual environment..."
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        call :print_error "Failed to create virtual environment"
        exit /b 1
    )
    call :print_success "Virtual environment created"
)

REM Activate virtual environment
call "%VENV_DIR%\Scripts\activate.bat"
call :print_success "Virtual environment activated"

REM Upgrade pip
call :print_info "Upgrading pip..."
python -m pip install --upgrade pip >nul 2>&1
call :print_success "Pip upgraded"
goto :eof

:install_pytorch
call :print_section "Installing PyTorch"

REM Check if PyTorch is already installed
python -c "import torch; print('PyTorch version:', torch.__version__)" >nul 2>&1
if not errorlevel 1 (
    call :print_info "PyTorch already installed"
    python -c "import torch; print('  Version:', torch.__version__)"
    python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
    if not errorlevel 1 (
        call :print_success "PyTorch with CUDA support detected"
        goto :eof
    ) else (
        call :print_warning "PyTorch without CUDA detected, reinstalling with CUDA support"
    )
)

call :print_info "Installing PyTorch with CUDA support..."

REM Try to detect CUDA version
nvcc --version >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%i in ('nvcc --version ^| findstr "release"') do set CUDA_LINE=%%i
    for /f "tokens=6 delims=, " %%a in ("!CUDA_LINE!") do set CUDA_VERSION=%%a
    call :print_info "Detected CUDA version: !CUDA_VERSION!"

    if "!CUDA_VERSION:~0,3!"=="12." (
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ) else if "!CUDA_VERSION:~0,3!"=="11." (
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ) else (
        call :print_warning "Unsupported CUDA version, installing CPU version"
        pip install torch torchvision torchaudio
    )
) else (
    call :print_info "CUDA not detected, installing CPU version"
    pip install torch torchvision torchaudio
)

call :print_success "PyTorch installation completed"
goto :eof

:install_dependencies
call :print_section "Installing Dependencies"

if not exist "%SCRIPT_DIR%requirements.txt" (
    call :print_error "requirements.txt not found"
    exit /b 1
)

call :print_info "Installing packages from requirements.txt..."
pip install -r "%SCRIPT_DIR%requirements.txt"
if errorlevel 1 (
    call :print_error "Failed to install dependencies"
    exit /b 1
)
call :print_success "Dependencies installed successfully"
goto :eof

:verify_installation
call :print_section "Verifying Installation"

python -c "import torch, torchvision, timm, transformers, cv2, numpy, matplotlib.pyplot, pandas, sklearn; print('✅ All core packages imported successfully'); print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'CPU only'); print(f'GPU name: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '')"
if errorlevel 1 (
    call :print_error "Package verification failed"
    exit /b 1
)

call :print_success "Installation verification completed"
goto :eof

REM =============================================================================
REM Training Functions
REM =============================================================================

:prepare_training_environment
call :print_section "Preparing Training Environment"

REM Create output directories
if not exist "%SCRIPT_DIR%%OUTPUT_DIR%" mkdir "%SCRIPT_DIR%%OUTPUT_DIR%"
if not exist "%SCRIPT_DIR%%OUTPUT_DIR%\models" mkdir "%SCRIPT_DIR%%OUTPUT_DIR%\models"
if not exist "%SCRIPT_DIR%%OUTPUT_DIR%\plots" mkdir "%SCRIPT_DIR%%OUTPUT_DIR%\plots"
if not exist "%SCRIPT_DIR%%OUTPUT_DIR%\logs" mkdir "%SCRIPT_DIR%%OUTPUT_DIR%\logs"

REM Check data directories
if not exist "%SCRIPT_DIR%train" (
    call :print_error "Training data directory not found. Expected: train/"
    call :print_info "Please ensure the dataset is properly structured as described in README.md"
    exit /b 1
)

if not exist "%SCRIPT_DIR%val" (
    call :print_error "Validation data directory not found. Expected: val/"
    call :print_info "Please ensure the dataset is properly structured as described in README.md"
    exit /b 1
)

REM Count images (simplified)
for /f %%i in ('dir /b /s "%SCRIPT_DIR%train\female\*.jpg" "%SCRIPT_DIR%train\female\*.png" "%SCRIPT_DIR%train\female\*.jpeg" 2^>nul ^| find /c /v ""') do set TRAIN_FEMALE=%%i
for /f %%i in ('dir /b /s "%SCRIPT_DIR%train\male\*.jpg" "%SCRIPT_DIR%train\male\*.png" "%SCRIPT_DIR%train\male\*.jpeg" 2^>nul ^| find /c /v ""') do set TRAIN_MALE=%%i
for /f %%i in ('dir /b /s "%SCRIPT_DIR%val\female\*.jpg" "%SCRIPT_DIR%val\female\*.png" "%SCRIPT_DIR%val\female\*.jpeg" 2^>nul ^| find /c /v ""') do set VAL_FEMALE=%%i
for /f %%i in ('dir /b /s "%SCRIPT_DIR%val\male\*.jpg" "%SCRIPT_DIR%val\male\*.png" "%SCRIPT_DIR%val\male\*.jpeg" 2^>nul ^| find /c /v ""') do set VAL_MALE=%%i

call :print_info "Dataset statistics:"
call :print_info "  Training: %TRAIN_FEMALE% female, %TRAIN_MALE% male"
call :print_info "  Validation: %VAL_FEMALE% female, %VAL_MALE% male"

call :print_success "Training environment prepared"
goto :eof

:create_config_file
call :print_section "Creating Configuration File"

set CONFIG_PATH=%SCRIPT_DIR%runtime_config.json

if "%SKIP_ENSEMBLE%"=="true" (
    set ENSEMBLE_FLAG=false
) else (
    set ENSEMBLE_FLAG=true
)

if "%SKIP_OPTIMIZATION%"=="true" (
    set OPTIMIZATION_FLAG=false
) else (
    set OPTIMIZATION_FLAG=true
)

(
echo {
echo   "data": {
echo     "train_dir": "train",
echo     "val_dir": "val",
echo     "batch_size": %BATCH_SIZE%,
echo     "num_workers": 4,
echo     "image_size": 224
echo   },
echo   "training": {
echo     "num_epochs": %EPOCHS%,
echo     "learning_rate": %LEARNING_RATE%,
echo     "weight_decay": 1e-4,
echo     "dropout_rate": 0.3,
echo     "warmup_epochs": 5
echo   },
echo   "models": {
echo     "train_single": true,
echo     "train_ensemble": %ENSEMBLE_FLAG%,
echo     "apply_distillation": %OPTIMIZATION_FLAG%,
echo     "apply_quantization": %OPTIMIZATION_FLAG%
echo   },
echo   "output": {
echo     "output_dir": "%OUTPUT_DIR%",
echo     "save_plots": true,
echo     "save_logs": true
echo   }
echo }
) > "%CONFIG_PATH%"

call :print_success "Configuration file created: %CONFIG_PATH%"
goto :eof

:run_training
call :print_section "Starting Training Process"

REM Build training command
set TRAIN_CMD=python train_complete_system.py --epochs %EPOCHS% --batch-size %BATCH_SIZE% --lr %LEARNING_RATE% --output-dir %OUTPUT_DIR%

if "%SKIP_ENSEMBLE%"=="true" (
    set TRAIN_CMD=%TRAIN_CMD% --skip-ensemble
)

if "%SKIP_OPTIMIZATION%"=="true" (
    set TRAIN_CMD=%TRAIN_CMD% --skip-optimization
)

if "%CUSTOM_CONFIG%"=="true" (
    if exist "%SCRIPT_DIR%%CONFIG_FILE%" (
        set TRAIN_CMD=%TRAIN_CMD% --config %CONFIG_FILE%
    )
)

call :print_info "Training command: %TRAIN_CMD%"
call :print_info "Training started at: %date% %time%"

REM Run training
%TRAIN_CMD% 2>&1 | tee -a "%LOG_FILE%"
if errorlevel 1 (
    call :print_error "Training failed!"
    exit /b 1
)

call :print_success "Training completed successfully!"
goto :eof

:run_demo
call :print_section "Running Demonstration"

REM Check if models exist
if exist "%SCRIPT_DIR%%OUTPUT_DIR%\models\best_single_model.pth" (
    call :print_info "Running prediction demo with single model..."
    python demo_predictions.py --model "%OUTPUT_DIR%\models\best_single_model.pth" --demo-mode
) else if exist "%SCRIPT_DIR%%OUTPUT_DIR%\models\best_ensemble_model.pth" (
    call :print_info "Running prediction demo with ensemble model..."
    python demo_predictions.py --model "%OUTPUT_DIR%\models\best_ensemble_model.pth" --demo-mode
) else (
    call :print_warning "No trained models found for demo"
)
goto :eof

REM =============================================================================
REM Results and Cleanup Functions
REM =============================================================================

:show_results_summary
call :print_section "Training Results Summary"

if exist "%SCRIPT_DIR%%OUTPUT_DIR%\evaluation_results.json" (
    call :print_info "Loading evaluation results..."
    python -c "import json; results = json.load(open('%SCRIPT_DIR%%OUTPUT_DIR%/evaluation_results.json', 'r')); print('📊 Model Performance:'); [print(f'  {name}: {metrics[\"accuracy\"]:.2%%} accuracy') for name, metrics in results.items() if isinstance(metrics, dict) and 'accuracy' in metrics]; print(''); print('📁 Generated Files:'); print('  Models: %OUTPUT_DIR%/models/'); print('  Plots: %OUTPUT_DIR%/plots/'); print('  Logs: %OUTPUT_DIR%/logs/')"
) else (
    call :print_warning "No evaluation results file found"
)

REM Show model files
if exist "%SCRIPT_DIR%%OUTPUT_DIR%\models" (
    call :print_info "Model files:"
    for %%f in ("%SCRIPT_DIR%%OUTPUT_DIR%\models\*.pth") do (
        call :print_info "  %%~nxf"
    )
)
goto :eof

:cleanup_on_exit
call :print_section "Cleanup"

REM Deactivate virtual environment
if defined VIRTUAL_ENV (
    call deactivate
    call :print_info "Virtual environment deactivated"
)

call :print_info "Training session completed at: %date% %time%"
call :print_info "Full log available at: %LOG_FILE%"
goto :eof

REM =============================================================================
REM Main Execution
REM =============================================================================

:main
REM Initialize log file
echo === Gender Classification System Training Log === > "%LOG_FILE%"
echo Started at: %date% %time% >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

REM Change to script directory
cd /d "%SCRIPT_DIR%"

REM Print header
call :print_header

REM Parse command line arguments
call :parse_arguments %*

REM Show configuration
call :print_info "Training Configuration:"
call :print_info "  Epochs: %EPOCHS%"
call :print_info "  Batch Size: %BATCH_SIZE%"
call :print_info "  Learning Rate: %LEARNING_RATE%"
call :print_info "  Output Directory: %OUTPUT_DIR%"
call :print_info "  Skip Ensemble: %SKIP_ENSEMBLE%"
call :print_info "  Skip Optimization: %SKIP_OPTIMIZATION%"
call :print_info "  GPU Only: %GPU_ONLY%"
call :print_info "  Quick Mode: %QUICK_MODE%"
echo.

REM System checks
call :check_python_version
if errorlevel 1 exit /b 1

call :check_gpu_availability
if errorlevel 1 exit /b 1

call :check_disk_space
call :check_memory

REM Environment setup
call :setup_virtual_environment
if errorlevel 1 exit /b 1

call :install_pytorch
if errorlevel 1 exit /b 1

call :install_dependencies
if errorlevel 1 exit /b 1

call :verify_installation
if errorlevel 1 exit /b 1

REM Training preparation
call :prepare_training_environment
if errorlevel 1 exit /b 1

call :create_config_file

REM Record start time
set START_TIME=%time%

REM Run training
call :run_training
if errorlevel 1 exit /b 1

REM Calculate training time (simplified)
set END_TIME=%time%

REM Show results and run demo
call :show_results_summary
call :run_demo

REM Final message
echo.
call :print_success "🎉 Task A Complete System Training Finished Successfully!"
call :print_info "Check the %OUTPUT_DIR% directory for all generated files."
call :print_info "Use demo_predictions.py to test your trained models."

REM Cleanup
call :cleanup_on_exit

exit /b 0

REM Call main function
call :main %*
