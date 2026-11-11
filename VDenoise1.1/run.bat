@echo off
chcp 65001 >nul 2>&1
title Acoustic Studio - AI Audio Denoiser Workstation
cd /d "%~dp0"

echo ==============================================================================
echo       ACOUSTIC STUDIO // AI AUDIO DENOISER WORKSTATION
echo ==============================================================================
echo.

REM -----------------------------------------------------------------------------
REM 1. TIM KIEM TRINH THONG DICH PYTHON
REM -----------------------------------------------------------------------------
set "SYS_PYTHON="

REM Kiem tra lenh python tren PATH
python -c "import sys; sys.exit(0 if sys.version_info.major == 3 and sys.version_info.minor >= 9 else 1)" >nul 2>&1
if not errorlevel 1 for /f "delims=" %%i in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "SYS_PYTHON=%%i"

REM Neu chua tim thay, thu kiem tra python3 tren PATH
if not defined SYS_PYTHON (
    python3 -c "import sys; sys.exit(0 if sys.version_info.major == 3 and sys.version_info.minor >= 9 else 1)" >nul 2>&1
    if not errorlevel 1 for /f "delims=" %%i in ('python3 -c "import sys; print(sys.executable)" 2^>nul') do set "SYS_PYTHON=%%i"
)

REM Thu tim o cac thu muc cai dat pho bien tren Windows
if defined SYS_PYTHON goto :python_found

for %%p in (
    "C:\ProgramData\Anaconda3\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"
    "C:\Program Files\Python312\python.exe"
    "C:\Program Files\Python311\python.exe"
    "C:\Program Files\Python310\python.exe"
) do (
    if not defined SYS_PYTHON if exist %%p (
        %%p -c "import sys; sys.exit(0)" >nul 2>&1
        if not errorlevel 1 set "SYS_PYTHON=%%~p"
    )
)

if defined SYS_PYTHON goto :python_found

echo [LOI] Khong tim thay trinh thong dich Python 3.9 tro len tren may tinh!
echo Vui long tai va cai dat Python tai: https://www.python.org/downloads/
echo (Luu y: Nho tich chon vao muc 'Add Python to PATH' khi cai dat).
echo.
pause
exit /b 1

:python_found
echo [1/3] Da tim thay Python: %SYS_PYTHON%

REM -----------------------------------------------------------------------------
REM 2. XAC DINH MOI TRUONG THUC THI PHU HOP (.venv HOAC PYTHON HE THONG)
REM -----------------------------------------------------------------------------
set "PY_EXEC="

REM Uu tien 1: Kiem tra neu Python he thong da co san day du tat ca thu vien
"%SYS_PYTHON%" -c "import gradio, soundfile, scipy, matplotlib, imageio_ffmpeg, torch, torchaudio, df" >nul 2>&1
if not errorlevel 1 goto :sys_python_fully_ready

REM Uu tien 2: Kiem tra neu .venv da ton tai va chay duoc tat ca thu vien
if not exist ".venv\Scripts\python.exe" goto :try_venv_pip
".venv\Scripts\python.exe" -c "import gradio, soundfile, scipy, matplotlib, imageio_ffmpeg, torch, torchaudio, df" >nul 2>&1
if not errorlevel 1 goto :venv_fully_ready

:try_venv_pip
REM Uu tien 3: Kiem tra neu .venv ton tai va co san module pip
if not exist ".venv\Scripts\python.exe" goto :make_new_venv
".venv\Scripts\python.exe" -c "import sys, pip" >nul 2>&1
if not errorlevel 1 (
    set "PY_EXEC=.venv\Scripts\python.exe"
    goto :venv_ready
)

:make_new_venv
REM Neu .venv chua co hoac thieu pip, thu tao moi .venv
echo [INFO] Dang khoi tao moi truong ao .venv...
if exist ".venv" rmdir /s /q .venv >nul 2>&1
"%SYS_PYTHON%" -m venv .venv >nul 2>&1

if not exist ".venv\Scripts\python.exe" goto :fallback_sys
".venv\Scripts\python.exe" -c "import sys, pip" >nul 2>&1
if not errorlevel 1 (
    set "PY_EXEC=.venv\Scripts\python.exe"
    goto :venv_ready
)

:fallback_sys
REM Neu tao .venv khong co pip (vi du do Anaconda), fallback ve dung truc tiep SYS_PYTHON
echo [INFO] Moi truong .venv khong kha dung hoac thieu pip. Chuyen sang dung Python he thong.
set "PY_EXEC=%SYS_PYTHON%"
goto :install_packages

:sys_python_fully_ready
set "PY_EXEC=%SYS_PYTHON%"
echo [2/3] Su dung Python he thong [da san sang day du thu vien]: %PY_EXEC%
echo [3/3] Cac thu vien da duoc cai dat day du va san sang!
goto :launch_app

:venv_fully_ready
set "PY_EXEC=.venv\Scripts\python.exe"
echo [2/3] Moi truong ao .venv da san sang: %PY_EXEC%
echo [3/3] Cac thu vien da duoc cai dat day du va san sang!
goto :launch_app

:venv_ready
echo [2/3] Moi truong ao .venv da san sang: %PY_EXEC%

:install_packages
REM -----------------------------------------------------------------------------
REM 3. KIEM TRA VA CAI DAT THU VIEN
REM -----------------------------------------------------------------------------
echo [INFO] Dang kiem tra cac thu vien can thiet...
"%PY_EXEC%" -c "import gradio, soundfile, scipy, matplotlib, imageio_ffmpeg, torch, torchaudio, df" >nul 2>&1
if not errorlevel 1 (
    echo [3/3] Cac thu vien da duoc cai dat day du va san sang!
    goto :launch_app
)

echo.
echo ==============================================================================
echo [CAI DAT] Phat hien thieu thu vien, dang tien hanh cai dat tu dong...
echo (Qua trinh nay chi dien ra lan dau tien, co the mat vai phut tuy toc do mang)
echo ==============================================================================
echo.
echo [1/3] Dang cap nhat cong cu pip...
"%PY_EXEC%" -m pip install --upgrade pip >nul 2>&1

echo [2/3] Dang cai dat cac goi tu requirements.txt...
"%PY_EXEC%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [LOI] Khong the cai dat cac goi tu requirements.txt!
    echo Vui long kiem tra ket noi mang va thu lai.
    echo.
    pause
    exit /b 1
)

echo [3/3] Dang cai dat goi deepfilternet 0.5.6...
"%PY_EXEC%" -m pip install --no-deps deepfilternet==0.5.6
if errorlevel 1 (
    echo.
    echo [LOI] Khong the cai dat deepfilternet!
    echo.
    pause
    exit /b 1
)
echo.
echo [THANH CONG] Da hoan tat cai dat cac thu vien can thiet!
echo.

:launch_app
REM -----------------------------------------------------------------------------
REM KIEM TRA CHE DO DRY-RUN / CHECK
REM -----------------------------------------------------------------------------
if "%1"=="--check" (
    echo.
    echo [KIEM TRA THANH CONG] Moi truong va cac thu vien deu hoat dong tot!
    exit /b 0
)

REM -----------------------------------------------------------------------------
REM 4. KHOI CHAY UNG DUNG APP.PY
REM -----------------------------------------------------------------------------
echo.
echo ==============================================================================
echo [KHOI DONG] Dang khoi chay Acoustic Studio...
echo Truy cap giao dien tai trinh duyet web (hoac nhap vao duong link ben duoi)
echo Nhan Ctrl + C trong cua so nay neu muon tat ung dung.
echo ==============================================================================
echo.

if not defined DENOISE_ENGINE set "DENOISE_ENGINE=dfnet2"

"%PY_EXEC%" app.py %*

if errorlevel 1 (
    echo.
    echo [LOI] Ung dung da dung lai voi ma loi: %errorlevel%
    echo.
    pause
)
