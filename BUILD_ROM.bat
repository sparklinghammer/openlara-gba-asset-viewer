@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion
title Asset Viewer - build the ROM

rem ---------------------------------------------------------------------------
rem  Builds the asset viewer ROM from the models in models\.
rem
rem  Double-click this file. Nothing else to type: the model list lives in
rem  models\pack.json, the path to your OpenLara checkout in settings.json.
rem
rem  You can also drop ANOTHER folder of models onto this .bat to build from it
rem  without touching models\.
rem
rem  The OpenLara folder is only ever read: everything produced stays here.
rem ---------------------------------------------------------------------------

set "HERE=%~dp0"
if "%HERE:~-1%"=="\" set "HERE=%HERE:~0,-1%"
set "BUILDER=%HERE%\scripts\build_rom.py"
set "VERIFIER=%HERE%\scripts\verify_asset_viewer.py"
set "ROMS=%HERE%\roms"
set "LOG=%TEMP%\openlara_asset_viewer_build.log"
set "EXTRA="
if not "%~1"=="" set "EXTRA=--models "%~1""

echo.
echo ===============================================================
echo   ASSET VIEWER  -  build the ROM
echo ===============================================================
echo   Project : %HERE%
if not "%~1"=="" echo   Models  : %~1
echo   Log     : %LOG%
echo.

if not exist "%BUILDER%" (
  echo [ERROR] Build script not found:
  echo         %BUILDER%
  echo.
  echo This .bat has to stay at the root of the project folder.
  goto :end
)
where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python 3 is not on PATH. The converter and the driver both need it.
  echo         Install it from python.org, then: pip install pillow
  goto :end
)
if not exist "%HERE%\settings.json" (
  echo [ERROR] settings.json is missing. It says where your OpenLara checkout is.
  echo.
  echo         Copy settings.example.json to settings.json and set "openlara"
  echo         to the folder that holds src\platform\gba.
  goto :end
)

echo Converting models and compiling... ^(1 to 2 minutes^)
echo.

python "%BUILDER%" %EXTRA% > "%LOG%" 2>&1
set "BUILD_CODE=%ERRORLEVEL%"

rem The ARM compiler emits hundreds of warnings about libtonc that have nothing
rem to do with the models; only the useful lines are shown.
findstr /C:"Sources:" /C:"donor fix" /C:"-> slot" /C:"joints," /C:"note:" /C:"ASSET_VIEWER.PAK:" /C:"ROM:" /C:"Bytes:" /C:"SHA256:" /C:"Mode:" "%LOG%"

if not "%BUILD_CODE%"=="0" (
  echo.
  echo ===============================================================
  echo   BUILD FAILED
  echo ===============================================================
  echo.
  findstr /C:"FAIL:" /C:"Exception" /C:"not found" /C:"is missing" /C:"does not exist" "%LOG%"
  echo.
  echo Full log: %LOG%
  goto :end
)

echo.
echo Verifying the ROM...
echo.

python "%VERIFIER%" --bundle "%HERE%"
if errorlevel 1 (
  echo.
  echo ===============================================================
  echo   THE ROM WAS BUILT BUT DID NOT PASS VERIFICATION
  echo ===============================================================
  goto :end
)

:ok
echo.
echo ===============================================================
echo   DONE
echo ===============================================================
echo.
echo   ROM: %ROMS%\openlara-asset-viewer-custom.gba
echo.
echo   In the viewer: L / R change model, Start changes animation,
echo   Select opens the settings menu.
echo.
choice /C YN /N /T 15 /D N /M "Open the roms folder? [Y/N] (N in 15 s) "
if errorlevel 2 goto :end
start "" "%ROMS%"

:end
echo.
pause
endlocal
