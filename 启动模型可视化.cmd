@echo off
setlocal
set "PATH=C:\Qt\6.10.2\mingw_64\bin;C:\Qt\Tools\mingw1310_64\bin;%PATH%"
cd /d "%~dp0build-qt\build"
start "" "cnn_gui.exe"
