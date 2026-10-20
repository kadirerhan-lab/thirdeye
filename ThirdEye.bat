@echo off
rem ThirdEye demo baslatici: cift tiklayin, tarayici otomatik acilir.
cd /d "%~dp0"
where node >nul 2>nul || (echo Node.js bulunamadi. https://nodejs.org adresinden kurun. & pause & exit /b 1)
node server.js
pause
