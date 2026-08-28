@echo off
setlocal enableextensions
set JAVA_PATH=C:\Users\stuar\AppData\Local\Programs\Common\i4j_jres\Oda-jK0QgTEmVssfllLP\17.0.16.0.101-zulu_64\bin
echo === Full absolute path ===
"%JAVA_PATH%\java.exe" -version
echo Exit code: %ERRORLEVEL%
echo.
echo === Current directory search ===
pushd "%JAVA_PATH%"
echo Files in cwd containing java in the name:
dir /b java*.*
echo.
echo === Test 1: java.exe (bare) ===
java.exe -version
echo Exit1: %ERRORLEVEL%
echo.
echo === Test 2: .\java.exe (explicit cwd) ===
.\java.exe -version
echo Exit2: %ERRORLEVEL%
echo.
echo === Test 3: full quoted path ===
"%JAVA_PATH%\java.exe" -version
echo Exit3: %ERRORLEVEL%
popd
