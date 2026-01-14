@echo off
echo ========================================
echo PsiML++ Test Suite Compiler and Runner
echo ========================================
echo.

:: ANSI colors
set "GREEN=[32m"
set "YELLOW=[33m"
set "RED=[31m"
set "RESET=[0m"

REM Set compiler (change this if using a different compiler)
set CXX=g++
set CXXFLAGS=-std=c++20 -I.. -Wall -O2

REM Directories
set SRC_DIR=..\src
set CORE_DIR=%SRC_DIR%\core
set MATH_DIR=%SRC_DIR%\math
set LINALG_DIR=%MATH_DIR%\linalg

echo Checking for compiler...
where %CXX% >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: g++ not found in PATH
    echo Please install MinGW or use Visual Studio instead
    echo See README.md for Visual Studio instructions
    pause
    exit /b 1
)
echo Found %CXX%
echo.

REM Create bin directory for executables
if not exist "bin" mkdir bin

echo ========================================
echo Compiling Core Tests
echo ========================================
echo.

echo %YELLOW%[1/16] Compiling test_device...%RESET%
%CXX% %CXXFLAGS% test_device.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp %CORE_DIR%\memory.cpp -o bin\test_device.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_device
    pause
    exit /b 1
)
echo  %GREEN%test_device compiled%RESET%

echo %YELLOW%[2/16] Compiling test_logging...%RESET%
%CXX% %CXXFLAGS% test_logging.cpp %CORE_DIR%\logging.cpp %CORE_DIR%\device.cpp %CORE_DIR%\memory.cpp -o bin\test_logging.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_logging
    pause
    exit /b 1
)
echo  %GREEN%test_logging compiled%RESET%

echo %YELLOW%[3/16] Compiling test_memory...%RESET%
%CXX% %CXXFLAGS% test_memory.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp -o bin\test_memory.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_memory
    pause
    exit /b 1
)
echo  %GREEN%test_memory compiled%RESET%

echo.
echo ========================================
echo Compiling Math Tests
echo ========================================
echo.

echo %YELLOW%[4/16] Compiling test_vector...%RESET%
%CXX% %CXXFLAGS% test_vector.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp -o bin\test_vector.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_vector
    pause
    exit /b 1
)
echo  %GREEN%test_vector compiled%RESET%

echo %YELLOW%[5/16] Compiling test_matrix...%RESET%
%CXX% %CXXFLAGS% test_matrix.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp -o bin\test_matrix.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_matrix
    pause
    exit /b 1
)
echo  %GREEN%test_matrix compiled%RESET%

echo %YELLOW%[6/16] Compiling test_tensor...%RESET%
%CXX% %CXXFLAGS% test_tensor.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp -o bin\test_tensor.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_tensor
    pause
    exit /b 1
)
echo  %GREEN%test_tensor compiled%RESET%

echo %YELLOW%[7/16] Compiling test_random...%RESET%
%CXX% %CXXFLAGS% test_random.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp -o bin\test_random.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_random
    pause
    exit /b 1
)
echo  %GREEN%test_random compiled%RESET%

echo.
echo ========================================
echo Compiling Linear Algebra Tests
echo ========================================
echo.

echo %YELLOW%[8/16] Compiling test_blas...%RESET%
%CXX% %CXXFLAGS% test_blas.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp %LINALG_DIR%\blas.cpp -o bin\test_blas.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_blas
    pause
    exit /b 1
)
echo  %GREEN%test_blas compiled%RESET%

echo %YELLOW%[9/16] Compiling test_decomposition...
%CXX% %CXXFLAGS% test_decomposition.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp %LINALG_DIR%\blas.cpp -o bin\test_decomposition.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_decomposition
    pause
    exit /b 1
)
echo  %GREEN%test_decomposition compiled%RESET%

echo %YELLOW%[10/16] Compiling test_solvers...%RESET%
%CXX% %CXXFLAGS% test_solvers.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp %LINALG_DIR%\blas.cpp -o bin\test_solvers.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_solvers
    pause
    exit /b 1
)
echo  %GREEN%test_solvers compiled%RESET%

echo %YELLOW%[11/16] Compiling test_eigen...%RESET%
%CXX% %CXXFLAGS% test_eigen.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp %LINALG_DIR%\blas.cpp -o bin\test_eigen.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_eigen
    pause
    exit /b 1
)
echo  %GREEN%test_eigen compiled%RESET%

echo.
echo ========================================
echo Compiling Operations Tests
echo ========================================
echo.

echo %YELLOW%[12/16] Compiling test_arithmetic...%RESET%
%CXX% %CXXFLAGS% test_arithmetic.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp -o bin\test_arithmetic.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_arithmetic
    pause
    exit /b 1
)
echo  %GREEN%test_arithmetic compiled%RESET%

echo %YELLOW%[13/16] Compiling test_reduction...%RESET%
%CXX% %CXXFLAGS% test_reduction.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp -o bin\test_reduction.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_reduction
    pause
    exit /b 1
)
echo  %GREEN%test_reduction compiled%RESET%

echo %YELLOW%[14/16] Compiling test_broadcasting...%RESET%
%CXX% %CXXFLAGS% test_broadcasting.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp -o bin\test_broadcasting.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_broadcasting
    pause
    exit /b 1
)
echo  %GREEN%test_broadcasting compiled%RESET%

echo %YELLOW%[15/16] Compiling test_statistics...%RESET%
%CXX% %CXXFLAGS% test_statistics.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp -o bin\test_statistics.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_statistics
    pause
    exit /b 1
)
echo  %GREEN%test_statistics compiled%RESET%

echo.
echo ========================================
echo Compiling ML Tests
echo ========================================
echo.

echo %YELLOW%[16/16] Compiling test_ml...%RESET%
%CXX% %CXXFLAGS% test_ml.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp %LINALG_DIR%\blas.cpp -o bin\test_ml.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_ml
    pause
    exit /b 1
)
echo  %GREEN%test_ml compiled%RESET%

echo.
echo ========================================
echo All tests compiled successfully!
echo ========================================
echo.
echo Running tests...
echo.

set TOTAL=0
set PASSED=0
set FAILED=0

REM Run each test
for %%t in (device logging memory vector matrix tensor random blas decomposition solvers eigen arithmetic reduction broadcasting statistics ml) do (
    set /a TOTAL+=1
    echo.
    echo ========================================
    echo Running test_%%t
    echo ========================================
    bin\test_%%t.exe
    if errorlevel 1 (
        echo %RED%FAILED: test_%%t%RESET%
        set /a FAILED+=1
    ) else (
        echo %GREEN%PASSED: test_%%t%RESET%
        set /a PASSED+=1
    )
)

echo.
echo ========================================
echo Test Summary
echo ========================================
echo Total tests: %TOTAL%
echo %GREEN%Passed: %PASSED%%RESET%
echo %RED%Failed: %FAILED%%RESET%
echo.

if %FAILED% gtr 0 (
    echo %RED%SOME TESTS FAILED%RESET%
    pause
    exit /b 1
) else (
    echo %GREEN%mALL TESTS PASSED!%RESET%
    pause
    exit /b 0
)
