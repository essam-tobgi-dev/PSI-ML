@echo off
echo ========================================
echo PsiML++ Test Suite Compiler and Runner
echo ========================================
echo.

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

echo [1/15] Compiling test_device...
%CXX% %CXXFLAGS% test_device.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp %CORE_DIR%\memory.cpp -o bin\test_device.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_device
    pause
    exit /b 1
)
echo ✓ test_device compiled

echo [2/15] Compiling test_logging...
%CXX% %CXXFLAGS% test_logging.cpp %CORE_DIR%\logging.cpp %CORE_DIR%\device.cpp %CORE_DIR%\memory.cpp -o bin\test_logging.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_logging
    pause
    exit /b 1
)
echo ✓ test_logging compiled

echo [3/15] Compiling test_memory...
%CXX% %CXXFLAGS% test_memory.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp -o bin\test_memory.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_memory
    pause
    exit /b 1
)
echo ✓ test_memory compiled

echo.
echo ========================================
echo Compiling Math Tests
echo ========================================
echo.

echo [4/15] Compiling test_vector...
%CXX% %CXXFLAGS% test_vector.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp -o bin\test_vector.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_vector
    pause
    exit /b 1
)
echo ✓ test_vector compiled

echo [5/15] Compiling test_matrix...
%CXX% %CXXFLAGS% test_matrix.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp -o bin\test_matrix.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_matrix
    pause
    exit /b 1
)
echo ✓ test_matrix compiled

echo [6/15] Compiling test_tensor...
%CXX% %CXXFLAGS% test_tensor.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp -o bin\test_tensor.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_tensor
    pause
    exit /b 1
)
echo ✓ test_tensor compiled

echo [7/15] Compiling test_random...
%CXX% %CXXFLAGS% test_random.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp -o bin\test_random.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_random
    pause
    exit /b 1
)
echo ✓ test_random compiled

echo.
echo ========================================
echo Compiling Linear Algebra Tests
echo ========================================
echo.

echo [8/15] Compiling test_blas...
%CXX% %CXXFLAGS% test_blas.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp %LINALG_DIR%\blas.cpp -o bin\test_blas.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_blas
    pause
    exit /b 1
)
echo ✓ test_blas compiled

echo [9/15] Compiling test_decomposition...
%CXX% %CXXFLAGS% test_decomposition.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp %LINALG_DIR%\blas.cpp -o bin\test_decomposition.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_decomposition
    pause
    exit /b 1
)
echo ✓ test_decomposition compiled

echo [10/15] Compiling test_solvers...
%CXX% %CXXFLAGS% test_solvers.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp %LINALG_DIR%\blas.cpp -o bin\test_solvers.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_solvers
    pause
    exit /b 1
)
echo ✓ test_solvers compiled

echo [11/15] Compiling test_eigen...
%CXX% %CXXFLAGS% test_eigen.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp %LINALG_DIR%\blas.cpp -o bin\test_eigen.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_eigen
    pause
    exit /b 1
)
echo ✓ test_eigen compiled

echo.
echo ========================================
echo Compiling Operations Tests
echo ========================================
echo.

echo [12/15] Compiling test_arithmetic...
%CXX% %CXXFLAGS% test_arithmetic.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp -o bin\test_arithmetic.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_arithmetic
    pause
    exit /b 1
)
echo ✓ test_arithmetic compiled

echo [13/15] Compiling test_reduction...
%CXX% %CXXFLAGS% test_reduction.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp -o bin\test_reduction.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_reduction
    pause
    exit /b 1
)
echo ✓ test_reduction compiled

echo [14/15] Compiling test_broadcasting...
%CXX% %CXXFLAGS% test_broadcasting.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp -o bin\test_broadcasting.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_broadcasting
    pause
    exit /b 1
)
echo ✓ test_broadcasting compiled

echo [15/15] Compiling test_statistics...
%CXX% %CXXFLAGS% test_statistics.cpp %CORE_DIR%\memory.cpp %CORE_DIR%\device.cpp %CORE_DIR%\logging.cpp -o bin\test_statistics.exe
if %errorlevel% neq 0 (
    echo Failed to compile test_statistics
    pause
    exit /b 1
)
echo ✓ test_statistics compiled

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
for %%t in (device logging memory vector matrix tensor random blas decomposition solvers eigen arithmetic reduction broadcasting statistics) do (
    set /a TOTAL+=1
    echo.
    echo ========================================
    echo Running test_%%t
    echo ========================================
    bin\test_%%t.exe
    if errorlevel 1 (
        echo [31mFAILED: test_%%t[0m
        set /a FAILED+=1
    ) else (
        echo [32mPASSED: test_%%t[0m
        set /a PASSED+=1
    )
)

echo.
echo ========================================
echo Test Summary
echo ========================================
echo Total tests: %TOTAL%
echo Passed: %PASSED%
echo Failed: %FAILED%
echo.

if %FAILED% gtr 0 (
    echo [31mSOME TESTS FAILED[0m
    pause
    exit /b 1
) else (
    echo [32mALL TESTS PASSED![0m
    pause
    exit /b 0
)
