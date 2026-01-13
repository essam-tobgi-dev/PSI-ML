# How to Compile and Run All Tests

## Method 1: Batch Script (Easiest - Windows)

### Prerequisites
- Install MinGW-w64 or similar (for g++ compiler)
- Add g++ to your PATH

### Steps
1. Open Command Prompt
2. Navigate to the tests directory:
   ```cmd
   cd C:\Users\asus\source\repos\PsiML++\PsiML++\tests
   ```

3. Run the batch script:
   ```cmd
   compile_and_run.bat
   ```

This will automatically:
- Compile all 8 test files
- Run each test
- Show pass/fail status
- Display a summary

---

## Method 2: CMake (Recommended for Cross-Platform)

### Prerequisites
- Install CMake (https://cmake.org/download/)
- Install a C++ compiler (Visual Studio, MinGW, or Clang)

### Steps

#### Windows with Visual Studio:
```cmd
cd C:\Users\asus\source\repos\PsiML++\PsiML++\tests

# Create build directory
mkdir build
cd build

# Generate Visual Studio project
cmake ..

# Build all tests
cmake --build . --config Release

# Run tests
ctest -C Release --output-on-failure
```

#### Windows with MinGW:
```cmd
cd C:\Users\asus\source\repos\PsiML++\PsiML++\tests

# Create build directory
mkdir build
cd build

# Generate Makefiles
cmake -G "MinGW Makefiles" ..

# Build all tests
cmake --build .

# Run tests
ctest --output-on-failure
```

---

## Method 3: Visual Studio IDE

### Option A: Add Tests to Existing Project

1. Open your PsiML++.sln in Visual Studio
2. For each test file:
   - Right-click the project → Add → Existing Item
   - Select the test file (e.g., test_device.cpp)
3. Right-click project → Properties:
   - C/C++ → General → Additional Include Directories: Add parent directory
   - Linker → System → SubSystem: Console
4. Build and run each test individually

### Option B: Create Separate Test Project

1. Right-click solution → Add → New Project
2. Choose "Console Application"
3. Name it "PsiMLTests"
4. Delete the auto-generated main.cpp
5. Add all test files and source files
6. Configure include directories
7. Build and run

---

## Method 4: Manual Compilation (Windows Command Line)

If you have g++ installed via MinGW:

```cmd
cd C:\Users\asus\source\repos\PsiML++\PsiML++\tests

REM Create output directory
mkdir bin

REM Compile core tests
g++ -std=c++20 -I.. test_device.cpp ..\src\core\device.cpp ..\src\core\logging.cpp ..\src\core\memory.cpp -o bin\test_device.exe
g++ -std=c++20 -I.. test_logging.cpp ..\src\core\logging.cpp ..\src\core\device.cpp ..\src\core\memory.cpp -o bin\test_logging.exe
g++ -std=c++20 -I.. test_memory.cpp ..\src\core\memory.cpp ..\src\core\device.cpp ..\src\core\logging.cpp -o bin\test_memory.exe

REM Compile math tests
g++ -std=c++20 -I.. test_vector.cpp ..\src\core\memory.cpp ..\src\core\device.cpp ..\src\core\logging.cpp -o bin\test_vector.exe
g++ -std=c++20 -I.. test_matrix.cpp ..\src\core\memory.cpp ..\src\core\device.cpp ..\src\core\logging.cpp -o bin\test_matrix.exe
g++ -std=c++20 -I.. test_tensor.cpp ..\src\core\memory.cpp ..\src\core\device.cpp ..\src\core\logging.cpp -o bin\test_tensor.exe
g++ -std=c++20 -I.. test_random.cpp ..\src\core\memory.cpp ..\src\core\device.cpp ..\src\core\logging.cpp -o bin\test_random.exe
g++ -std=c++20 -I.. test_blas.cpp ..\src\core\memory.cpp ..\src\core\device.cpp ..\src\core\logging.cpp ..\src\math\linalg\blas.cpp -o bin\test_blas.exe

REM Run all tests
bin\test_device.exe
bin\test_logging.exe
bin\test_memory.exe
bin\test_vector.exe
bin\test_matrix.exe
bin\test_tensor.exe
bin\test_random.exe
bin\test_blas.exe
```

---

## Method 5: PowerShell Script

Save this as `run_tests.ps1`:

```powershell
$tests = @(
    @{name="device"; deps="device.cpp logging.cpp memory.cpp"}
    @{name="logging"; deps="logging.cpp device.cpp memory.cpp"}
    @{name="memory"; deps="memory.cpp device.cpp logging.cpp"}
    @{name="vector"; deps="memory.cpp device.cpp logging.cpp"}
    @{name="matrix"; deps="memory.cpp device.cpp logging.cpp"}
    @{name="tensor"; deps="memory.cpp device.cpp logging.cpp"}
    @{name="random"; deps="memory.cpp device.cpp logging.cpp"}
    @{name="blas"; deps="memory.cpp device.cpp logging.cpp linalg\blas.cpp"}
)

New-Item -ItemType Directory -Force -Path "bin" | Out-Null

$passed = 0
$failed = 0

foreach ($test in $tests) {
    $name = $test.name
    $deps = $test.deps -split ' ' | ForEach-Object { "..\src\core\$_" }

    Write-Host "`nCompiling test_$name..." -ForegroundColor Yellow

    $cmd = "g++ -std=c++20 -I.. test_$name.cpp " + ($deps -join ' ') + " -o bin\test_$name.exe"
    Invoke-Expression $cmd

    if ($LASTEXITCODE -eq 0) {
        Write-Host "Running test_$name..." -ForegroundColor Cyan
        & "bin\test_$name.exe"

        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ PASSED: test_$name" -ForegroundColor Green
            $passed++
        } else {
            Write-Host "✗ FAILED: test_$name" -ForegroundColor Red
            $failed++
        }
    } else {
        Write-Host "✗ COMPILATION FAILED: test_$name" -ForegroundColor Red
        $failed++
    }
}

Write-Host "`n========================================" -ForegroundColor White
Write-Host "Test Summary" -ForegroundColor White
Write-Host "========================================" -ForegroundColor White
Write-Host "Passed: $passed" -ForegroundColor Green
Write-Host "Failed: $failed" -ForegroundColor Red
```

Run with: `powershell -ExecutionPolicy Bypass -File run_tests.ps1`

---

## Troubleshooting

### "g++ is not recognized"
- Install MinGW-w64: https://www.mingw-w64.org/downloads/
- Or use MSYS2: https://www.msys2.org/
- Add to PATH: `C:\msys64\mingw64\bin` (or wherever installed)

### "Cannot find source files"
- Make sure you're in the `tests` directory
- Check that source files exist in `../src/core/` and `../src/math/`

### Compilation errors
- Ensure C++20 support (`-std=c++20`)
- Check that all header files are present
- Verify include paths are correct

### Tests fail
- Check that implementations match the headers
- Look at the assertion that failed
- Review the specific test function that failed

---

## Quick Start (Recommended)

**Fastest way to get started:**

1. Install MinGW-w64 or ensure you have g++ in PATH
2. Open Command Prompt in the tests directory
3. Run: `compile_and_run.bat`

That's it! The script will compile and run everything automatically.
