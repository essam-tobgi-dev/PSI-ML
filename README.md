# PsiML++

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/4151d17c-d367-48ce-9fdb-d64eff2261c9" />

**PsiML++** is a modern C++ machine learning library designed for high-performance computations, providing core utilities, linear algebra operations, machine learning algorithms, preprocessing tools, and vision functionalities. The library is structured for modular development, testing, and easy integration into C++ projects.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Features](#features)
4. [Dependencies](#dependencies)
5. [Building the Project](#building-the-project)
6. [Running Tests](#running-tests)
7. [Modules Overview](#modules-overview)
8. [Contributing](#contributing)
9. [License](#license)

---

## Project Overview

PsiML++ is a versatile C++ library aimed at researchers, developers, and engineers who need:

* Fast and efficient **matrix/tensor operations**.
* Built-in **machine learning algorithms** like linear regression, logistic regression, PCA, KMeans, and SVM.
* **Preprocessing utilities** for dataset normalization, encoding, and scaling.
* **Vision tools** for image I/O, processing, and drawing.
* Utilities for logging, profiling, and serialization.

The library leverages modern C++ features and is designed to be **cross-platform** with minimal dependencies.

---

## Directory Structure

```text
PsiML++
│   PsiML++.slnx                  # Visual Studio 2026 solution file
│   .gitignore
│   .gitattributes
│
├── PsiML++/                       # Main source folder
│   ├── include/                   # Header files
│   │   ├── core/                  # Core utilities (logging, memory, device)
│   │   ├── math/                  # Linear algebra and math utilities
│   │   │   ├── linalg/            # BLAS, solvers, decomposition
│   │   │   └── ops/               # Arithmetic, broadcasting, reduction
│   │   ├── ml/                    # Machine learning algorithms
│   │   │   ├── algorithms/        # KMeans, Linear/Logistic Regression, PCA, SVM
│   │   │   ├── optimizers/        # Gradient Descent, Momentum, SGD
│   │   │   └── preprocessing/     # Encoders, normalizers, scalars
│   │   ├── utils/                  # Data loaders, file IO, model IO, profiling
│   │   └── vision/                 # Image processing, drawing
│   │
│   ├── src/                        # Source files (.cpp)
│   │   ├── core/
│   │   ├── math/
│   │   │   └── linalg/
│   │   └── vision/
│   │
│   ├── dependencies/               # Third-party libraries (if any)
│   ├── tests/                      # Unit tests and test data
│   │   ├── bin/                    # Compiled test binaries
│   │   ├── data/                   # Sample CSV datasets
│   │   └── *.cpp                   # Individual test files
│   └── x64/                        # Build output (Debug/Release)
```

---

## Features

### Core Utilities

* Device management, memory management, custom exception handling, logging, timers, and configuration management.

### Math & Linear Algebra

* Tensor, matrix, vector types
* Linear algebra operations: BLAS routines, decompositions, eigenvalues, solvers, statistics
* Element-wise operations: arithmetic, broadcasting, reduction, statistics

### Machine Learning

* Algorithms: Linear Regression, Logistic Regression, PCA, KMeans, SVM
* Optimizers: Gradient Descent, Momentum, SGD
* Preprocessing: Normalizers, Encoders, Scalar transformations
* Dataset handling and metrics evaluation

### Vision & Image Processing

* Image I/O, drawing primitives, and processing operations

### Utilities

* File I/O, Model serialization/deserialization, data loading, string utilities, profiler

---

## Dependencies

* Visual Studio 2026
* C++17 or higher
* Linking OpenCV (for image processing in vision module)

---

## Building the Project

1. Open `PsiML++.slnx` in **Visual Studio 2026**.
2. Select your desired build configuration: `Debug` or `Release`.
3. Build the solution using **Build → Build Solution**.
4. The output libraries and binaries will be generated under `x64/Debug` or `x64/Release`.

### Notes:

* All source files are organized into modules (`core`, `math`, `ml`, `vision`) for modular compilation.
* Dependencies (like OpenCV) should be linked via the **Project Properties → VC++ Directories → Include/Library Directories**.

---

## Running Tests

Tests are located in `PsiML++/tests`. You can run them individually or collectively:

### Using Visual Studio:

* Open the test project in the solution and run tests from **Test Explorer**.

### Using Batch Script:

```bat
cd PsiML++\tests
compile_and_run.bat
```

### Test Coverage

* `test_arithmetic.cpp`, `test_blas.cpp`, `test_decomposition.cpp`, `test_ml.cpp`, `test_vision.cpp`, and more.
* Test datasets are included under `tests/data` (`iris_binary.csv`, `housing.csv`, `clustering.csv`).

---

## Modules Overview

### Core

* `config.h`, `device.h`, `memory.h`, `logging.h`, `exception.h`
* Provides basic utilities required for ML computation.

### Math

* `matrix.h`, `tensor.h`, `vector.h`, `random.h`
* Linear algebra (`blas.h`, `solvers.h`, `decomposition.h`) and operations (`arithmetic.h`, `broadcasting.h`, `reduction.h`)

### ML

* Algorithms: `kmeans.h`, `linear_regression.h`, `logistic_regression.h`, `pca.h`, `svm.h`
* Optimizers: `gradient_descent.h`, `momentum.h`, `sgd.h`
* Preprocessing: `encoder.h`, `normalizer.h`, `scalar.h`
* Utilities: `dataset.h`, `metrics.h`, `model.h`

### Vision

* Image manipulation, drawing, and processing via `image.h`, `drawing.h`, `image_io.h`, `image_processing.h`

### Utilities

* `data_loader.h`, `file_io.h`, `model_io.h`, `profiler.h`, `serialization.h`, `string_utils.h`, `timer.h`

---

## Contributing

* Fork the repository and create a feature branch.
* Follow the folder structure when adding new modules.
* Include corresponding tests for new functionalities.
* Submit pull requests with detailed descriptions.

---

## License

```
MIT License
```
