# Machine Learning Workbench

This project provides a native C++ MNIST model workbench with both a command-line interface and a Qt desktop interface.

Implemented models:
- Support Vector Machine (SVM)
- Fully Connected Neural Network (FCNN / BP)
- Convolutional Neural Network (CNN)
- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)

## Project layout

- `Models.cpp`: core model implementations, training, testing, metrics, ROC/AUC logic
- `ModelInterface.h`: shared model interface and factory definitions
- `cli_main.cpp`: command-line entry
- `build-qt/gui-src/`: Qt desktop UI source
- `启动模型可视化.cmd`: Windows launcher for the desktop app

## Usage

### Command-line interface

Build the project, then run the CLI executable:

```powershell
.\build-qt\build\cnn_cli.exe
```

The CLI supports:
- selecting a model
- training from new initialization or existing parameters
- testing and viewing metrics

### Desktop interface

After building the Qt target, start the desktop application:

```powershell
.\启动模型可视化.cmd
```

The desktop application provides three tabs:
- `Train`: choose a model and run training
- `Test`: run evaluation, inspect metrics, ROC, confusion matrix and per-class bars
- `Visualize`: inspect structure diagrams for each implemented model

## Build

The Qt application is built from `build-qt/gui-src/` with Qt 6 and MinGW. The repository only keeps source files; local datasets, parameter files, generated ROC files and deployed binaries are excluded from version control.
