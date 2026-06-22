#include "AppBackend.h"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <clocale>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {
ModelFactory::ModelType chooseModel() {
    std::cout << "\nAvailable models:\n";
    for (const auto& item : AppBackend::availableModels()) {
        std::cout << "  " << static_cast<int>(item.first) << ". " << item.second << "\n";
    }
    std::cout << "Choose model: ";
    int raw = 0;
    std::cin >> raw;
    return static_cast<ModelFactory::ModelType>(raw);
}
}

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    std::setlocale(LC_ALL, ".UTF-8");

    AppBackend backend;

    while (true) {
        std::cout << "\n==============================\n";
        std::cout << "MNIST Command Line Interface\n";
        std::cout << "1. Train model\n";
        std::cout << "2. Test model\n";
        std::cout << "0. Exit\n";
        std::cout << "Select: ";

        int choice = -1;
        std::cin >> choice;
        if (choice == 0) {
            break;
        }

        try {
            if (choice == 1) {
                TrainOptions options;
                options.modelType = chooseModel();
                std::cout << "Epochs (default " << AppBackend::defaultEpochs(options.modelType) << "): ";
                std::cin >> options.epochs;
                std::cout << "Continue existing parameter file? (1=yes, 0=no): ";
                int keep = 0;
                std::cin >> keep;
                options.continueFromExisting = keep == 1;

                const TrainResult result = backend.trainModel(options);
                std::cout << "\n" << result.summary << "\n";
            } else if (choice == 2) {
                TestOptions options;
                options.modelType = chooseModel();
                const TestResult result = backend.testModel(options);

                std::cout << "\nModel: " << result.modelName << "\n";
                std::cout << "Parameters: " << result.parameterFile << "\n";
                std::cout << "ROC data: " << result.rocDataFile << "\n";
                std::cout << std::fixed << std::setprecision(4);
                std::cout << "Accuracy:  " << result.metrics.accuracy * 100.0 << "%\n";
                std::cout << "Precision: " << result.metrics.precision * 100.0 << "%\n";
                std::cout << "Recall:    " << result.metrics.recall * 100.0 << "%\n";
                std::cout << "F1 Score:  " << result.metrics.f1 * 100.0 << "%\n";
                std::cout << "AUC:       " << result.metrics.auc << "\n";
            }
        } catch (const std::exception& ex) {
            std::cerr << "\nError: " << ex.what() << "\n";
        }
    }

    return 0;
}
