#include "AppBackend.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

std::vector<std::vector<double>> readImages(const std::string& filename, int& numImages);
std::vector<int> readLabels(const std::string& filename, int& numLabels);

namespace {
namespace fs = std::filesystem;

double elapsedSeconds(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
}

std::vector<std::pair<ModelFactory::ModelType, std::string>> AppBackend::availableModels() {
    std::vector<std::pair<ModelFactory::ModelType, std::string>> models;
    for (const auto& item : ModelFactory::getAvailableModels()) {
        models.push_back({static_cast<ModelFactory::ModelType>(item.first), item.second});
    }
    return models;
}

std::string AppBackend::defaultParamFile(ModelFactory::ModelType type) {
    switch (type) {
        case ModelFactory::SVM: return "params/svm-params.txt";
        case ModelFactory::FCNN: return "params/fcnn-params.txt";
        case ModelFactory::CNN: return "params/cnn-params.txt";
        case ModelFactory::LOGISTIC_REGRESSION: return "params/lr-params.txt";
        case ModelFactory::RANDOM_FOREST: return "params/rf-params.txt";
        case ModelFactory::KNN: return "params/knn-params.txt";
        default: return "params/model-params.txt";
    }
}

int AppBackend::defaultEpochs(ModelFactory::ModelType type) {
    switch (type) {
        case ModelFactory::SVM: return 5;
        case ModelFactory::FCNN: return 5;
        case ModelFactory::CNN: return 15;
        case ModelFactory::LOGISTIC_REGRESSION: return 5;
        case ModelFactory::RANDOM_FOREST: return 5;
        case ModelFactory::KNN: return 1;
        default: return 5;
    }
}

std::vector<ModelVisualSpec> AppBackend::visualSpecs() {
    return {
        {ModelFactory::SVM, "SVM", "Linear one-vs-rest baseline for 10 classes.",
         "Input(784) -> Linear heads(10) -> argmax",
         {"Input vector", "Margin heads", "Class selection"},
         {"784 normalized pixel features from MNIST digits.",
          "Ten one-vs-rest linear decision functions optimized with hinge loss and L2-style regularization.",
          "The class with the largest decision margin is returned as the prediction."},
         {"Mini-batch SGD", "Hinge loss", "Interpretable baseline"},
         {"#6b7280", "#8b9bb4", "#b0bacf"}},
        {ModelFactory::FCNN, "FCNN / BP Neural Network", "Classic multilayer perceptron trained by backpropagation.",
         "Input(784) -> Dense(256, ReLU) -> Dense(128, ReLU) -> Softmax(10)",
         {"Input layer", "Hidden layer 1", "Hidden layer 2", "Softmax output"},
         {"The input image is flattened into a 784-dimensional vector.",
          "The first dense layer expands representation capacity to 256 units and applies ReLU activation.",
          "A second hidden layer compresses the representation to 128 nonlinear features.",
          "The final dense head produces ten logits which are normalized by Softmax."},
         {"Momentum optimizer", "Best accuracy in current report", "Nonlinear decision boundary"},
         {"#6d5bd0", "#7c6ee6", "#8b82f4", "#b5a8ff"}},
        {ModelFactory::CNN, "CNN", "Simplified convolution-style pipeline in current project implementation.",
         "Input(28x28) -> simplified feature extractor -> classifier head -> Softmax(10)",
         {"Image input", "Feature maps", "Classifier head", "Probability output"},
         {"A 28 by 28 grayscale digit enters as a spatial image tensor.",
          "The current code approximates a convolution-style feature extraction stage and preserves an image-first pipeline.",
          "The extracted feature representation is consumed by a dense classification head.",
          "Ten normalized output probabilities summarize confidence for each digit class."},
         {"Image-oriented pipeline", "Current scaffold can be upgraded later", "Useful for comparison"},
         {"#4f8f8f", "#5ba7a4", "#78c1ba", "#9ce5db"}},
        {ModelFactory::LOGISTIC_REGRESSION, "Logistic Regression", "Single-layer Softmax classifier.",
         "Input(784) -> Linear projection -> Softmax(10)",
         {"Input vector", "Linear projection", "Softmax output"},
         {"The image is flattened into a single feature vector.",
          "A single affine transformation maps the input directly into ten class logits.",
          "Softmax converts logits into a normalized probability distribution over classes."},
         {"Fast convergence", "Interpretable", "Limited nonlinear capacity"},
         {"#7b6751", "#9a7b58", "#c08d5d"}},
        {ModelFactory::RANDOM_FOREST, "Random Forest", "Ensemble of randomized decision trees.",
         "Input(784) -> 50 decision trees -> vote / probability average",
         {"Input vector", "Bootstrap sampling", "Tree ensemble", "Vote aggregation"},
         {"Normalized pixel features are fed into an ensemble pipeline.",
          "Each tree is trained on sampled data and random feature subsets.",
          "The forest learns multiple nonlinear partitionings of the feature space.",
          "Predictions are aggregated through voting or averaged class probabilities."},
         {"Robust nonlinear baseline", "Native C++ training", "Larger persisted model"},
         {"#4b6a43", "#5e8650", "#7aa564", "#a2c57c"}},
        {ModelFactory::KNN, "KNN", "Prototype-based nearest-neighbor classifier.",
         "Input(784) -> prototype library -> k-nearest vote",
         {"Input vector", "Prototype store", "Distance search", "Vote result"},
         {"The test image is flattened into a feature vector.",
          "Training builds and stores a prototype memory from representative samples.",
          "Inference compares the query against stored prototypes to find the nearest neighbors.",
          "The final class is decided by top-k voting over neighbor labels."},
         {"Non-parametric baseline", "Cheap training", "Costly inference"},
         {"#915f7d", "#aa6f93", "#c286ad", "#dea4ca"}}
    };
}

TrainResult AppBackend::trainModel(const TrainOptions& options) {
    auto dataset = loadTrainDataset();
    auto model = createModel(options.modelType);
    const std::string paramFile = defaultParamFile(options.modelType);

    if (options.continueFromExisting) {
        if (!model->loadParams(resolvePath(paramFile))) {
            throw std::runtime_error("Failed to load existing parameter file: " + paramFile);
        }
    } else {
        model->initWeights();
    }

    const auto started = std::chrono::steady_clock::now();
    model->train(dataset.images, dataset.labels, options.epochs);
    model->saveParams(resolvePath(paramFile));

    TrainResult result;
    result.modelName = model->getName();
    result.parameterFile = paramFile;
    result.epochs = options.epochs;
    result.durationSeconds = elapsedSeconds(started);

    std::ostringstream summary;
    summary << "Model: " << result.modelName << "\n";
    summary << "Mode: " << (options.continueFromExisting ? "continue existing parameters" : "new parameter initialization") << "\n";
    summary << "Epochs: " << result.epochs << "\n";
    summary << "Output: " << result.parameterFile << "\n";
    summary << "Elapsed: " << result.durationSeconds << " s";
    result.summary = summary.str();
    return result;
}

TestResult AppBackend::testModel(const TestOptions& options) {
    auto dataset = loadTestDataset();
    auto model = createModel(options.modelType);
    const std::string paramFile = defaultParamFile(options.modelType);
    const std::string resolvedParam = resolvePath(paramFile);

    if (!model->loadParams(resolvedParam)) {
        throw std::runtime_error("Failed to load parameter file: " + paramFile);
    }

    TestResult result;
    result.modelName = model->getName();
    result.parameterFile = paramFile;
    result.metrics = model->testAndGetMetrics(dataset.images, dataset.labels);
    result.rocDataFile = model->getROCDataFilename();
    result.rocCurve = readRocCurve(resolvePath(result.rocDataFile));

    std::vector<double> precision;
    std::vector<double> recall;
    std::vector<double> f1;
    model->computeMetrics(result.metrics.confusionMatrix, precision, recall, f1);
    for (size_t i = 0; i < precision.size(); ++i) {
        result.classMetrics.push_back({static_cast<int>(i), precision[i], recall[i], f1[i]});
    }
    return result;
}

AppBackend::Dataset AppBackend::loadTrainDataset() {
    Dataset dataset;
    int imageCount = 0;
    int labelCount = 0;
    dataset.images = readImages(resolvePath("train-images-idx3-ubyte"), imageCount);
    dataset.labels = readLabels(resolvePath("train-labels-idx1-ubyte"), labelCount);
    if (imageCount != labelCount) {
        throw std::runtime_error("Train dataset image/label mismatch.");
    }
    return dataset;
}

AppBackend::Dataset AppBackend::loadTestDataset() {
    Dataset dataset;
    int imageCount = 0;
    int labelCount = 0;
    dataset.images = readImages(resolvePath("t10k-images-idx3-ubyte"), imageCount);
    dataset.labels = readLabels(resolvePath("t10k-labels-idx1-ubyte"), labelCount);
    if (imageCount != labelCount) {
        throw std::runtime_error("Test dataset image/label mismatch.");
    }
    return dataset;
}

std::shared_ptr<IClassificationModel> AppBackend::createModel(ModelFactory::ModelType type) const {
    auto model = ModelFactory::createModel(type);
    if (!model) {
        throw std::runtime_error("Unknown model type.");
    }
    return model;
}

std::string AppBackend::resolvePath(const std::string& relative) const {
    const std::vector<fs::path> candidates = {
        fs::path("..") / relative,
        fs::path("..") / "训练集" / relative,
        fs::path(relative),
        fs::path("训练集") / relative
    };
    for (const auto& candidate : candidates) {
        if (fs::exists(candidate)) {
            return candidate.string();
        }
    }
    throw std::runtime_error("Required file not found: " + relative);
}

std::vector<RocPoint> AppBackend::readRocCurve(const std::string& path) const {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("ROC data file not found: " + path);
    }

    std::vector<std::pair<double, int>> ranked;
    double score = 0.0;
    int label = 0;
    while (file >> score >> label) {
        ranked.push_back({score, label});
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& left, const auto& right) {
        if (left.first != right.first) {
            return left.first > right.first;
        }
        return left.second > right.second;
    });

    double positives = 0.0;
    double negatives = 0.0;
    for (const auto& item : ranked) {
        if (item.second == 1) positives += 1.0;
        else negatives += 1.0;
    }

    std::vector<RocPoint> curve{{0.0, 0.0}};
    if (positives <= 0.0 || negatives <= 0.0) {
        curve.push_back({1.0, 1.0});
        return curve;
    }

    double tp = 0.0;
    double fp = 0.0;
    for (size_t i = 0; i < ranked.size();) {
        size_t j = i;
        const double threshold = ranked[i].first;
        while (j < ranked.size() && ranked[j].first == threshold) {
            if (ranked[j].second == 1) tp += 1.0;
            else fp += 1.0;
            ++j;
        }
        curve.push_back({fp / negatives, tp / positives});
        i = j;
    }

    if (curve.back().falsePositiveRate < 1.0 || curve.back().truePositiveRate < 1.0) {
        curve.push_back({1.0, 1.0});
    }
    return curve;
}
