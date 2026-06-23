#include "ModelInterface.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <cstdint>
#include <array>
#include <limits>
#include <clocale>
#include <utility>
#include <functional>
#include <filesystem>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// ==================== Global Hyperparameters ====================

// SVM hyperparameters
const double SVM_LEARNING_RATE = 1e-3;          
const double SVM_REGULARIZATION_C = 0.01;     
const int SVM_EPOCHS = 5;                     
const int SVM_BATCH_SIZE = 64;

// FCNN hyperparameters
const double FCNN_LEARNING_RATE = 1e-3;       
const double FCNN_MOMENTUM = 0.9;            
const int FCNN_EPOCHS = 5;                     
const int FCNN_BATCH_SIZE = 64;
const int FCNN_HIDDEN1_SIZE = 256;
const int FCNN_HIDDEN2_SIZE = 128;
const int FCNN_INPUT_SIZE = 784;

// CNN hyperparameters
const double CNN_LEARNING_RATE = 1e-3;     
const int CNN_EPOCHS = 15;                  
const int CNN_BATCH_SIZE = 64;

// Logistic regression hyperparameters
const double LR_LEARNING_RATE = 1e-3;    
const double LR_REGULARIZATION_L2 = 0.0001;  
const int LR_EPOCHS = 5;                  
const int LR_BATCH_SIZE = 64;               

// Random forest hyperparameters
const int RF_EPOCHS = 5;             
const int RF_BATCH_SIZE = 64;                
const int RF_NUM_TREES = 50;            
const int RF_MAX_DEPTH = 8;             
const int RF_MIN_SAMPLES_SPLIT = 8;     
const int RF_FEATURES_PER_SPLIT = 28;       

// KNN hyperparameters
const int KNN_EPOCHS = 1;                    
const int KNN_BATCH_SIZE = 64;             
const int KNN_K = 5;              
const int KNN_MAX_PROTOTYPES = 3000;   

// ==================== Dataset Loading ====================

namespace {
constexpr int kNumClasses = 10;
}

Metrics buildMetricsSnapshot(IClassificationModel& model,
                             const std::vector<std::vector<double>>& images,
                             const std::vector<int>& labels);

// Read a 32-bit big-endian integer from an MNIST file.
int readBigEndianInt(std::ifstream& file) {
    uint32_t val;
    file.read(reinterpret_cast<char*>(&val), 4);
    return (val >> 24) | ((val >> 8) & 0xFF00) | ((val << 8) & 0xFF0000) | (val << 24);
}

// Read MNIST image data.
std::vector<std::vector<double>> readImages(const std::string& filename, int& numImages) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open image file: " + filename);
    }
    int magic = readBigEndianInt(file);
    numImages = readBigEndianInt(file);
    int rows = readBigEndianInt(file);
    int cols = readBigEndianInt(file);
    int imageSize = rows * cols;
    std::vector<std::vector<double>> images(numImages, std::vector<double>(imageSize));
    for (int i = 0; i < numImages; ++i) {
        std::vector<unsigned char> buffer(imageSize);
        file.read(reinterpret_cast<char*>(buffer.data()), imageSize);
        for (int j = 0; j < imageSize; ++j) {
            images[i][j] = buffer[j] / 255.0;
        }
    }
    std::cout << "Loaded " << numImages << " images" << std::endl;
    return images;
}

// Read MNIST labels.
std::vector<int> readLabels(const std::string& filename, int& numLabels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open label file: " + filename);
    }
    int magic = readBigEndianInt(file);
    numLabels = readBigEndianInt(file);
    std::vector<int> labels(numLabels);
    for (int i = 0; i < numLabels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels[i] = label;
    }
    std::cout << "Loaded " << numLabels << " labels" << std::endl;
    return labels;
}

// Write ROC data and compute binary AUC for one-vs-rest evaluation.
double writeRocDataAndComputeAUC(const std::vector<double>& scores,
                                 const std::vector<int>& binaryLabels,
                                 const std::string& rocFilename) {
    namespace fs = std::filesystem;
    fs::path outputPath;
    const bool shouldWriteFile = !rocFilename.empty();
    if (shouldWriteFile) {
        outputPath = fs::path(rocFilename);
        if (!outputPath.has_parent_path() || !fs::exists(outputPath.parent_path())) {
            fs::path parentCandidate = fs::path("..") / outputPath.parent_path();
            if (outputPath.has_parent_path() && fs::exists(parentCandidate)) {
                outputPath = fs::path("..") / outputPath;
            }
        }
        if (outputPath.has_parent_path()) {
            std::error_code ec;
            fs::create_directories(outputPath.parent_path(), ec);
        }
    }

    std::vector<std::pair<double, int>> rankedSamples;
    rankedSamples.reserve(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        rankedSamples.emplace_back(scores[i], binaryLabels[i]);
    }

    std::sort(rankedSamples.begin(), rankedSamples.end(),
              [](const auto& left, const auto& right) {
                  if (left.first != right.first) {
                      return left.first > right.first;
                  }
                  return left.second > right.second;
              });

    double positiveCount = 0.0;
    double negativeCount = 0.0;
    for (int label : binaryLabels) {
        if (label == 1) {
            positiveCount += 1.0;
        } else {
            negativeCount += 1.0;
        }
    }

    if (positiveCount <= 0.0 || negativeCount <= 0.0) {
        return 0.0;
    }

    if (shouldWriteFile) {
        std::ofstream file(outputPath);
        if (file) {
            for (size_t i = 0; i < scores.size(); ++i) {
                file << scores[i] << " " << binaryLabels[i] << "\n";
            }
            file.close();
        } else {
            std::cerr << "Failed to write ROC data file: " << outputPath.string() << std::endl;
        }
    }

    double truePositive = 0.0;
    double falsePositive = 0.0;
    double previousTruePositiveRate = 0.0;
    double previousFalsePositiveRate = 0.0;
    double auc = 0.0;

    for (size_t i = 0; i < rankedSamples.size(); ) {
        size_t j = i;
        double score = rankedSamples[i].first;
        double groupPositives = 0.0;
        double groupNegatives = 0.0;

        while (j < rankedSamples.size() && rankedSamples[j].first == score) {
            if (rankedSamples[j].second == 1) {
                groupPositives += 1.0;
            } else {
                groupNegatives += 1.0;
            }
            ++j;
        }

        truePositive += groupPositives;
        falsePositive += groupNegatives;

        double truePositiveRate = truePositive / positiveCount;
        double falsePositiveRate = falsePositive / negativeCount;
        auc += (falsePositiveRate - previousFalsePositiveRate) *
               (truePositiveRate + previousTruePositiveRate) * 0.5;

        previousTruePositiveRate = truePositiveRate;
        previousFalsePositiveRate = falsePositiveRate;
        i = j;
    }

    return auc;
}

double computeAccuracyFromPredictions(IClassificationModel& model,
                                      const std::vector<std::vector<double>>& images,
                                      const std::vector<int>& labels) {
    if (images.empty() || labels.empty() || images.size() != labels.size()) {
        return 0.0;
    }

    int correct = 0;
    for (size_t i = 0; i < images.size(); ++i) {
        if (model.predict(images[i]) == labels[i]) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(images.size());
}

std::vector<std::vector<int>> computeConfusionMatrixFromPredictions(
    IClassificationModel& model,
    const std::vector<std::vector<double>>& images,
    const std::vector<int>& labels) {
    std::vector<std::vector<int>> matrix(kNumClasses, std::vector<int>(kNumClasses, 0));
    const size_t sampleCount = std::min(images.size(), labels.size());
    for (size_t i = 0; i < sampleCount; ++i) {
        const int truth = labels[i];
        const int predicted = model.predict(images[i]);
        if (truth >= 0 && truth < kNumClasses && predicted >= 0 && predicted < kNumClasses) {
            matrix[truth][predicted]++;
        }
    }
    return matrix;
}

void computePerClassMetricsFromConfusionMatrix(const std::vector<std::vector<int>>& matrix,
                                               std::vector<double>& precision,
                                               std::vector<double>& recall,
                                               std::vector<double>& f1) {
    const size_t classCount = matrix.size();
    precision.assign(classCount, 0.0);
    recall.assign(classCount, 0.0);
    f1.assign(classCount, 0.0);

    for (size_t c = 0; c < classCount; ++c) {
        const int tp = matrix[c][c];
        int fp = 0;
        int fn = 0;
        for (size_t i = 0; i < classCount; ++i) {
            if (i == c) {
                continue;
            }
            fp += matrix[i][c];
            fn += matrix[c][i];
        }

        precision[c] = (tp + fp > 0) ? static_cast<double>(tp) / static_cast<double>(tp + fp) : 0.0;
        recall[c] = (tp + fn > 0) ? static_cast<double>(tp) / static_cast<double>(tp + fn) : 0.0;
        f1[c] = (precision[c] + recall[c] > 0.0)
                    ? 2.0 * precision[c] * recall[c] / (precision[c] + recall[c])
                    : 0.0;
    }
}

void populateMetricSummary(Metrics& metrics) {
    metrics.precision = 0.0;
    metrics.recall = 0.0;
    metrics.f1 = 0.0;
    metrics.microPrecision = 0.0;
    metrics.microRecall = 0.0;
    metrics.microF1 = 0.0;

    const size_t classCount = metrics.perClassPrecision.size();
    if (classCount > 0) {
        for (size_t i = 0; i < classCount; ++i) {
            metrics.precision += metrics.perClassPrecision[i];
            metrics.recall += metrics.perClassRecall[i];
            metrics.f1 += metrics.perClassF1[i];
        }
        const double count = static_cast<double>(classCount);
        metrics.precision /= count;
        metrics.recall /= count;
        metrics.f1 /= count;
    }

    int totalTp = 0;
    int totalFp = 0;
    int totalFn = 0;
    for (size_t c = 0; c < metrics.confusionMatrix.size(); ++c) {
        totalTp += metrics.confusionMatrix[c][c];
        for (size_t i = 0; i < metrics.confusionMatrix.size(); ++i) {
            if (i == c) {
                continue;
            }
            totalFp += metrics.confusionMatrix[i][c];
            totalFn += metrics.confusionMatrix[c][i];
        }
    }

    metrics.microPrecision = (totalTp + totalFp > 0)
                                 ? static_cast<double>(totalTp) / static_cast<double>(totalTp + totalFp)
                                 : 0.0;
    metrics.microRecall = (totalTp + totalFn > 0)
                              ? static_cast<double>(totalTp) / static_cast<double>(totalTp + totalFn)
                              : 0.0;
    metrics.microF1 = (metrics.microPrecision + metrics.microRecall > 0.0)
                          ? 2.0 * metrics.microPrecision * metrics.microRecall /
                                (metrics.microPrecision + metrics.microRecall)
                          : 0.0;
}

double computeMacroAUCFromProbabilities(IClassificationModel& model,
                                        const std::vector<std::vector<double>>& images,
                                        const std::vector<int>& labels,
                                        std::string& rocFilename) {
    rocFilename = model.getROCDataFilename();

    double aucSum = 0.0;
    int aucCount = 0;
    std::vector<double> classZeroScores;
    std::vector<int> classZeroLabels;
    classZeroScores.reserve(images.size());
    classZeroLabels.reserve(images.size());

    for (int targetClass = 0; targetClass < kNumClasses; ++targetClass) {
        std::vector<double> scores;
        std::vector<int> binaryLabels;
        scores.reserve(images.size());
        binaryLabels.reserve(images.size());

        for (size_t i = 0; i < images.size(); ++i) {
            const auto probabilities = model.predictProba(images[i]);
            const double score = targetClass < static_cast<int>(probabilities.size())
                                     ? probabilities[targetClass]
                                     : 0.0;
            scores.push_back(score);
            binaryLabels.push_back(labels[i] == targetClass ? 1 : 0);

            if (targetClass == 0) {
                classZeroScores.push_back(score);
                classZeroLabels.push_back(labels[i] == 0 ? 1 : 0);
            }
        }

        double positives = 0.0;
        double negatives = 0.0;
        for (int binaryLabel : binaryLabels) {
            if (binaryLabel == 1) {
                positives += 1.0;
            } else {
                negatives += 1.0;
            }
        }

        if (positives > 0.0 && negatives > 0.0) {
            aucSum += writeRocDataAndComputeAUC(scores, binaryLabels, "");
            ++aucCount;
        }
    }

    writeRocDataAndComputeAUC(classZeroScores, classZeroLabels, rocFilename);
    return aucCount > 0 ? aucSum / static_cast<double>(aucCount) : 0.0;
}

void printEvaluationReport(const std::string& title,
                           IClassificationModel& model,
                           const std::vector<std::vector<double>>& images,
                           const std::vector<int>& labels) {
    Metrics metrics = buildMetricsSnapshot(model, images, labels);
    std::cout << "\n======== " << title << " Test Results ========" << std::endl;
    std::cout << "Accuracy:  " << std::fixed << std::setprecision(4) << metrics.accuracy * 100 << "%" << std::endl;
    std::cout << "Precision: " << std::setprecision(4) << metrics.precision * 100 << "%" << std::endl;
    std::cout << "Recall:    " << std::setprecision(4) << metrics.recall * 100 << "%" << std::endl;
    std::cout << "F1 Score:  " << std::setprecision(4) << metrics.f1 * 100 << "%" << std::endl;
    std::cout << "AUC:       " << std::setprecision(4) << metrics.auc << std::endl;
    std::cout << "========================================" << std::endl;
}

bool writeCnnDashboardData(IClassificationModel& model,
                           const std::vector<std::vector<double>>& images,
                           const std::vector<int>& labels,
                           const std::vector<std::vector<int>>& confusionMatrix,
                           const std::vector<double>& precision,
                           const std::vector<double>& recall,
                           const std::vector<double>& f1,
                           double accuracy,
                           double auc,
                           const std::string& outputPath) {
    std::ofstream file(outputPath);
    if (!file.is_open()) {
        std::cerr << "Failed to write CNN visualization data file: " << outputPath << std::endl;
        return false;
    }

    auto writeDoubleArray = [&file](const std::vector<double>& values) {
        file << "[";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i > 0) file << ",";
            file << std::fixed << std::setprecision(6) << values[i];
        }
        file << "]";
    };

    const size_t sampleCount = std::min<size_t>(100, images.size());

    file << "window.CNN_DASHBOARD_DATA = {\n";
    file << "  \"meta\": {\n";
    file << "    \"model\": \"CNN\",\n";
    file << "    \"sampleCount\": " << sampleCount << ",\n";
    file << "    \"sourceRoc\": \"../ROC/cnn_roc_data.txt\"\n";
    file << "  },\n";

    file << "  \"metrics\": {\n";
    file << "    \"accuracy\": " << std::fixed << std::setprecision(6) << accuracy << ",\n";
    file << "    \"auc\": " << std::fixed << std::setprecision(6) << auc << "\n";
    file << "  },\n";

    file << "  \"classMetrics\": [\n";
    for (int c = 0; c < 10; ++c) {
        file << "    {\"class\": " << c
             << ", \"precision\": " << std::fixed << std::setprecision(6) << precision[c]
             << ", \"recall\": " << std::fixed << std::setprecision(6) << recall[c]
             << ", \"f1\": " << std::fixed << std::setprecision(6) << f1[c] << "}";
        if (c < 9) {
            file << ",";
        }
        file << "\n";
    }
    file << "  ],\n";

    file << "  \"confusionMatrix\": [\n";
    for (size_t r = 0; r < confusionMatrix.size(); ++r) {
        file << "    [";
        for (size_t c = 0; c < confusionMatrix[r].size(); ++c) {
            if (c > 0) file << ",";
            file << confusionMatrix[r][c];
        }
        file << "]";
        if (r + 1 < confusionMatrix.size()) {
            file << ",";
        }
        file << "\n";
    }
    file << "  ],\n";

    file << "  \"samples\": [\n";
    for (size_t i = 0; i < sampleCount; ++i) {
        auto probabilities = model.predictProba(images[i]);
        int predicted = static_cast<int>(std::distance(probabilities.begin(),
                         std::max_element(probabilities.begin(), probabilities.end())));
        double confidence = probabilities[predicted];

        file << "    {\n";
        file << "      \"index\": " << i << ",\n";
        file << "      \"trueLabel\": " << labels[i] << ",\n";
        file << "      \"predLabel\": " << predicted << ",\n";
        file << "      \"confidence\": " << std::fixed << std::setprecision(6) << confidence << ",\n";
        file << "      \"probabilities\": ";
        writeDoubleArray(probabilities);
        file << ",\n";

        file << "      \"pixels\": [";
        for (size_t p = 0; p < images[i].size(); ++p) {
            if (p > 0) file << ",";
            double normalized = images[i][p];
            if (normalized < 0.0) normalized = 0.0;
            if (normalized > 1.0) normalized = 1.0;
            int value = static_cast<int>(std::round(normalized * 255.0));
            file << value;
        }
        file << "]\n";
        file << "    }";
        if (i + 1 < sampleCount) {
            file << ",";
        }
        file << "\n";
    }
    file << "  ]\n";
    file << "};\n";

    std::cout << "CNN visualization data exported: " << outputPath << std::endl;
    return true;
}

Metrics buildMetricsSnapshot(IClassificationModel& model,
                             const std::vector<std::vector<double>>& images,
                             const std::vector<int>& labels) {
    Metrics metrics{};
    metrics.accuracy = model.evaluate(images, labels);
    metrics.confusionMatrix = model.computeConfusionMatrix(images, labels);
    model.computeMetrics(metrics.confusionMatrix,
                         metrics.perClassPrecision,
                         metrics.perClassRecall,
                         metrics.perClassF1);
    populateMetricSummary(metrics);

    std::string rocFilename;
    metrics.auc = model.computeAUC(images, labels, rocFilename);
    return metrics;
}

// ==================== Progress Display ====================

class ProgressBar {
public:
    static void show(size_t current, size_t total, const std::string& label = "") {
        int barWidth = 40;
        double progress = static_cast<double>(current) / total;
        int filledWidth = static_cast<int>(barWidth * progress);
        
        std::cout << "\r  " << label << " [";
        for (int i = 0; i < barWidth; ++i) {
            if (i < filledWidth) std::cout << "#";
            else std::cout << "-";
        }
        std::cout << "] " << static_cast<int>(progress * 100) << "% (" 
                  << current << "/" << total << ")" << std::flush;
    }
    
    static void finish() {
        std::cout << std::endl;
    }
};

static double estimateRemainingSeconds(
    const std::chrono::steady_clock::time_point& started,
    int completed,
    int total) {
    if (completed <= 0 || total <= 0 || completed >= total) {
        return completed >= total ? 0.0 : -1.0;
    }
    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now - started).count();
    const double rate = elapsed / static_cast<double>(completed);
    return rate * static_cast<double>(total - completed);
}

// ==================== Forward Declarations ====================

// Model class declarations. Factory implementations appear later in the file.
class SVMModel;
class FCNNModel;
class CNNModel;
class RandomForestModel;
class KNNModel;

// ==================== SVM Model ====================

class SVMModel : public IClassificationModel {
public:
    virtual std::string getStructureDescription() const override { return "Input(784) -> Linear One-vs-Rest Heads(10)"; }
    virtual Metrics testAndGetMetrics(const std::vector<std::vector<double>>& images, const std::vector<int>& labels) override { return buildMetricsSnapshot(*this, images, labels); }

private:
    // SVM parameters
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    static const int NUM_CLASSES = 10;
    static const int INPUT_SIZE = 784;

public:
    std::string getName() const override {
        return "Support Vector Machine (SVM)";
    }

    std::string getDescription() const override {
        return "Linear SVM with one-vs-rest training, hinge loss and SGD optimization";
    }

    void initWeights() override {
        weights.resize(NUM_CLASSES, std::vector<double>(INPUT_SIZE, 0.0));
        biases.assign(NUM_CLASSES, 0.0);
        std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
        std::normal_distribution<double> dist(0.0, 0.01);
        for (auto& w_row : weights) {
            for (auto& w : w_row) {
                w = dist(rng);
            }
        }
        std::cout << "SVM weights initialized" << std::endl;
    }

    bool loadParams(const std::string& filename) override {
        std::ifstream file(filename);
        if (!file) {
            std::cerr << "Failed to open parameter file " << filename << std::endl;
            return false;
        }
        int classes, inputs;
        file >> classes >> inputs;
        if (classes != NUM_CLASSES || inputs != INPUT_SIZE) {
            std::cerr << "Parameter file shape mismatch" << std::endl;
            return false;
        }
        weights.resize(NUM_CLASSES, std::vector<double>(INPUT_SIZE));
        for (int c = 0; c < NUM_CLASSES; ++c) {
            for (int i = 0; i < INPUT_SIZE; ++i) {
                file >> weights[c][i];
            }
        }
        biases.resize(NUM_CLASSES);
        for (int c = 0; c < NUM_CLASSES; ++c) {
            file >> biases[c];
        }
        std::cout << "Parameters loaded from " << filename << std::endl;
        return true;
    }

    void saveParams(const std::string& filename) override {
        std::ofstream file(filename);
        file << NUM_CLASSES << " " << INPUT_SIZE << std::endl;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            for (int i = 0; i < INPUT_SIZE; ++i) {
                file << weights[c][i] << " ";
            }
            file << std::endl;
        }
        for (int c = 0; c < NUM_CLASSES; ++c) {
            file << biases[c] << " ";
        }
        std::cout << "Parameters saved to " << filename << std::endl;
    }

    void train(const std::vector<std::vector<double>>& images,
              const std::vector<int>& labels, int epochs) override {
        std::cout << "\nSVM training started (" << epochs << " epochs, batch_size=" << SVM_BATCH_SIZE << ")..." << std::endl;
        
        std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
        const int batchSize = SVM_BATCH_SIZE;
        const auto started = std::chrono::steady_clock::now();
        const int totalUnits = std::max(1, epochs * static_cast<int>(images.size()));
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double totalLoss = 0.0;
            int correctPredictions = 0;
            
            // Process one mini-batch.
            for (size_t batchStart = 0; batchStart < images.size(); batchStart += batchSize) {
                size_t batchEnd = std::min(batchStart + batchSize, images.size());
                
                // Accumulate gradients for this mini-batch.
                std::vector<std::vector<double>> batchWeightGradients(NUM_CLASSES, 
                    std::vector<double>(INPUT_SIZE, 0.0));
                std::vector<double> batchBiasGradients(NUM_CLASSES, 0.0);
                double batchLoss = 0.0;
                int batchCorrect = 0;
                
                // Process each sample in the batch.
                for (size_t idx = batchStart; idx < batchEnd; ++idx) {
                    const auto& image = images[idx];
                    int label = labels[idx];
                    
                    // Forward pass: compute class scores.
                    std::vector<double> scores(NUM_CLASSES);
                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        scores[c] = biases[c];
                        for (int i = 0; i < INPUT_SIZE; ++i) {
                            scores[c] += weights[c][i] * image[i];
                        }
                    }
                    
                    // Track training accuracy.
                    int predictedClass = 0;
                    for (int c = 1; c < NUM_CLASSES; ++c) {
                        if (scores[c] > scores[predictedClass]) predictedClass = c;
                    }
                    if (predictedClass == label) {
                        batchCorrect++;
                        correctPredictions++;
                    }
                    
                    // One-vs-rest hinge loss.
                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        int y = (c == label) ? 1 : -1;
                        double margin = y * scores[c];
                        
                        // Standard hinge loss term.
                        double loss = std::max(0.0, 1.0 - margin);
                        totalLoss += loss;
                        
                        // Gradient includes hinge term and L2 regularization.
                        double gradient = 0.0;
                        if (margin < 1.0) {
                            gradient = -y;
                        }
                        
                        // Accumulate weight gradients.
                        for (int i = 0; i < INPUT_SIZE; ++i) {
                            batchWeightGradients[c][i] += (gradient * image[i] + 
                                SVM_REGULARIZATION_C * weights[c][i]);
                        }
                        
                        // Accumulate bias gradients.
                        batchBiasGradients[c] += gradient;
                    }
                }
                
                // Apply one SGD update after the batch.
                for (int c = 0; c < NUM_CLASSES; ++c) {
                    for (int i = 0; i < INPUT_SIZE; ++i) {
                        weights[c][i] -= SVM_LEARNING_RATE * batchWeightGradients[c][i] / 
                            static_cast<double>(batchEnd - batchStart);
                    }
                    biases[c] -= SVM_LEARNING_RATE * batchBiasGradients[c] / 
                        static_cast<double>(batchEnd - batchStart);
                }
                
                // Report training progress.
                ProgressBar::show(batchEnd, images.size(), "Processing data");
                const int completedUnits = epoch * static_cast<int>(images.size()) + static_cast<int>(batchEnd);
                reportProgress(completedUnits, totalUnits,
                               static_cast<double>(completedUnits) / totalUnits,
                               estimateRemainingSeconds(started, completedUnits, totalUnits));
            }
            
            ProgressBar::finish();
            double accuracy = static_cast<double>(correctPredictions) / images.size();
            std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                      << " - avg loss: " << std::fixed << std::setprecision(6) << totalLoss / images.size()
                      << " - train acc: " << std::setprecision(4)
                      << accuracy * 100 << "%"
                      << " - remaining epochs: " << (epochs - epoch - 1) << std::endl;
        }
        
        std::cout << "SVM training completed" << std::endl;
    }

    int predict(const std::vector<double>& image) override {
        int bestClass = 0;
        double bestScore = biases[0];
        for (int i = 0; i < INPUT_SIZE; ++i) {
            bestScore += weights[0][i] * image[i];
        }
        for (int c = 1; c < NUM_CLASSES; ++c) {
            double score = biases[c];
            for (int i = 0; i < INPUT_SIZE; ++i) {
                score += weights[c][i] * image[i];
            }
            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }
        return bestClass;
    }

    std::vector<double> predictProba(const std::vector<double>& image) override {
        std::vector<double> scores(NUM_CLASSES);
        for (int c = 0; c < NUM_CLASSES; ++c) {
            scores[c] = biases[c];
            for (int i = 0; i < INPUT_SIZE; ++i) {
                scores[c] += weights[c][i] * image[i];
            }
        }
        // Simple softmax normalization.
        double maxScore = *std::max_element(scores.begin(), scores.end());
        double sum = 0.0;
        for (auto& s : scores) {
            s = std::exp(s - maxScore);
            sum += s;
        }
        for (auto& s : scores) s /= sum;
        return scores;
    }

    double evaluate(const std::vector<std::vector<double>>& images,
                   const std::vector<int>& labels) override {
        return computeAccuracyFromPredictions(*this, images, labels);
    }

    std::vector<std::vector<int>> computeConfusionMatrix(
        const std::vector<std::vector<double>>& images,
        const std::vector<int>& labels) override {
        return computeConfusionMatrixFromPredictions(*this, images, labels);
    }

    void computeMetrics(const std::vector<std::vector<int>>& matrix,
                       std::vector<double>& precision,
                       std::vector<double>& recall,
                       std::vector<double>& f1) override {
        computePerClassMetricsFromConfusionMatrix(matrix, precision, recall, f1);
    }

    double computeAUC(const std::vector<std::vector<double>>& images,
                     const std::vector<int>& labels,
                     std::string& rocFilename) override {
        return computeMacroAUCFromProbabilities(*this, images, labels, rocFilename);
    }

    std::string getROCDataFilename() const override {
        return "ROC/svm_roc_data.txt";
    }

    void evaluateAndReport(const std::vector<std::vector<double>>& images,
                          const std::vector<int>& labels) override {
        printEvaluationReport("SVM", *this, images, labels);
    }
};

// ==================== FCNN Model ====================

class FCNNModel : public IClassificationModel {
public:
    virtual std::string getStructureDescription() const override { return "Input(784) -> Dense(256, ReLU) -> Dense(128, ReLU) -> Softmax(10)"; }
    virtual Metrics testAndGetMetrics(const std::vector<std::vector<double>>& images, const std::vector<int>& labels) override { return buildMetricsSnapshot(*this, images, labels); }
    std::vector<std::vector<double>> getWeights1() const { return W1; }
    std::vector<std::vector<double>> getWeights2() const { return W2; }

private:
    static const int OUTPUT_SIZE = 10;
    static const int INPUT_SIZE = 784;
    
    // Dense layer weights and biases.
    std::vector<std::vector<double>> W1, W2, W3;
    std::vector<double> b1, b2, b3;
    
public:
    std::string getName() const override {
        return "Full Connected Neural Network (FCNN)";
    }

    std::string getDescription() const override {
        return "Fully connected neural network with two hidden layers (256, 128), ReLU activations and a Softmax output";
    }

    void initWeights() override {
        // Weights are initialized lazily in train().
        std::cout << "FCNN weights will be initialized in train()" << std::endl;
    }

    bool loadParams(const std::string& filename) override {
        std::ifstream file(filename);
        if (!file) {
            std::cerr << "Failed to open parameter file " << filename << std::endl;
            return false;
        }
        
        W1.assign(FCNN_HIDDEN1_SIZE, std::vector<double>(INPUT_SIZE));
        W2.assign(FCNN_HIDDEN2_SIZE, std::vector<double>(FCNN_HIDDEN1_SIZE));
        W3.assign(OUTPUT_SIZE, std::vector<double>(FCNN_HIDDEN2_SIZE));
        b1.assign(FCNN_HIDDEN1_SIZE, 0.0);
        b2.assign(FCNN_HIDDEN2_SIZE, 0.0);
        b3.assign(OUTPUT_SIZE, 0.0);
        
        // Read W1.
        for (int i = 0; i < FCNN_HIDDEN1_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                file >> W1[i][j];
            }
        }
        // Read b1.
        for (int i = 0; i < FCNN_HIDDEN1_SIZE; ++i) {
            file >> b1[i];
        }
        // Read W2.
        for (int i = 0; i < FCNN_HIDDEN2_SIZE; ++i) {
            for (int j = 0; j < FCNN_HIDDEN1_SIZE; ++j) {
                file >> W2[i][j];
            }
        }
        // Read b2.
        for (int i = 0; i < FCNN_HIDDEN2_SIZE; ++i) {
            file >> b2[i];
        }
        // Read W3.
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            for (int j = 0; j < FCNN_HIDDEN2_SIZE; ++j) {
                file >> W3[i][j];
            }
        }
        // Read b3.
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            file >> b3[i];
        }
        
        std::cout << "Parameters loaded from " << filename << std::endl;
        return true;
    }

    void saveParams(const std::string& filename) override {
        std::ofstream file(filename);
        if (!file) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }
        
        // Write W1.
        for (int i = 0; i < FCNN_HIDDEN1_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                file << W1[i][j] << " ";
            }
            file << "\n";
        }
        // Write b1.
        for (int i = 0; i < FCNN_HIDDEN1_SIZE; ++i) {
            file << b1[i] << " ";
        }
        file << "\n";
        
        // Write W2.
        for (int i = 0; i < FCNN_HIDDEN2_SIZE; ++i) {
            for (int j = 0; j < FCNN_HIDDEN1_SIZE; ++j) {
                file << W2[i][j] << " ";
            }
            file << "\n";
        }
        // Write b2.
        for (int i = 0; i < FCNN_HIDDEN2_SIZE; ++i) {
            file << b2[i] << " ";
        }
        file << "\n";
        
        // Write W3.
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            for (int j = 0; j < FCNN_HIDDEN2_SIZE; ++j) {
                file << W3[i][j] << " ";
            }
            file << "\n";
        }
        // Write b3.
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            file << b3[i] << " ";
        }
        
        std::cout << "Parameters saved to " << filename << std::endl;
    }

    void train(const std::vector<std::vector<double>>& images,
              const std::vector<int>& labels, int epochs) override {
        std::cout << "\nFCNN training started (" << epochs << " epochs, batch_size=" << FCNN_BATCH_SIZE << ")..." << std::endl;
        std::cout << "  Network: [" << INPUT_SIZE << "] -> [" << FCNN_HIDDEN1_SIZE
                  << "] -> [" << FCNN_HIDDEN2_SIZE << "] -> [" << OUTPUT_SIZE << "]" << std::endl;
        const auto started = std::chrono::steady_clock::now();
        const int totalUnits = std::max(1, epochs * static_cast<int>(images.size()));
        
        // If no weights were loaded, start from a fresh initialization.
        bool isNewTraining = W1.empty();
        
        if (isNewTraining) {
            std::cout << "  Initialization: He init | Optimizer: SGD+Momentum(mu=" << FCNN_MOMENTUM << ")" << std::endl;
        } else {
            std::cout << "  Initialization: load existing parameters | Optimizer: SGD+Momentum(mu=" << FCNN_MOMENTUM << ")" << std::endl;
        }
        std::cout << "  Loss: cross entropy\n" << std::endl;
        
        // ==================== Weight Initialization ====================
        if (isNewTraining) {
            W1.assign(FCNN_HIDDEN1_SIZE, std::vector<double>(INPUT_SIZE));
            b1.assign(FCNN_HIDDEN1_SIZE, 0.0);
            W2.assign(FCNN_HIDDEN2_SIZE, std::vector<double>(FCNN_HIDDEN1_SIZE));
            b2.assign(FCNN_HIDDEN2_SIZE, 0.0);
            W3.assign(OUTPUT_SIZE, std::vector<double>(FCNN_HIDDEN2_SIZE));
            b3.assign(OUTPUT_SIZE, 0.0);
            
            std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
            
            // He initialization: std = sqrt(2 / fan_in).
            double std_W1 = std::sqrt(2.0 / INPUT_SIZE);
            double std_W2 = std::sqrt(2.0 / FCNN_HIDDEN1_SIZE);
            double std_W3 = std::sqrt(2.0 / FCNN_HIDDEN2_SIZE);
            
            std::normal_distribution<double> dist_W1(0.0, std_W1);
            std::normal_distribution<double> dist_W2(0.0, std_W2);
            std::normal_distribution<double> dist_W3(0.0, std_W3);
            
            for (auto& row : W1) for (auto& w : row) w = dist_W1(rng);
            for (auto& row : W2) for (auto& w : row) w = dist_W2(rng);
            for (auto& row : W3) for (auto& w : row) w = dist_W3(rng);
        }
        
        // ==================== Momentum Buffers ====================
        // dW is the current gradient, vW is the velocity term.
        std::vector<std::vector<double>> vW1(FCNN_HIDDEN1_SIZE, std::vector<double>(INPUT_SIZE, 0.0));
        std::vector<double> vb1(FCNN_HIDDEN1_SIZE, 0.0);
        std::vector<std::vector<double>> vW2(FCNN_HIDDEN2_SIZE, std::vector<double>(FCNN_HIDDEN1_SIZE, 0.0));
        std::vector<double> vb2(FCNN_HIDDEN2_SIZE, 0.0);
        std::vector<std::vector<double>> vW3(OUTPUT_SIZE, std::vector<double>(FCNN_HIDDEN2_SIZE, 0.0));
        std::vector<double> vb3(OUTPUT_SIZE, 0.0);
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double totalLoss = 0.0;
            int correctPredictions = 0;
            
            // Process one mini-batch.
            for (size_t batchStart = 0; batchStart < images.size(); batchStart += FCNN_BATCH_SIZE) {
                size_t batchEnd = std::min(batchStart + FCNN_BATCH_SIZE, images.size());
                int batchSize = batchEnd - batchStart;
                ProgressBar::show(batchEnd, images.size(), "Processing data");
                const int completedUnits = epoch * static_cast<int>(images.size()) + static_cast<int>(batchEnd);
                reportProgress(completedUnits, totalUnits,
                               static_cast<double>(completedUnits) / totalUnits,
                               estimateRemainingSeconds(started, completedUnits, totalUnits));
                
                // ==================== Gradient Accumulation ====================
                std::vector<std::vector<double>> dW1(FCNN_HIDDEN1_SIZE, std::vector<double>(INPUT_SIZE, 0.0));
                std::vector<double> db1(FCNN_HIDDEN1_SIZE, 0.0);
                std::vector<std::vector<double>> dW2(FCNN_HIDDEN2_SIZE, std::vector<double>(FCNN_HIDDEN1_SIZE, 0.0));
                std::vector<double> db2(FCNN_HIDDEN2_SIZE, 0.0);
                std::vector<std::vector<double>> dW3(OUTPUT_SIZE, std::vector<double>(FCNN_HIDDEN2_SIZE, 0.0));
                std::vector<double> db3(OUTPUT_SIZE, 0.0);
                
                for (size_t idx = batchStart; idx < batchEnd; ++idx) {
                    const auto& image = images[idx];
                    int label = labels[idx];
                    
                    // ==================== Forward Pass ====================
                    // Layer 1: dense + ReLU.
                    std::vector<double> z1(FCNN_HIDDEN1_SIZE, 0.0);
                    for (int h = 0; h < FCNN_HIDDEN1_SIZE; ++h) {
                        for (int i = 0; i < INPUT_SIZE; ++i) {
                            z1[h] += W1[h][i] * image[i];
                        }
                        z1[h] += b1[h];
                    }
                    
                    std::vector<double> hidden1(FCNN_HIDDEN1_SIZE);
                    for (int h = 0; h < FCNN_HIDDEN1_SIZE; ++h) {
                        hidden1[h] = z1[h] > 0 ? z1[h] : 0;
                    }
                    
                    // Layer 2: dense + ReLU.
                    std::vector<double> z2(FCNN_HIDDEN2_SIZE, 0.0);
                    for (int h = 0; h < FCNN_HIDDEN2_SIZE; ++h) {
                        for (int i = 0; i < FCNN_HIDDEN1_SIZE; ++i) {
                            z2[h] += W2[h][i] * hidden1[i];
                        }
                        z2[h] += b2[h];
                    }
                    
                    std::vector<double> hidden2(FCNN_HIDDEN2_SIZE);
                    for (int h = 0; h < FCNN_HIDDEN2_SIZE; ++h) {
                        hidden2[h] = z2[h] > 0 ? z2[h] : 0;
                    }
                    
                    // Layer 3: output logits.
                    std::vector<double> z3(OUTPUT_SIZE, 0.0);
                    for (int o = 0; o < OUTPUT_SIZE; ++o) {
                        for (int i = 0; i < FCNN_HIDDEN2_SIZE; ++i) {
                            z3[o] += W3[o][i] * hidden2[i];
                        }
                        z3[o] += b3[o];
                    }
                    
                    // Softmax normalization.
                    std::vector<double> output(OUTPUT_SIZE);
                    double maxZ3 = *std::max_element(z3.begin(), z3.end());
                    double sumExp = 0.0;
                    for (int o = 0; o < OUTPUT_SIZE; ++o) {
                        output[o] = std::exp(z3[o] - maxZ3);
                        sumExp += output[o];
                    }
                    for (int o = 0; o < OUTPUT_SIZE; ++o) {
                        output[o] /= sumExp;
                    }
                    
                    // ==================== Loss ====================
                    double loss = -std::log(std::max(output[label], 1e-7));
                    totalLoss += loss;
                    
                    // Track training accuracy.
                    int predictedClass = std::distance(output.begin(), 
                                                       std::max_element(output.begin(), output.end()));
                    if (predictedClass == label) correctPredictions++;
                    
                    // ==================== Backpropagation ====================
                    // Output-layer gradient: dL / dz3.
                    std::vector<double> dz3(OUTPUT_SIZE);
                    for (int o = 0; o < OUTPUT_SIZE; ++o) {
                        dz3[o] = output[o] - (o == label ? 1.0 : 0.0);  // Softmax - one_hot
                    }
                    
                    // Gradients for W3 and b3.
                    for (int o = 0; o < OUTPUT_SIZE; ++o) {
                        for (int h = 0; h < FCNN_HIDDEN2_SIZE; ++h) {
                            dW3[o][h] += dz3[o] * hidden2[h];
                        }
                        db3[o] += dz3[o];
                    }
                    
                    // Hidden layer 2 gradient: dL / dz2.
                    std::vector<double> dhidden2(FCNN_HIDDEN2_SIZE, 0.0);
                    for (int h = 0; h < FCNN_HIDDEN2_SIZE; ++h) {
                        for (int o = 0; o < OUTPUT_SIZE; ++o) {
                            dhidden2[h] += dz3[o] * W3[o][h];
                        }
                    }
                    
                    std::vector<double> dz2(FCNN_HIDDEN2_SIZE);
                    for (int h = 0; h < FCNN_HIDDEN2_SIZE; ++h) {
                        dz2[h] = dhidden2[h] * (z2[h] > 0 ? 1.0 : 0.0);
                    }
                    
                    // Gradients for W2 and b2.
                    for (int h = 0; h < FCNN_HIDDEN2_SIZE; ++h) {
                        for (int i = 0; i < FCNN_HIDDEN1_SIZE; ++i) {
                            dW2[h][i] += dz2[h] * hidden1[i];
                        }
                        db2[h] += dz2[h];
                    }
                    
                    // Hidden layer 1 gradient: dL / dz1.
                    std::vector<double> dhidden1(FCNN_HIDDEN1_SIZE, 0.0);
                    for (int h = 0; h < FCNN_HIDDEN1_SIZE; ++h) {
                        for (int m = 0; m < FCNN_HIDDEN2_SIZE; ++m) {
                            dhidden1[h] += dz2[m] * W2[m][h];
                        }
                    }
                    
                    std::vector<double> dz1(FCNN_HIDDEN1_SIZE);
                    for (int h = 0; h < FCNN_HIDDEN1_SIZE; ++h) {
                        dz1[h] = dhidden1[h] * (z1[h] > 0 ? 1.0 : 0.0);
                    }
                    
                    // Gradients for W1 and b1.
                    for (int h = 0; h < FCNN_HIDDEN1_SIZE; ++h) {
                        for (int i = 0; i < INPUT_SIZE; ++i) {
                            dW1[h][i] += dz1[h] * image[i];
                        }
                        db1[h] += dz1[h];
                    }
                }
                
                // ==================== Average Gradients ====================
                double learningRate = FCNN_LEARNING_RATE;
                for (int h = 0; h < FCNN_HIDDEN1_SIZE; ++h) {
                    for (int i = 0; i < INPUT_SIZE; ++i) {
                        dW1[h][i] /= batchSize;
                    }
                    db1[h] /= batchSize;
                }
                for (int h = 0; h < FCNN_HIDDEN2_SIZE; ++h) {
                    for (int i = 0; i < FCNN_HIDDEN1_SIZE; ++i) {
                        dW2[h][i] /= batchSize;
                    }
                    db2[h] /= batchSize;
                }
                for (int o = 0; o < OUTPUT_SIZE; ++o) {
                    for (int h = 0; h < FCNN_HIDDEN2_SIZE; ++h) {
                        dW3[o][h] /= batchSize;
                    }
                    db3[o] /= batchSize;
                }
                
                // ==================== Momentum Update ====================
                // v = mu * v + grad; w = w - lr * v
                
                for (int h = 0; h < FCNN_HIDDEN1_SIZE; ++h) {
                    for (int i = 0; i < INPUT_SIZE; ++i) {
                        vW1[h][i] = FCNN_MOMENTUM * vW1[h][i] + dW1[h][i];
                        W1[h][i] -= learningRate * vW1[h][i];
                    }
                    vb1[h] = FCNN_MOMENTUM * vb1[h] + db1[h];
                    b1[h] -= learningRate * vb1[h];
                }
                
                for (int h = 0; h < FCNN_HIDDEN2_SIZE; ++h) {
                    for (int i = 0; i < FCNN_HIDDEN1_SIZE; ++i) {
                        vW2[h][i] = FCNN_MOMENTUM * vW2[h][i] + dW2[h][i];
                        W2[h][i] -= learningRate * vW2[h][i];
                    }
                    vb2[h] = FCNN_MOMENTUM * vb2[h] + db2[h];
                    b2[h] -= learningRate * vb2[h];
                }
                
                for (int o = 0; o < OUTPUT_SIZE; ++o) {
                    for (int h = 0; h < FCNN_HIDDEN2_SIZE; ++h) {
                        vW3[o][h] = FCNN_MOMENTUM * vW3[o][h] + dW3[o][h];
                        W3[o][h] -= learningRate * vW3[o][h];
                    }
                    vb3[o] = FCNN_MOMENTUM * vb3[o] + db3[o];
                    b3[o] -= learningRate * vb3[o];
                }
            }
            
            ProgressBar::finish();
            double accuracy = static_cast<double>(correctPredictions) / images.size();
            std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                      << " - loss: " << std::fixed << std::setprecision(6) << totalLoss / images.size()
                      << " - accuracy: " << std::setprecision(4)
                      << accuracy * 100 << "%"
                      << " - remaining epochs: " << (epochs - epoch - 1) << std::endl;
        }
        
        std::cout << "FCNN training completed" << std::endl;
    }

    int predict(const std::vector<double>& image) override {
        // Return a default class if the model is not trained yet.
        if (W1.empty() || W2.empty() || W3.empty()) {
            return 0;
        }
        
        // Inference forward pass with trained weights.
        // Layer 1
        std::vector<double> hidden1(FCNN_HIDDEN1_SIZE, 0.0);
        for (int h = 0; h < FCNN_HIDDEN1_SIZE; ++h) {
            for (int i = 0; i < INPUT_SIZE; ++i) {
                hidden1[h] += W1[h][i] * image[i];
            }
            hidden1[h] += b1[h];
            if (hidden1[h] < 0) hidden1[h] = 0;  // ReLU
        }
        
        // Layer 2
        std::vector<double> hidden2(FCNN_HIDDEN2_SIZE, 0.0);
        for (int h = 0; h < FCNN_HIDDEN2_SIZE; ++h) {
            for (int i = 0; i < FCNN_HIDDEN1_SIZE; ++i) {
                hidden2[h] += W2[h][i] * hidden1[i];
            }
            hidden2[h] += b2[h];
            if (hidden2[h] < 0) hidden2[h] = 0;  // ReLU
        }
        
        // Output Layer
        std::vector<double> output(OUTPUT_SIZE, 0.0);
        for (int o = 0; o < OUTPUT_SIZE; ++o) {
            for (int i = 0; i < FCNN_HIDDEN2_SIZE; ++i) {
                output[o] += W3[o][i] * hidden2[i];
            }
            output[o] += b3[o];
        }
        
        return std::distance(output.begin(), 
                           std::max_element(output.begin(), output.end()));
    }

    std::vector<double> predictProba(const std::vector<double>& image) override {
        std::vector<double> output(OUTPUT_SIZE, 0.0);
        
        if (W1.empty() || W2.empty() || W3.empty()) {
            for (auto& o : output) o = 0.1;
            return output;
        }
        
        // 鍓嶅悜浼犳挱
        // Layer 1
        std::vector<double> hidden1(FCNN_HIDDEN1_SIZE, 0.0);
        for (int h = 0; h < FCNN_HIDDEN1_SIZE; ++h) {
            for (int i = 0; i < INPUT_SIZE; ++i) {
                hidden1[h] += W1[h][i] * image[i];
            }
            hidden1[h] += b1[h];
            if (hidden1[h] < 0) hidden1[h] = 0;  // ReLU
        }
        
        // Layer 2
        std::vector<double> hidden2(FCNN_HIDDEN2_SIZE, 0.0);
        for (int h = 0; h < FCNN_HIDDEN2_SIZE; ++h) {
            for (int i = 0; i < FCNN_HIDDEN1_SIZE; ++i) {
                hidden2[h] += W2[h][i] * hidden1[i];
            }
            hidden2[h] += b2[h];
            if (hidden2[h] < 0) hidden2[h] = 0;  // ReLU
        }
        
        // Output Layer
        for (int o = 0; o < OUTPUT_SIZE; ++o) {
            for (int i = 0; i < FCNN_HIDDEN2_SIZE; ++i) {
                output[o] += W3[o][i] * hidden2[i];
            }
            output[o] += b3[o];
        }
        
        // Softmax
        double maxOut = *std::max_element(output.begin(), output.end());
        double sumExp = 0.0;
        for (auto& o : output) {
            o = std::exp(o - maxOut);
            sumExp += o;
        }
        for (auto& o : output) o /= sumExp;
        
        return output;
    }

    double evaluate(const std::vector<std::vector<double>>& images,
                   const std::vector<int>& labels) override {
        return computeAccuracyFromPredictions(*this, images, labels);
    }

    std::vector<std::vector<int>> computeConfusionMatrix(
        const std::vector<std::vector<double>>& images,
        const std::vector<int>& labels) override {
        return computeConfusionMatrixFromPredictions(*this, images, labels);
    }

    void computeMetrics(const std::vector<std::vector<int>>& matrix,
                       std::vector<double>& precision,
                       std::vector<double>& recall,
                       std::vector<double>& f1) override {
        computePerClassMetricsFromConfusionMatrix(matrix, precision, recall, f1);
    }

    double computeAUC(const std::vector<std::vector<double>>& images,
                     const std::vector<int>& labels,
                     std::string& rocFilename) override {
        return computeMacroAUCFromProbabilities(*this, images, labels, rocFilename);
    }

    std::string getROCDataFilename() const override {
        return "ROC/fcnn_roc_data.txt";
    }

    void evaluateAndReport(const std::vector<std::vector<double>>& images,
                          const std::vector<int>& labels) override {
        printEvaluationReport("FCNN", *this, images, labels);
    }
};

// ==================== CNN Model ====================

class CNNModel : public IClassificationModel {
public:
    virtual std::string getStructureDescription() const override { return "Input(28x28) -> Conv(8,3x3) -> MaxPool -> Conv(16,3x3) -> MaxPool -> Dense(10) -> Softmax"; }
    virtual Metrics testAndGetMetrics(const std::vector<std::vector<double>>& images, const std::vector<int>& labels) override { return buildMetricsSnapshot(*this, images, labels); }
    std::vector<std::vector<std::vector<double>>> getConvKernels() const { return conv1Kernels; }
    std::vector<std::vector<double>> getFCWeights() const { return fcWeights; }

private:
    static const int OUTPUT_SIZE = 10;
    static const int INPUT_HEIGHT = 28;
    static const int INPUT_WIDTH = 28;
    static const int CONV1_CHANNELS = 8;
    static const int CONV2_CHANNELS = 16;
    static const int KERNEL_SIZE = 3;
    static const int POOL_SIZE = 2;
    static const int CONV1_HEIGHT = INPUT_HEIGHT - KERNEL_SIZE + 1;
    static const int CONV1_WIDTH = INPUT_WIDTH - KERNEL_SIZE + 1;
    static const int POOL1_HEIGHT = CONV1_HEIGHT / POOL_SIZE;
    static const int POOL1_WIDTH = CONV1_WIDTH / POOL_SIZE;
    static const int CONV2_HEIGHT = POOL1_HEIGHT - KERNEL_SIZE + 1;
    static const int CONV2_WIDTH = POOL1_WIDTH - KERNEL_SIZE + 1;
    static const int POOL2_HEIGHT = CONV2_HEIGHT / POOL_SIZE;
    static const int POOL2_WIDTH = CONV2_WIDTH / POOL_SIZE;
    static const int FLATTEN_SIZE = CONV2_CHANNELS * POOL2_HEIGHT * POOL2_WIDTH;

    using Matrix = std::vector<std::vector<double>>;
    using Tensor3 = std::vector<std::vector<std::vector<double>>>;
    using Tensor4 = std::vector<std::vector<std::vector<std::vector<double>>>>;
    using Mask3 = std::vector<std::vector<std::vector<int>>>;

    Tensor3 conv1Kernels;
    std::vector<double> conv1Biases;
    Tensor4 conv2Kernels;
    std::vector<double> conv2Biases;
    std::vector<std::vector<double>> fcWeights;
    std::vector<double> fcBiases;

    struct ForwardCache {
        Matrix input;
        Tensor3 conv1Pre;
        Tensor3 conv1Act;
        Tensor3 pool1;
        Mask3 pool1Mask;
        Tensor3 conv2Pre;
        Tensor3 conv2Act;
        Tensor3 pool2;
        Mask3 pool2Mask;
        std::vector<double> flattened;
        std::vector<double> logits;
        std::vector<double> probabilities;
    };

    Matrix reshapeInput(const std::vector<double>& image) const {
        Matrix reshaped(INPUT_HEIGHT, std::vector<double>(INPUT_WIDTH, 0.0));
        for (int r = 0; r < INPUT_HEIGHT; ++r) {
            for (int c = 0; c < INPUT_WIDTH; ++c) {
                reshaped[r][c] = image[r * INPUT_WIDTH + c];
            }
        }
        return reshaped;
    }

    void initializeParameters() {
        std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
        std::normal_distribution<double> conv1Dist(0.0, std::sqrt(2.0 / (KERNEL_SIZE * KERNEL_SIZE)));
        std::normal_distribution<double> conv2Dist(0.0, std::sqrt(2.0 / (CONV1_CHANNELS * KERNEL_SIZE * KERNEL_SIZE)));
        std::normal_distribution<double> fcDist(0.0, std::sqrt(2.0 / FLATTEN_SIZE));

        conv1Kernels.assign(CONV1_CHANNELS,
            Matrix(KERNEL_SIZE, std::vector<double>(KERNEL_SIZE, 0.0)));
        conv1Biases.assign(CONV1_CHANNELS, 0.0);
        for (int oc = 0; oc < CONV1_CHANNELS; ++oc) {
            for (int kr = 0; kr < KERNEL_SIZE; ++kr) {
                for (int kc = 0; kc < KERNEL_SIZE; ++kc) {
                    conv1Kernels[oc][kr][kc] = conv1Dist(rng);
                }
            }
        }

        conv2Kernels.assign(CONV2_CHANNELS,
            Tensor3(CONV1_CHANNELS, Matrix(KERNEL_SIZE, std::vector<double>(KERNEL_SIZE, 0.0))));
        conv2Biases.assign(CONV2_CHANNELS, 0.0);
        for (int oc = 0; oc < CONV2_CHANNELS; ++oc) {
            for (int ic = 0; ic < CONV1_CHANNELS; ++ic) {
                for (int kr = 0; kr < KERNEL_SIZE; ++kr) {
                    for (int kc = 0; kc < KERNEL_SIZE; ++kc) {
                        conv2Kernels[oc][ic][kr][kc] = conv2Dist(rng);
                    }
                }
            }
        }

        fcWeights.assign(OUTPUT_SIZE, std::vector<double>(FLATTEN_SIZE, 0.0));
        fcBiases.assign(OUTPUT_SIZE, 0.0);
        for (int o = 0; o < OUTPUT_SIZE; ++o) {
            for (int i = 0; i < FLATTEN_SIZE; ++i) {
                fcWeights[o][i] = fcDist(rng);
            }
        }
    }

    bool isInitialized() const {
        return !conv1Kernels.empty() && !conv2Kernels.empty() &&
               !fcWeights.empty() && !fcWeights[0].empty();
    }

    Tensor3 convForwardSingleChannel(const Matrix& input,
                                     const Tensor3& kernels,
                                     const std::vector<double>& biases) const {
        const int outChannels = static_cast<int>(kernels.size());
        const int outHeight = static_cast<int>(input.size()) - KERNEL_SIZE + 1;
        const int outWidth = static_cast<int>(input[0].size()) - KERNEL_SIZE + 1;
        Tensor3 output(outChannels, Matrix(outHeight, std::vector<double>(outWidth, 0.0)));

        for (int oc = 0; oc < outChannels; ++oc) {
            for (int r = 0; r < outHeight; ++r) {
                for (int c = 0; c < outWidth; ++c) {
                    double sum = biases[oc];
                    for (int kr = 0; kr < KERNEL_SIZE; ++kr) {
                        for (int kc = 0; kc < KERNEL_SIZE; ++kc) {
                            sum += input[r + kr][c + kc] * kernels[oc][kr][kc];
                        }
                    }
                    output[oc][r][c] = sum;
                }
            }
        }
        return output;
    }

    Tensor3 convForwardMultiChannel(const Tensor3& input,
                                    const Tensor4& kernels,
                                    const std::vector<double>& biases) const {
        const int outChannels = static_cast<int>(kernels.size());
        const int inChannels = static_cast<int>(input.size());
        const int outHeight = static_cast<int>(input[0].size()) - KERNEL_SIZE + 1;
        const int outWidth = static_cast<int>(input[0][0].size()) - KERNEL_SIZE + 1;
        Tensor3 output(outChannels, Matrix(outHeight, std::vector<double>(outWidth, 0.0)));

        for (int oc = 0; oc < outChannels; ++oc) {
            for (int r = 0; r < outHeight; ++r) {
                for (int c = 0; c < outWidth; ++c) {
                    double sum = biases[oc];
                    for (int ic = 0; ic < inChannels; ++ic) {
                        for (int kr = 0; kr < KERNEL_SIZE; ++kr) {
                            for (int kc = 0; kc < KERNEL_SIZE; ++kc) {
                                sum += input[ic][r + kr][c + kc] * kernels[oc][ic][kr][kc];
                            }
                        }
                    }
                    output[oc][r][c] = sum;
                }
            }
        }
        return output;
    }

    void applyReLU(Tensor3& tensor) const {
        for (auto& channel : tensor) {
            for (auto& row : channel) {
                for (double& value : row) {
                    if (value < 0.0) {
                        value = 0.0;
                    }
                }
            }
        }
    }

    std::pair<Tensor3, Mask3> maxPool(const Tensor3& input) const {
        const int channels = static_cast<int>(input.size());
        const int outHeight = static_cast<int>(input[0].size()) / POOL_SIZE;
        const int outWidth = static_cast<int>(input[0][0].size()) / POOL_SIZE;
        Tensor3 pooled(channels, Matrix(outHeight, std::vector<double>(outWidth, 0.0)));
        Mask3 mask(channels, std::vector<std::vector<int>>(outHeight, std::vector<int>(outWidth, 0)));

        for (int ch = 0; ch < channels; ++ch) {
            for (int r = 0; r < outHeight; ++r) {
                for (int c = 0; c < outWidth; ++c) {
                    double bestValue = -std::numeric_limits<double>::infinity();
                    int bestOffset = 0;
                    for (int pr = 0; pr < POOL_SIZE; ++pr) {
                        for (int pc = 0; pc < POOL_SIZE; ++pc) {
                            const int inputRow = r * POOL_SIZE + pr;
                            const int inputCol = c * POOL_SIZE + pc;
                            const double candidate = input[ch][inputRow][inputCol];
                            const int offset = pr * POOL_SIZE + pc;
                            if (candidate > bestValue) {
                                bestValue = candidate;
                                bestOffset = offset;
                            }
                        }
                    }
                    pooled[ch][r][c] = bestValue;
                    mask[ch][r][c] = bestOffset;
                }
            }
        }

        return {pooled, mask};
    }

    std::vector<double> flatten(const Tensor3& tensor) const {
        std::vector<double> values;
        values.reserve(FLATTEN_SIZE);
        for (const auto& channel : tensor) {
            for (const auto& row : channel) {
                values.insert(values.end(), row.begin(), row.end());
            }
        }
        return values;
    }

    Tensor3 unflatten(const std::vector<double>& values) const {
        Tensor3 tensor(CONV2_CHANNELS, Matrix(POOL2_HEIGHT, std::vector<double>(POOL2_WIDTH, 0.0)));
        size_t index = 0;
        for (int ch = 0; ch < CONV2_CHANNELS; ++ch) {
            for (int r = 0; r < POOL2_HEIGHT; ++r) {
                for (int c = 0; c < POOL2_WIDTH; ++c) {
                    tensor[ch][r][c] = values[index++];
                }
            }
        }
        return tensor;
    }

    std::vector<double> softmax(const std::vector<double>& logits) const {
        std::vector<double> probs = logits;
        const double maxLogit = *std::max_element(probs.begin(), probs.end());
        double sumExp = 0.0;
        for (double& value : probs) {
            value = std::exp(value - maxLogit);
            sumExp += value;
        }
        for (double& value : probs) {
            value /= sumExp;
        }
        return probs;
    }

    ForwardCache forward(const std::vector<double>& image) const {
        ForwardCache cache;
        cache.input = reshapeInput(image);
        cache.conv1Pre = convForwardSingleChannel(cache.input, conv1Kernels, conv1Biases);
        cache.conv1Act = cache.conv1Pre;
        applyReLU(cache.conv1Act);

        auto pool1Result = maxPool(cache.conv1Act);
        cache.pool1 = std::move(pool1Result.first);
        cache.pool1Mask = std::move(pool1Result.second);

        cache.conv2Pre = convForwardMultiChannel(cache.pool1, conv2Kernels, conv2Biases);
        cache.conv2Act = cache.conv2Pre;
        applyReLU(cache.conv2Act);

        auto pool2Result = maxPool(cache.conv2Act);
        cache.pool2 = std::move(pool2Result.first);
        cache.pool2Mask = std::move(pool2Result.second);

        cache.flattened = flatten(cache.pool2);
        cache.logits.assign(OUTPUT_SIZE, 0.0);
        for (int o = 0; o < OUTPUT_SIZE; ++o) {
            double sum = fcBiases[o];
            for (int i = 0; i < FLATTEN_SIZE; ++i) {
                sum += fcWeights[o][i] * cache.flattened[i];
            }
            cache.logits[o] = sum;
        }
        cache.probabilities = softmax(cache.logits);
        return cache;
    }

    Tensor3 maxPoolBackward(const Tensor3& pooledGradient,
                            const Mask3& mask,
                            int inputHeight,
                            int inputWidth) const {
        const int channels = static_cast<int>(pooledGradient.size());
        Tensor3 grad(channels, Matrix(inputHeight, std::vector<double>(inputWidth, 0.0)));
        const int outHeight = static_cast<int>(pooledGradient[0].size());
        const int outWidth = static_cast<int>(pooledGradient[0][0].size());

        for (int ch = 0; ch < channels; ++ch) {
            for (int r = 0; r < outHeight; ++r) {
                for (int c = 0; c < outWidth; ++c) {
                    const int offset = mask[ch][r][c];
                    const int pr = offset / POOL_SIZE;
                    const int pc = offset % POOL_SIZE;
                    grad[ch][r * POOL_SIZE + pr][c * POOL_SIZE + pc] += pooledGradient[ch][r][c];
                }
            }
        }
        return grad;
    }

public:
    std::string getName() const override {
        return "Convolutional Neural Network (CNN)";
    }

    std::string getDescription() const override {
        return "Two-layer convolutional neural network with ReLU, max pooling and Softmax output";
    }

    void initWeights() override {
        initializeParameters();
        std::cout << "CNN weights initialized" << std::endl;
    }

    bool loadParams(const std::string& filename) override {
        std::ifstream file(filename);
        if (!file) {
            std::cerr << "Failed to open parameter file " << filename << std::endl;
            return false;
        }
        conv1Kernels.assign(CONV1_CHANNELS, Matrix(KERNEL_SIZE, std::vector<double>(KERNEL_SIZE, 0.0)));
        conv1Biases.assign(CONV1_CHANNELS, 0.0);
        conv2Kernels.assign(CONV2_CHANNELS, Tensor3(CONV1_CHANNELS, Matrix(KERNEL_SIZE, std::vector<double>(KERNEL_SIZE, 0.0))));
        conv2Biases.assign(CONV2_CHANNELS, 0.0);
        fcWeights.assign(OUTPUT_SIZE, std::vector<double>(FLATTEN_SIZE, 0.0));
        fcBiases.assign(OUTPUT_SIZE, 0.0);

        for (int oc = 0; oc < CONV1_CHANNELS; ++oc) {
            for (int kr = 0; kr < KERNEL_SIZE; ++kr) {
                for (int kc = 0; kc < KERNEL_SIZE; ++kc) {
                    file >> conv1Kernels[oc][kr][kc];
                }
            }
        }
        for (int oc = 0; oc < CONV1_CHANNELS; ++oc) {
            file >> conv1Biases[oc];
        }

        for (int oc = 0; oc < CONV2_CHANNELS; ++oc) {
            for (int ic = 0; ic < CONV1_CHANNELS; ++ic) {
                for (int kr = 0; kr < KERNEL_SIZE; ++kr) {
                    for (int kc = 0; kc < KERNEL_SIZE; ++kc) {
                        file >> conv2Kernels[oc][ic][kr][kc];
                    }
                }
            }
        }
        for (int oc = 0; oc < CONV2_CHANNELS; ++oc) {
            file >> conv2Biases[oc];
        }

        for (int o = 0; o < OUTPUT_SIZE; ++o) {
            for (int i = 0; i < FLATTEN_SIZE; ++i) {
                file >> fcWeights[o][i];
            }
        }
        for (int o = 0; o < OUTPUT_SIZE; ++o) {
            file >> fcBiases[o];
        }
        
        std::cout << "Parameters loaded from " << filename << std::endl;
        return true;
    }

    void saveParams(const std::string& filename) override {
        std::ofstream file(filename);
        if (!file) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }
        for (int oc = 0; oc < CONV1_CHANNELS; ++oc) {
            for (int kr = 0; kr < KERNEL_SIZE; ++kr) {
                for (int kc = 0; kc < KERNEL_SIZE; ++kc) {
                    file << conv1Kernels[oc][kr][kc] << " ";
                }
            }
            file << "\n";
        }
        for (int oc = 0; oc < CONV1_CHANNELS; ++oc) {
            file << conv1Biases[oc] << " ";
        }
        file << "\n";

        for (int oc = 0; oc < CONV2_CHANNELS; ++oc) {
            for (int ic = 0; ic < CONV1_CHANNELS; ++ic) {
                for (int kr = 0; kr < KERNEL_SIZE; ++kr) {
                    for (int kc = 0; kc < KERNEL_SIZE; ++kc) {
                        file << conv2Kernels[oc][ic][kr][kc] << " ";
                    }
                }
                file << "\n";
            }
        }
        for (int oc = 0; oc < CONV2_CHANNELS; ++oc) {
            file << conv2Biases[oc] << " ";
        }
        file << "\n";

        for (int o = 0; o < OUTPUT_SIZE; ++o) {
            for (int i = 0; i < FLATTEN_SIZE; ++i) {
                file << fcWeights[o][i] << " ";
            }
            file << "\n";
        }
        for (int o = 0; o < OUTPUT_SIZE; ++o) {
            file << fcBiases[o] << " ";
        }
        
        std::cout << "Parameters saved to " << filename << std::endl;
    }

    void train(const std::vector<std::vector<double>>& images,
              const std::vector<int>& labels, int epochs) override {
        std::cout << "\nCNN training started (" << epochs << " epochs, batch_size=" << CNN_BATCH_SIZE << ")..." << std::endl;
        std::cout << "  Network: Conv(8,3x3) -> MaxPool -> Conv(16,3x3) -> MaxPool -> Dense(10)" << std::endl;
        const auto started = std::chrono::steady_clock::now();
        const int totalUnits = std::max(1, epochs * static_cast<int>(images.size()));
        if (!isInitialized()) {
            initializeParameters();
            std::cout << "  Initialization: He init" << std::endl;
        } else {
            std::cout << "  Initialization: load existing parameters" << std::endl;
        }

        for (int epoch = 0; epoch < epochs; ++epoch) {
            double totalLoss = 0.0;
            int correctPredictions = 0;

            for (size_t batchStart = 0; batchStart < images.size(); batchStart += CNN_BATCH_SIZE) {
                size_t batchEnd = std::min(batchStart + CNN_BATCH_SIZE, images.size());
                ProgressBar::show(batchEnd, images.size(), "Processing data");
                const int completedUnits = epoch * static_cast<int>(images.size()) + static_cast<int>(batchEnd);
                reportProgress(completedUnits, totalUnits,
                               static_cast<double>(completedUnits) / totalUnits,
                               estimateRemainingSeconds(started, completedUnits, totalUnits));

                for (size_t idx = batchStart; idx < batchEnd; ++idx) {
                    const auto cache = forward(images[idx]);
                    int label = labels[idx];

                    double loss = -std::log(std::max(cache.probabilities[label], 1e-7));
                    totalLoss += loss;

                    int predictedClass = static_cast<int>(std::distance(
                        cache.probabilities.begin(),
                        std::max_element(cache.probabilities.begin(), cache.probabilities.end())));
                    if (predictedClass == label) correctPredictions++;

                    std::vector<double> dLogits = cache.probabilities;
                    dLogits[label] -= 1.0;
                    std::vector<double> dFlatten(FLATTEN_SIZE, 0.0);

                    for (int o = 0; o < OUTPUT_SIZE; ++o) {
                        for (int i = 0; i < FLATTEN_SIZE; ++i) {
                            dFlatten[i] += fcWeights[o][i] * dLogits[o];
                            fcWeights[o][i] -= CNN_LEARNING_RATE * dLogits[o] * cache.flattened[i];
                        }
                        fcBiases[o] -= CNN_LEARNING_RATE * dLogits[o];
                    }

                    Tensor3 dPool2 = unflatten(dFlatten);
                    Tensor3 dConv2Act = maxPoolBackward(dPool2, cache.pool2Mask, CONV2_HEIGHT, CONV2_WIDTH);
                    for (int oc = 0; oc < CONV2_CHANNELS; ++oc) {
                        for (int r = 0; r < CONV2_HEIGHT; ++r) {
                            for (int c = 0; c < CONV2_WIDTH; ++c) {
                                if (cache.conv2Pre[oc][r][c] <= 0.0) {
                                    dConv2Act[oc][r][c] = 0.0;
                                }
                            }
                        }
                    }

                    Tensor3 dPool1(CONV1_CHANNELS, Matrix(POOL1_HEIGHT, std::vector<double>(POOL1_WIDTH, 0.0)));
                    for (int oc = 0; oc < CONV2_CHANNELS; ++oc) {
                        for (int ic = 0; ic < CONV1_CHANNELS; ++ic) {
                            for (int kr = 0; kr < KERNEL_SIZE; ++kr) {
                                for (int kc = 0; kc < KERNEL_SIZE; ++kc) {
                                    double kernelGradient = 0.0;
                                    for (int r = 0; r < CONV2_HEIGHT; ++r) {
                                        for (int c = 0; c < CONV2_WIDTH; ++c) {
                                            kernelGradient += dConv2Act[oc][r][c] * cache.pool1[ic][r + kr][c + kc];
                                            dPool1[ic][r + kr][c + kc] += dConv2Act[oc][r][c] * conv2Kernels[oc][ic][kr][kc];
                                        }
                                    }
                                    conv2Kernels[oc][ic][kr][kc] -= CNN_LEARNING_RATE * kernelGradient;
                                }
                            }
                        }

                        double biasGradient = 0.0;
                        for (int r = 0; r < CONV2_HEIGHT; ++r) {
                            for (int c = 0; c < CONV2_WIDTH; ++c) {
                                biasGradient += dConv2Act[oc][r][c];
                            }
                        }
                        conv2Biases[oc] -= CNN_LEARNING_RATE * biasGradient;
                    }

                    Tensor3 dConv1Act = maxPoolBackward(dPool1, cache.pool1Mask, CONV1_HEIGHT, CONV1_WIDTH);
                    for (int oc = 0; oc < CONV1_CHANNELS; ++oc) {
                        for (int r = 0; r < CONV1_HEIGHT; ++r) {
                            for (int c = 0; c < CONV1_WIDTH; ++c) {
                                if (cache.conv1Pre[oc][r][c] <= 0.0) {
                                    dConv1Act[oc][r][c] = 0.0;
                                }
                            }
                        }
                    }

                    for (int oc = 0; oc < CONV1_CHANNELS; ++oc) {
                        for (int kr = 0; kr < KERNEL_SIZE; ++kr) {
                            for (int kc = 0; kc < KERNEL_SIZE; ++kc) {
                                double kernelGradient = 0.0;
                                for (int r = 0; r < CONV1_HEIGHT; ++r) {
                                    for (int c = 0; c < CONV1_WIDTH; ++c) {
                                        kernelGradient += dConv1Act[oc][r][c] * cache.input[r + kr][c + kc];
                                    }
                                }
                                conv1Kernels[oc][kr][kc] -= CNN_LEARNING_RATE * kernelGradient;
                            }
                        }

                        double biasGradient = 0.0;
                        for (int r = 0; r < CONV1_HEIGHT; ++r) {
                            for (int c = 0; c < CONV1_WIDTH; ++c) {
                                biasGradient += dConv1Act[oc][r][c];
                            }
                        }
                        conv1Biases[oc] -= CNN_LEARNING_RATE * biasGradient;
                    }
                }
            }

            ProgressBar::finish();
            double accuracy = static_cast<double>(correctPredictions) / images.size();
            std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                      << " - avg loss: " << std::fixed << std::setprecision(6) << totalLoss / images.size()
                      << " - train acc: " << std::setprecision(4)
                      << accuracy * 100 << "%"
                      << " - remaining epochs: " << (epochs - epoch - 1) << std::endl;
        }
        
        std::cout << "CNN training completed" << std::endl;
    }

    int predict(const std::vector<double>& image) override {
        if (!isInitialized()) {
            throw std::runtime_error("CNN model parameters are not initialized.");
        }

        const auto cache = forward(image);
        return static_cast<int>(std::distance(cache.probabilities.begin(),
                           std::max_element(cache.probabilities.begin(), cache.probabilities.end())));
    }

    std::vector<double> predictProba(const std::vector<double>& image) override {
        if (!isInitialized()) {
            return std::vector<double>(OUTPUT_SIZE, 1.0 / OUTPUT_SIZE);
        }

        return forward(image).probabilities;
    }

    double evaluate(const std::vector<std::vector<double>>& images,
                   const std::vector<int>& labels) override {
        return computeAccuracyFromPredictions(*this, images, labels);
    }

    std::vector<std::vector<int>> computeConfusionMatrix(
        const std::vector<std::vector<double>>& images,
        const std::vector<int>& labels) override {
        return computeConfusionMatrixFromPredictions(*this, images, labels);
    }

    void computeMetrics(const std::vector<std::vector<int>>& matrix,
                       std::vector<double>& precision,
                       std::vector<double>& recall,
                       std::vector<double>& f1) override {
        computePerClassMetricsFromConfusionMatrix(matrix, precision, recall, f1);
    }

    double computeAUC(const std::vector<std::vector<double>>& images,
                     const std::vector<int>& labels,
                     std::string& rocFilename) override {
        return computeMacroAUCFromProbabilities(*this, images, labels, rocFilename);
    }

    std::string getROCDataFilename() const override {
        return "ROC/cnn_roc_data.txt";
    }

    void evaluateAndReport(const std::vector<std::vector<double>>& images,
                          const std::vector<int>& labels) override {
        printEvaluationReport("CNN", *this, images, labels);
    }
};

// ==================== Random Forest Model ====================

class RandomForestModel : public IClassificationModel {
public:
    virtual std::string getStructureDescription() const override { return "Input(784) -> 50 Decision Trees -> Voting / Probability Average"; }
    virtual Metrics testAndGetMetrics(const std::vector<std::vector<double>>& images, const std::vector<int>& labels) override { return buildMetricsSnapshot(*this, images, labels); }

private:
    static const int NUM_CLASSES = 10;
    static const int INPUT_SIZE = 784;

    struct TreeNode {
        bool isLeaf = true;
        int featureIndex = -1;
        double threshold = 0.0;
        int predictedClass = 0;
        int leftChild = -1;
        int rightChild = -1;
        std::array<double, NUM_CLASSES> classCounts{};
    };

    std::vector<std::vector<TreeNode>> forest;
    std::mt19937 rng;

    int argmaxCounts(const std::array<double, NUM_CLASSES>& counts) const {
        int bestClass = 0;
        double bestValue = counts[0];
        for (int c = 1; c < NUM_CLASSES; ++c) {
            if (counts[c] > bestValue) {
                bestValue = counts[c];
                bestClass = c;
            }
        }
        return bestClass;
    }

    std::array<double, NUM_CLASSES> countLabels(const std::vector<int>& sampleIndices,
                                                const std::vector<int>& labels) const {
        std::array<double, NUM_CLASSES> counts{};
        for (int index : sampleIndices) {
            counts[labels[index]] += 1.0;
        }
        return counts;
    }

    double giniImpurity(const std::array<double, NUM_CLASSES>& counts, int total) const {
        if (total <= 0) {
            return 0.0;
        }

        double impurity = 1.0;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            double probability = counts[c] / total;
            impurity -= probability * probability;
        }
        return impurity;
    }

    std::vector<int> createMiniBatchIndices(size_t totalSamples) {
        size_t batchSize = std::min(static_cast<size_t>(RF_BATCH_SIZE), totalSamples);
        std::vector<int> batchIndices;
        batchIndices.reserve(batchSize);

        std::uniform_int_distribution<int> dist(0, static_cast<int>(totalSamples) - 1);
        for (size_t i = 0; i < batchSize; ++i) {
            batchIndices.push_back(dist(rng));
        }
        return batchIndices;
    }

    std::vector<int> createBootstrapSample(const std::vector<int>& batchIndices) {
        std::vector<int> sampleIndices;
        sampleIndices.reserve(batchIndices.size());

        if (batchIndices.empty()) {
            return sampleIndices;
        }

        std::uniform_int_distribution<int> dist(0, static_cast<int>(batchIndices.size()) - 1);
        for (size_t i = 0; i < batchIndices.size(); ++i) {
            sampleIndices.push_back(batchIndices[dist(rng)]);
        }
        return sampleIndices;
    }

    bool findBestSplit(const std::vector<std::vector<double>>& images,
                       const std::vector<int>& labels,
                       const std::vector<int>& sampleIndices,
                       int& bestFeature,
                       double& bestThreshold,
                       std::vector<int>& bestLeft,
                       std::vector<int>& bestRight) {
        bestFeature = -1;
        bestThreshold = 0.0;
        double bestScore = std::numeric_limits<double>::infinity();
        bestLeft.clear();
        bestRight.clear();

        std::vector<int> featureCandidates(INPUT_SIZE);
        std::iota(featureCandidates.begin(), featureCandidates.end(), 0);
        std::shuffle(featureCandidates.begin(), featureCandidates.end(), rng);

        int featureCount = std::min(RF_FEATURES_PER_SPLIT, INPUT_SIZE);
        featureCandidates.resize(featureCount);

        std::uniform_int_distribution<int> sampleDist(0, static_cast<int>(sampleIndices.size()) - 1);

        for (int feature : featureCandidates) {
            int thresholdCandidates = std::min(8, static_cast<int>(sampleIndices.size()));
            for (int candidateIndex = 0; candidateIndex < thresholdCandidates; ++candidateIndex) {
                int sampleIndex = sampleIndices[sampleDist(rng)];
                double threshold = images[sampleIndex][feature];

                std::vector<int> leftIndices;
                std::vector<int> rightIndices;
                leftIndices.reserve(sampleIndices.size());
                rightIndices.reserve(sampleIndices.size());

                std::array<double, NUM_CLASSES> leftCounts{};
                std::array<double, NUM_CLASSES> rightCounts{};

                for (int index : sampleIndices) {
                    int label = labels[index];
                    if (images[index][feature] <= threshold) {
                        leftIndices.push_back(index);
                        leftCounts[label] += 1.0;
                    } else {
                        rightIndices.push_back(index);
                        rightCounts[label] += 1.0;
                    }
                }

                if (leftIndices.empty() || rightIndices.empty()) {
                    continue;
                }

                double leftWeight = static_cast<double>(leftIndices.size()) / sampleIndices.size();
                double rightWeight = static_cast<double>(rightIndices.size()) / sampleIndices.size();
                double score = leftWeight * giniImpurity(leftCounts, static_cast<int>(leftIndices.size())) +
                               rightWeight * giniImpurity(rightCounts, static_cast<int>(rightIndices.size()));

                if (score < bestScore) {
                    bestScore = score;
                    bestFeature = feature;
                    bestThreshold = threshold;
                    bestLeft = std::move(leftIndices);
                    bestRight = std::move(rightIndices);
                }
            }
        }

        return bestFeature != -1;
    }

    int buildTreeRecursive(const std::vector<std::vector<double>>& images,
                           const std::vector<int>& labels,
                           const std::vector<int>& sampleIndices,
                           int depth,
                           std::vector<TreeNode>& tree) {
        TreeNode node;
        node.classCounts = countLabels(sampleIndices, labels);
        node.predictedClass = argmaxCounts(node.classCounts);

        int currentIndex = static_cast<int>(tree.size());
        tree.push_back(node);

        bool pureNode = node.classCounts[node.predictedClass] >= static_cast<double>(sampleIndices.size());
        bool stopSplitting = depth >= RF_MAX_DEPTH || static_cast<int>(sampleIndices.size()) < RF_MIN_SAMPLES_SPLIT || pureNode;
        if (stopSplitting) {
            tree[currentIndex].isLeaf = true;
            return currentIndex;
        }

        int bestFeature = -1;
        double bestThreshold = 0.0;
        std::vector<int> leftIndices;
        std::vector<int> rightIndices;

        if (!findBestSplit(images, labels, sampleIndices, bestFeature, bestThreshold, leftIndices, rightIndices)) {
            tree[currentIndex].isLeaf = true;
            return currentIndex;
        }

        tree[currentIndex].isLeaf = false;
        tree[currentIndex].featureIndex = bestFeature;
        tree[currentIndex].threshold = bestThreshold;

        int leftChild = buildTreeRecursive(images, labels, leftIndices, depth + 1, tree);
        int rightChild = buildTreeRecursive(images, labels, rightIndices, depth + 1, tree);

        tree[currentIndex].leftChild = leftChild;
        tree[currentIndex].rightChild = rightChild;
        return currentIndex;
    }

    int predictTree(const std::vector<double>& image, const std::vector<TreeNode>& tree) const {
        if (tree.empty()) {
            return 0;
        }

        int currentIndex = 0;
        while (currentIndex >= 0 && currentIndex < static_cast<int>(tree.size())) {
            const TreeNode& node = tree[currentIndex];
            if (node.isLeaf || node.leftChild < 0 || node.rightChild < 0 || node.featureIndex < 0) {
                return node.predictedClass;
            }

            currentIndex = (image[node.featureIndex] <= node.threshold) ? node.leftChild : node.rightChild;
        }

        return 0;
    }

    std::vector<double> predictTreeProba(const std::vector<double>& image, const std::vector<TreeNode>& tree) const {
        std::vector<double> probabilities(NUM_CLASSES, 0.0);
        if (tree.empty()) {
            return probabilities;
        }

        int currentIndex = 0;
        while (currentIndex >= 0 && currentIndex < static_cast<int>(tree.size())) {
            const TreeNode& node = tree[currentIndex];
            if (node.isLeaf || node.leftChild < 0 || node.rightChild < 0 || node.featureIndex < 0) {
                double sumCounts = std::accumulate(node.classCounts.begin(), node.classCounts.end(), 0.0);
                if (sumCounts <= 0.0) {
                    probabilities[node.predictedClass] = 1.0;
                } else {
                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        probabilities[c] = node.classCounts[c] / sumCounts;
                    }
                }
                return probabilities;
            }

            currentIndex = (image[node.featureIndex] <= node.threshold) ? node.leftChild : node.rightChild;
        }

        probabilities[0] = 1.0;
        return probabilities;
    }

    void writeTree(std::ofstream& file, const std::vector<TreeNode>& tree) {
        file << tree.size() << std::endl;
        for (const auto& node : tree) {
            file << (node.isLeaf ? 1 : 0) << " "
                 << node.featureIndex << " "
                 << std::setprecision(17) << node.threshold << " "
                 << node.predictedClass << " "
                 << node.leftChild << " "
                 << node.rightChild;
            for (double count : node.classCounts) {
                file << " " << count;
            }
            file << std::endl;
        }
    }

    bool readTree(std::ifstream& file, std::vector<TreeNode>& tree) {
        int nodeCount = 0;
        file >> nodeCount;
        if (!file || nodeCount <= 0) {
            return false;
        }

        tree.resize(nodeCount);
        for (int i = 0; i < nodeCount; ++i) {
            int isLeafInt = 0;
            file >> isLeafInt
                 >> tree[i].featureIndex
                 >> tree[i].threshold
                 >> tree[i].predictedClass
                 >> tree[i].leftChild
                 >> tree[i].rightChild;
            tree[i].isLeaf = (isLeafInt != 0);
            for (int c = 0; c < NUM_CLASSES; ++c) {
                file >> tree[i].classCounts[c];
            }
            if (!file) {
                return false;
            }
        }

        return true;
    }

public:
    std::string getName() const override {
        return "Random Forest (RF)";
    }

    std::string getDescription() const override {
        return "Random forest ensemble trained with bootstrap samples and random feature subsets";
    }

    void initWeights() override {
        forest.clear();
        forest.reserve(RF_NUM_TREES);
        rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
        std::cout << "Random forest structure initialized" << std::endl;
    }

    bool loadParams(const std::string& filename) override {
        std::ifstream file(filename);
        if (!file) {
            std::cerr << "Failed to open parameter file " << filename << std::endl;
            return false;
        }

        int treeCount = 0;
        int maxDepth = 0;
        int minSamplesSplit = 0;
        int featuresPerSplit = 0;
        file >> treeCount >> maxDepth >> minSamplesSplit >> featuresPerSplit;
        if (!file) {
            return false;
        }

        forest.clear();
        forest.resize(treeCount);
        for (int i = 0; i < treeCount; ++i) {
            if (!readTree(file, forest[i])) {
                std::cerr << "Random forest parameter file read failed" << std::endl;
                forest.clear();
                return false;
            }
        }

        std::cout << "Parameters loaded from " << filename << std::endl;
        return true;
    }

    void saveParams(const std::string& filename) override {
        std::ofstream file(filename);
        if (!file) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }

        file << forest.size() << " " << RF_MAX_DEPTH << " " << RF_MIN_SAMPLES_SPLIT << " " << RF_FEATURES_PER_SPLIT << std::endl;
        for (const auto& tree : forest) {
            writeTree(file, tree);
        }

        std::cout << "Parameters saved to " << filename << std::endl;
    }

    void train(const std::vector<std::vector<double>>& images,
              const std::vector<int>& labels, int epochs) override {
        if (images.empty()) {
            std::cout << "Training set is empty, random forest cannot train" << std::endl;
            return;
        }

        if (forest.capacity() == 0) {
            forest.reserve(RF_NUM_TREES);
        }

        std::cout << "\nRandom forest training started (" << epochs << " epochs, batch_size=" << RF_BATCH_SIZE
                  << ", trees=" << RF_NUM_TREES << ")..." << std::endl;
        std::cout << "  Trees: decision-tree ensemble | max depth: " << RF_MAX_DEPTH
                  << " | features per split: " << RF_FEATURES_PER_SPLIT << std::endl;
        std::cout << "  Standard random forest is not epoch-optimized; the epoch parameter is ignored and a fixed-size forest is built." << std::endl;
        std::cout << "  Initialization: bootstrap over full training set + random feature subsets | evaluation: vote average" << std::endl;
        const auto started = std::chrono::steady_clock::now();
        const int totalUnits = std::max(1, RF_NUM_TREES);

        std::vector<int> allSampleIndices(images.size());
        std::iota(allSampleIndices.begin(), allSampleIndices.end(), 0);

        forest.clear();
        double totalLoss = 0.0;
        int correctPredictions = 0;
        size_t sampleCount = 0;

        for (int treeIndex = 0; treeIndex < RF_NUM_TREES; ++treeIndex) {
            std::vector<int> bootstrapSample = createBootstrapSample(allSampleIndices);

            std::vector<TreeNode> tree;
            tree.reserve((1 << (RF_MAX_DEPTH + 1)) - 1);
            buildTreeRecursive(images, labels, bootstrapSample, 0, tree);
            forest.push_back(std::move(tree));

            for (int index : bootstrapSample) {
                auto probabilities = predictProba(images[index]);
                int predictedClass = std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()));
                totalLoss += -std::log(std::max(probabilities[labels[index]], 1e-7));
                if (predictedClass == labels[index]) {
                    correctPredictions++;
                }
                sampleCount++;
            }

            ProgressBar::show(forest.size(), RF_NUM_TREES, "Building forest");
            const int completedUnits = static_cast<int>(forest.size());
            reportProgress(completedUnits, totalUnits,
                           static_cast<double>(completedUnits) / totalUnits,
                           estimateRemainingSeconds(started, completedUnits, totalUnits));
        }

        ProgressBar::finish();
        double accuracy = sampleCount > 0 ? static_cast<double>(correctPredictions) / sampleCount : 0.0;
        double averageLoss = sampleCount > 0 ? totalLoss / sampleCount : 0.0;
        std::cout << "Forest build summary"
                  << " - avg loss: " << std::fixed << std::setprecision(6) << averageLoss
                  << " - train acc: " << std::setprecision(4) << accuracy * 100 << "%" << std::endl;
        std::cout << "Random forest training completed" << std::endl;
    }

    int predict(const std::vector<double>& image) override {
        if (forest.empty()) {
            return 0;
        }

        std::vector<int> votes(NUM_CLASSES, 0);
        for (const auto& tree : forest) {
            int prediction = predictTree(image, tree);
            votes[prediction]++;
        }

        return std::distance(votes.begin(), std::max_element(votes.begin(), votes.end()));
    }

    std::vector<double> predictProba(const std::vector<double>& image) override {
        std::vector<double> probabilities(NUM_CLASSES, 0.0);
        if (forest.empty()) {
            for (double& value : probabilities) {
                value = 1.0 / NUM_CLASSES;
            }
            return probabilities;
        }

        for (const auto& tree : forest) {
            auto treeProbabilities = predictTreeProba(image, tree);
            for (int c = 0; c < NUM_CLASSES; ++c) {
                probabilities[c] += treeProbabilities[c];
            }
        }

        for (double& value : probabilities) {
            value /= forest.size();
        }
        return probabilities;
    }

    double evaluate(const std::vector<std::vector<double>>& images,
                   const std::vector<int>& labels) override {
        return computeAccuracyFromPredictions(*this, images, labels);
    }

    std::vector<std::vector<int>> computeConfusionMatrix(
        const std::vector<std::vector<double>>& images,
        const std::vector<int>& labels) override {
        return computeConfusionMatrixFromPredictions(*this, images, labels);
    }

    void computeMetrics(const std::vector<std::vector<int>>& matrix,
                       std::vector<double>& precision,
                       std::vector<double>& recall,
                       std::vector<double>& f1) override {
        computePerClassMetricsFromConfusionMatrix(matrix, precision, recall, f1);
    }

    double computeAUC(const std::vector<std::vector<double>>& images,
                     const std::vector<int>& labels,
                     std::string& rocFilename) override {
        return computeMacroAUCFromProbabilities(*this, images, labels, rocFilename);
    }

    std::string getROCDataFilename() const override {
        return "ROC/rf_roc_data.txt";
    }

    void evaluateAndReport(const std::vector<std::vector<double>>& images,
                          const std::vector<int>& labels) override {
        printEvaluationReport("Random Forest", *this, images, labels);
    }
};

// ==================== KNN Model ====================

class KNNModel : public IClassificationModel {
public:
    virtual std::string getStructureDescription() const override { return "Input(784) -> Full Training Sample Memory -> k=5 Distance Vote"; }
    virtual Metrics testAndGetMetrics(const std::vector<std::vector<double>>& images, const std::vector<int>& labels) override { return buildMetricsSnapshot(*this, images, labels); }

private:
    static const int NUM_CLASSES = 10;
    static const int INPUT_SIZE = 784;

    int kValue = KNN_K;
    std::vector<std::vector<double>> prototypes;
    std::vector<int> prototypeLabels;
    std::mt19937 rng;

    double squaredDistance(const std::vector<double>& a, const std::vector<double>& b) const {
        double distance = 0.0;
        for (int i = 0; i < INPUT_SIZE; ++i) {
            double diff = a[i] - b[i];
            distance += diff * diff;
        }
        return distance;
    }

    void addPrototype(const std::vector<double>& image, int label) {
        if (prototypes.size() < static_cast<size_t>(KNN_MAX_PROTOTYPES)) {
            prototypes.push_back(image);
            prototypeLabels.push_back(label);
            return;
        }

        std::uniform_int_distribution<int> dist(0, static_cast<int>(prototypes.size()));
        size_t replaceIndex = static_cast<size_t>(dist(rng));
        if (replaceIndex < prototypes.size()) {
            prototypes[replaceIndex] = image;
            prototypeLabels[replaceIndex] = label;
        }
    }

    std::vector<std::pair<double, int>> getNearestNeighbors(const std::vector<double>& image) const {
        std::vector<std::pair<double, int>> distances;
        distances.reserve(prototypes.size());

        for (size_t i = 0; i < prototypes.size(); ++i) {
            distances.emplace_back(squaredDistance(image, prototypes[i]), prototypeLabels[i]);
        }

        int neighborCount = std::min(kValue, static_cast<int>(distances.size()));
        if (neighborCount <= 0) {
            return {};
        }

        if (neighborCount < static_cast<int>(distances.size())) {
            std::nth_element(distances.begin(), distances.begin() + neighborCount, distances.end(),
                             [](const auto& left, const auto& right) {
                                 return left.first < right.first;
                             });
        }
        distances.resize(neighborCount);
        std::sort(distances.begin(), distances.end(), [](const auto& left, const auto& right) {
            return left.first < right.first;
        });
        return distances;
    }

public:
    std::string getName() const override {
        return "K-Nearest Neighbors (KNN)";
    }

    std::string getDescription() const override {
        return "Exact KNN classifier storing all training samples and using distance-weighted voting";
    }

    void initWeights() override {
        prototypes.clear();
        prototypeLabels.clear();
        rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
        std::cout << "KNN sample store initialized" << std::endl;
    }

    bool loadParams(const std::string& filename) override {
        std::ifstream file(filename);
        if (!file) {
            std::cerr << "Failed to open parameter file " << filename << std::endl;
            return false;
        }

        int prototypeCount = 0;
        int inputSize = 0;
        int storedK = 0;
        int storedMaxPrototypes = 0;
        file >> prototypeCount >> inputSize >> storedK >> storedMaxPrototypes;
        if (!file || inputSize != INPUT_SIZE) {
            std::cerr << "Parameter file shape mismatch" << std::endl;
            return false;
        }

        kValue = storedK > 0 ? storedK : KNN_K;
        prototypes.clear();
        prototypeLabels.clear();
        prototypes.reserve(prototypeCount);
        prototypeLabels.reserve(prototypeCount);

        for (int i = 0; i < prototypeCount; ++i) {
            int label = 0;
            std::vector<double> image(INPUT_SIZE, 0.0);
            file >> label;
            for (int j = 0; j < INPUT_SIZE; ++j) {
                file >> image[j];
            }
            if (!file) {
                std::cerr << "KNN parameter file read failed" << std::endl;
                prototypes.clear();
                prototypeLabels.clear();
                return false;
            }
            prototypes.push_back(std::move(image));
            prototypeLabels.push_back(label);
        }

        std::cout << "Parameters loaded from " << filename << std::endl;
        return true;
    }

    void saveParams(const std::string& filename) override {
        std::ofstream file(filename);
        if (!file) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }

        file << prototypes.size() << " " << INPUT_SIZE << " " << kValue << " " << KNN_MAX_PROTOTYPES << std::endl;
        for (size_t i = 0; i < prototypes.size(); ++i) {
            file << prototypeLabels[i] << " ";
            for (int j = 0; j < INPUT_SIZE; ++j) {
                file << prototypes[i][j] << " ";
            }
            file << std::endl;
        }

        std::cout << "Parameters saved to " << filename << std::endl;
    }

    void train(const std::vector<std::vector<double>>& images,
              const std::vector<int>& labels, int epochs) override {
        if (images.empty()) {
            std::cout << "Training set is empty, KNN cannot train" << std::endl;
            return;
        }

        std::cout << "\nKNN training started (" << epochs << " epochs, batch_size=" << KNN_BATCH_SIZE
                  << ", k=" << kValue << ")..." << std::endl;
        std::cout << "  Standard KNN does not optimize iteratively; the epoch parameter is ignored after loading the dataset." << std::endl;
        std::cout << "  Initialization: exact sample store | retained samples: " << images.size() << std::endl;
        const auto started = std::chrono::steady_clock::now();

        prototypes.clear();
        prototypeLabels.clear();
        prototypes.reserve(images.size());
        prototypeLabels.reserve(labels.size());

        const int totalUnits = std::max(1, static_cast<int>(images.size()));
        for (size_t i = 0; i < images.size(); ++i) {
            prototypes.push_back(images[i]);
            prototypeLabels.push_back(labels[i]);

            const int completedUnits = static_cast<int>(i + 1);
            ProgressBar::show(i + 1, images.size(), "Loading samples");
            reportProgress(completedUnits, totalUnits,
                           static_cast<double>(completedUnits) / totalUnits,
                           estimateRemainingSeconds(started, completedUnits, totalUnits));
        }

        ProgressBar::finish();
        std::cout << "Stored " << prototypes.size() << " training samples for exact KNN search" << std::endl;
        std::cout << "KNN training completed" << std::endl;
    }

    int predict(const std::vector<double>& image) override {
        if (prototypes.empty()) {
            return 0;
        }

        auto neighbors = getNearestNeighbors(image);
        if (neighbors.empty()) {
            return 0;
        }

        std::vector<double> votes(NUM_CLASSES, 0.0);
        for (const auto& neighbor : neighbors) {
            double weight = 1.0 / (std::sqrt(neighbor.first) + 1e-6);
            votes[neighbor.second] += weight;
        }

        return std::distance(votes.begin(), std::max_element(votes.begin(), votes.end()));
    }

    std::vector<double> predictProba(const std::vector<double>& image) override {
        std::vector<double> probabilities(NUM_CLASSES, 0.0);
        if (prototypes.empty()) {
            for (double& value : probabilities) {
                value = 1.0 / NUM_CLASSES;
            }
            return probabilities;
        }

        auto neighbors = getNearestNeighbors(image);
        if (neighbors.empty()) {
            for (double& value : probabilities) {
                value = 1.0 / NUM_CLASSES;
            }
            return probabilities;
        }

        double sumWeights = 0.0;
        for (const auto& neighbor : neighbors) {
            double weight = 1.0 / (std::sqrt(neighbor.first) + 1e-6);
            probabilities[neighbor.second] += weight;
            sumWeights += weight;
        }

        if (sumWeights > 0.0) {
            for (double& value : probabilities) {
                value /= sumWeights;
            }
        }

        return probabilities;
    }

    double evaluate(const std::vector<std::vector<double>>& images,
                   const std::vector<int>& labels) override {
        return computeAccuracyFromPredictions(*this, images, labels);
    }

    std::vector<std::vector<int>> computeConfusionMatrix(
        const std::vector<std::vector<double>>& images,
        const std::vector<int>& labels) override {
        return computeConfusionMatrixFromPredictions(*this, images, labels);
    }

    void computeMetrics(const std::vector<std::vector<int>>& matrix,
                       std::vector<double>& precision,
                       std::vector<double>& recall,
                       std::vector<double>& f1) override {
        computePerClassMetricsFromConfusionMatrix(matrix, precision, recall, f1);
    }

    double computeAUC(const std::vector<std::vector<double>>& images,
                     const std::vector<int>& labels,
                     std::string& rocFilename) override {
        return computeMacroAUCFromProbabilities(*this, images, labels, rocFilename);
    }

    std::string getROCDataFilename() const override {
        return "ROC/knn_roc_data.txt";
    }

    void evaluateAndReport(const std::vector<std::vector<double>>& images,
                          const std::vector<int>& labels) override {
        printEvaluationReport("KNN", *this, images, labels);
    }
};

// ==================== Logistic Regression Model ====================

class LogisticRegressionModel : public IClassificationModel {
public:
    virtual std::string getStructureDescription() const override { return "Input(784) -> Linear Projection -> Softmax(10)"; }
    virtual Metrics testAndGetMetrics(const std::vector<std::vector<double>>& images, const std::vector<int>& labels) override { return buildMetricsSnapshot(*this, images, labels); }

private:
    static const int NUM_CLASSES = 10;
    static const int INPUT_SIZE = 784;
    
    // Softmax regression parameters.
    std::vector<std::vector<double>> weights;  // NUM_CLASSES x INPUT_SIZE
    std::vector<double> biases;                // NUM_CLASSES

public:
    std::string getName() const override {
        return "Logistic Regression (Softmax)";
    }

    std::string getDescription() const override {
        return "Logistic regression with Softmax output, cross-entropy loss and SGD optimization";
    }

    void initWeights() override {
        weights.assign(NUM_CLASSES, std::vector<double>(INPUT_SIZE, 0.0));
        biases.assign(NUM_CLASSES, 0.0);
        
        // Xavier initialization: std = sqrt(2 / (fan_in + fan_out)).
        std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
        double std_w = std::sqrt(2.0 / (INPUT_SIZE + NUM_CLASSES));
        std::normal_distribution<double> dist(0.0, std_w);
        
        for (auto& w_row : weights) {
            for (auto& w : w_row) {
                w = dist(rng);
            }
        }
        std::cout << "Logistic regression weights initialized" << std::endl;
    }

    bool loadParams(const std::string& filename) override {
        std::ifstream file(filename);
        if (!file) {
            std::cerr << "Failed to open parameter file " << filename << std::endl;
            return false;
        }
        int classes, inputs;
        file >> classes >> inputs;
        if (classes != NUM_CLASSES || inputs != INPUT_SIZE) {
            std::cerr << "Parameter file shape mismatch" << std::endl;
            return false;
        }
        
        weights.resize(NUM_CLASSES, std::vector<double>(INPUT_SIZE));
        for (int c = 0; c < NUM_CLASSES; ++c) {
            for (int i = 0; i < INPUT_SIZE; ++i) {
                file >> weights[c][i];
            }
        }
        
        biases.resize(NUM_CLASSES);
        for (int c = 0; c < NUM_CLASSES; ++c) {
            file >> biases[c];
        }
        
        std::cout << "Parameters loaded from " << filename << std::endl;
        return true;
    }

    void saveParams(const std::string& filename) override {
        std::ofstream file(filename);
        file << NUM_CLASSES << " " << INPUT_SIZE << std::endl;
        
        for (int c = 0; c < NUM_CLASSES; ++c) {
            for (int i = 0; i < INPUT_SIZE; ++i) {
                file << weights[c][i] << " ";
            }
            file << std::endl;
        }
        
        for (int c = 0; c < NUM_CLASSES; ++c) {
            file << biases[c] << " ";
        }
        file << std::endl;
        
        std::cout << "Parameters saved to " << filename << std::endl;
    }

    void train(const std::vector<std::vector<double>>& images,
              const std::vector<int>& labels, int epochs) override {
        std::cout << "\nLogistic regression training started (" << epochs << " epochs, batch_size=" << LR_BATCH_SIZE << ")..." << std::endl;
        std::cout << "  Network: [" << INPUT_SIZE << "] -> Softmax -> [" << NUM_CLASSES << "]" << std::endl;
        std::cout << "  Initialization: Xavier init | Optimizer: SGD | Regularization: L2(" << LR_REGULARIZATION_L2 << ")\n" << std::endl;
        const auto started = std::chrono::steady_clock::now();
        const int totalUnits = std::max(1, epochs * static_cast<int>(images.size()));
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double totalLoss = 0.0;
            int correctPredictions = 0;
            
            // Process one mini-batch.
            for (size_t batchStart = 0; batchStart < images.size(); batchStart += LR_BATCH_SIZE) {
                size_t batchEnd = std::min(batchStart + LR_BATCH_SIZE, images.size());
                int batchSize = batchEnd - batchStart;
                
                ProgressBar::show(batchEnd, images.size(), "Processing data");
                const int completedUnits = epoch * static_cast<int>(images.size()) + static_cast<int>(batchEnd);
                reportProgress(completedUnits, totalUnits,
                               static_cast<double>(completedUnits) / totalUnits,
                               estimateRemainingSeconds(started, completedUnits, totalUnits));
                
                // Accumulate gradients.
                std::vector<std::vector<double>> dWeights(NUM_CLASSES, 
                    std::vector<double>(INPUT_SIZE, 0.0));
                std::vector<double> dBiases(NUM_CLASSES, 0.0);
                
                for (size_t idx = batchStart; idx < batchEnd; ++idx) {
                    const auto& image = images[idx];
                    int label = labels[idx];
                    
                    // ==================== 鍓嶅悜浼犳挱 ====================
                    // Compute logits: z = W * x + b.
                    std::vector<double> logits(NUM_CLASSES, 0.0);
                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        logits[c] = biases[c];
                        for (int i = 0; i < INPUT_SIZE; ++i) {
                            logits[c] += weights[c][i] * image[i];
                        }
                    }
                    
                    // Softmax normalization.
                    std::vector<double> probs(NUM_CLASSES);
                    double maxLogit = *std::max_element(logits.begin(), logits.end());
                    double sumExp = 0.0;
                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        probs[c] = std::exp(logits[c] - maxLogit);
                        sumExp += probs[c];
                    }
                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        probs[c] /= sumExp;
                    }
                    
                    // ==================== Loss ====================
                    double loss = -std::log(std::max(probs[label], 1e-7));
                    totalLoss += loss;
                    
                    // Track training accuracy.
                    int predictedClass = std::distance(probs.begin(), 
                                                       std::max_element(probs.begin(), probs.end()));
                    if (predictedClass == label) correctPredictions++;
                    
                    // ==================== Backpropagation ====================
                    // For softmax + cross-entropy: grad = probs - one_hot(label).
                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        double gradient = probs[c] - (c == label ? 1.0 : 0.0);
                        
                        // Accumulate weight gradients with L2 regularization.
                        for (int i = 0; i < INPUT_SIZE; ++i) {
                            dWeights[c][i] += gradient * image[i] + 
                                            LR_REGULARIZATION_L2 * weights[c][i];
                        }
                        
                        // Accumulate bias gradients.
                        dBiases[c] += gradient;
                    }
                }
                
                // ==================== Average Gradients and Update ====================
                double learningRate = LR_LEARNING_RATE;
                for (int c = 0; c < NUM_CLASSES; ++c) {
                    for (int i = 0; i < INPUT_SIZE; ++i) {
                        dWeights[c][i] /= batchSize;
                        weights[c][i] -= learningRate * dWeights[c][i];
                    }
                    dBiases[c] /= batchSize;
                    biases[c] -= learningRate * dBiases[c];
                }
            }
            
            ProgressBar::finish();
            double accuracy = static_cast<double>(correctPredictions) / images.size();
            std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                      << " - loss: " << std::fixed << std::setprecision(6) << totalLoss / images.size()
                      << " - accuracy: " << std::setprecision(4)
                      << accuracy * 100 << "%"
                      << " - remaining epochs: " << (epochs - epoch - 1) << std::endl;
        }
        
        std::cout << "Logistic regression training completed" << std::endl;
    }

    int predict(const std::vector<double>& image) override {
        std::vector<double> logits(NUM_CLASSES, 0.0);
        for (int c = 0; c < NUM_CLASSES; ++c) {
            logits[c] = biases[c];
            for (int i = 0; i < INPUT_SIZE; ++i) {
                logits[c] += weights[c][i] * image[i];
            }
        }
        
        return std::distance(logits.begin(), 
                           std::max_element(logits.begin(), logits.end()));
    }

    std::vector<double> predictProba(const std::vector<double>& image) override {
        std::vector<double> logits(NUM_CLASSES, 0.0);
        for (int c = 0; c < NUM_CLASSES; ++c) {
            logits[c] = biases[c];
            for (int i = 0; i < INPUT_SIZE; ++i) {
                logits[c] += weights[c][i] * image[i];
            }
        }
        
        // Softmax
        std::vector<double> probs(NUM_CLASSES);
        double maxLogit = *std::max_element(logits.begin(), logits.end());
        double sumExp = 0.0;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            probs[c] = std::exp(logits[c] - maxLogit);
            sumExp += probs[c];
        }
        for (int c = 0; c < NUM_CLASSES; ++c) {
            probs[c] /= sumExp;
        }
        
        return probs;
    }

    double evaluate(const std::vector<std::vector<double>>& images,
                   const std::vector<int>& labels) override {
        return computeAccuracyFromPredictions(*this, images, labels);
    }

    std::vector<std::vector<int>> computeConfusionMatrix(
        const std::vector<std::vector<double>>& images,
        const std::vector<int>& labels) override {
        return computeConfusionMatrixFromPredictions(*this, images, labels);
    }

    void computeMetrics(const std::vector<std::vector<int>>& matrix,
                       std::vector<double>& precision,
                       std::vector<double>& recall,
                       std::vector<double>& f1) override {
        computePerClassMetricsFromConfusionMatrix(matrix, precision, recall, f1);
    }

    double computeAUC(const std::vector<std::vector<double>>& images,
                     const std::vector<int>& labels,
                     std::string& rocFilename) override {
        return computeMacroAUCFromProbabilities(*this, images, labels, rocFilename);
    }

    std::string getROCDataFilename() const override {
        return "ROC/lr_roc_data.txt";
    }

    void evaluateAndReport(const std::vector<std::vector<double>>& images,
                          const std::vector<int>& labels) override {
        printEvaluationReport("Logistic Regression", *this, images, labels);
    }
};

// ==================== Menu System ====================

class MenuSystem {
public:
    /**
     * Show the main menu.
     */
    static void showMainMenu() {
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "  MNIST training and evaluation system" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        std::cout << "\nChoose an operation mode:\n" << std::endl;
        std::cout << "  1. Train model" << std::endl;
        std::cout << "  2. Test model" << std::endl;
        std::cout << "  3. View ROC data guidance" << std::endl;
        std::cout << "  4. Open CNN visualization guide" << std::endl;
        std::cout << "  5. Open desktop visualization guide" << std::endl;
        std::cout << "  6. Exit" << std::endl;
        std::cout << "\nEnter choice (1-6): ";
    }

    /**
     * Show the model selection menu.
     */
    static int showModelSelectionMenu() {
        std::cout << "\n" << std::string(50, '-') << std::endl;
        std::cout << "  Choose model" << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        auto models = ModelFactory::getAvailableModels();
        for (const auto& model : models) {
            std::cout << "  " << model.first << ". " << model.second << std::endl;
        }
        std::cout << "\nEnter choice: ";

        int choice;
        std::cin >> choice;
        return choice;
    }

    /**
     * Show the training options menu.
     */
    static int showTrainingMenu() {
        std::cout << "\n" << std::string(50, '-') << std::endl;
        std::cout << "  Training options" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        std::cout << "  1. Use random initialization and start new training" << std::endl;
        std::cout << "  2. Load existing parameters and continue training" << std::endl;
        std::cout << "\nEnter choice: ";

        int choice;
        std::cin >> choice;
        return choice;
    }

    /**
     * Show the test result menu.
     */
    static void showTestResultMenu(const std::string& rocDataFile) {
        std::cout << "\n" << std::string(50, '-') << std::endl;
        std::cout << "  Test completed" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        std::cout << "\nROC data saved to: " << rocDataFile << std::endl;
        std::cout << "\nROC viewing note:\n" << std::endl;
        std::cout << "  Open the desktop Test page to view ROC charts directly" << std::endl;
        std::cout << "\n";
    }

    /**
     * Show the ROC data menu.
     */
    static std::string showPlotMenu() {
        std::cout << "\n" << std::string(50, '-') << std::endl;
        std::cout << "  ROC data options" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        std::cout << "  1. SVM ROC data" << std::endl;
        std::cout << "  2. FCNN ROC data" << std::endl;
        std::cout << "  3. CNN ROC data" << std::endl;
        std::cout << "  4. Logistic Regression ROC data" << std::endl;
        std::cout << "  5. Random Forest ROC data" << std::endl;
        std::cout << "  6. KNN ROC data" << std::endl;
        std::cout << "  7. All model ROC data (compare together)" << std::endl;
        std::cout << "\nEnter choice: ";

        int choice;
        std::cin >> choice;

        switch (choice) {
            case 1: return "ROC/svm_roc_data.txt";
            case 2: return "ROC/fcnn_roc_data.txt";
            case 3: return "ROC/cnn_roc_data.txt";
            case 4: return "ROC/lr_roc_data.txt";
            case 5: return "ROC/rf_roc_data.txt";
            case 6: return "ROC/knn_roc_data.txt";
            case 7: return "__ALL__";
            default: return "";
        }
    }

    static void showCnnVisualizationGuide() {
        std::cout << "\n" << std::string(50, '-') << std::endl;
        std::cout << "  CNN visualization guide" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        std::cout << "1) First select [2. Test model] and then choose CNN to generate visualization data" << std::endl;
        std::cout << "2) Then inspect the result in the desktop Test and Visualize pages" << std::endl;
        std::cout << "3) The browser-based visualization path is no longer the primary entry" << std::endl;
    }

    static void showVisualMenuGuide() {
        std::cout << "\n" << std::string(50, '-') << std::endl;
        std::cout << "  Desktop visualization guide" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        std::cout << "1) The desktop application supports training, testing and structure visualization" << std::endl;
        std::cout << "2) The command-line menu is still retained" << std::endl;
        std::cout << "3) Core training and testing still run entirely in native C++" << std::endl;
    }
};

// ==================== Model Factory ====================

// All model classes are defined above, so the factory can be implemented here.
std::shared_ptr<IClassificationModel> ModelFactory::createModel(ModelType type) {
    switch (type) {
        case SVM:
            return std::make_shared<SVMModel>();
        case FCNN:
            return std::make_shared<FCNNModel>();
        case CNN:
            return std::make_shared<CNNModel>();
        case LOGISTIC_REGRESSION:
            return std::make_shared<LogisticRegressionModel>();
        case RANDOM_FOREST:
            return std::make_shared<RandomForestModel>();
        case KNN:
            return std::make_shared<KNNModel>();
        default:
            return nullptr;
    }
}

std::vector<std::pair<int, std::string>> ModelFactory::getAvailableModels() {
    return {
        {SVM, "Support Vector Machine (SVM) - linear baseline"},
        {FCNN, "Fully Connected Neural Network (FCNN) - backpropagation network"},
        {CNN, "Convolutional Neural Network (CNN) - true convolution + pooling model"},
        {LOGISTIC_REGRESSION, "Logistic Regression (LR) - softmax linear classifier"},
        {RANDOM_FOREST, "Random Forest (RF) - tree ensemble"},
        {KNN, "K-Nearest Neighbors (KNN) - exact sample-memory voting"}
    };
}

std::string ModelFactory::getModelTypeName(ModelType type) {
    switch (type) {
        case SVM:
            return "SVM";
        case FCNN:
            return "FCNN";
        case CNN:
            return "CNN";
        case LOGISTIC_REGRESSION:
            return "LR";
        case RANDOM_FOREST:
            return "RF";
        case KNN:
            return "KNN";
        default:
            return "Unknown";
    }
}

// ==================== 涓荤▼搴?====================


