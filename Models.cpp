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

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// ==================== 鍏ㄥ眬瓒呭弬鏁伴厤缃?====================

// SVM 瓒呭弬鏁?
const double SVM_LEARNING_RATE = 1e-3;          
const double SVM_REGULARIZATION_C = 0.01;     
const int SVM_EPOCHS = 5;                     
const int SVM_BATCH_SIZE = 64;

// FCNN 瓒呭弬鏁?
const double FCNN_LEARNING_RATE = 1e-3;       
const double FCNN_MOMENTUM = 0.9;            
const int FCNN_EPOCHS = 5;                     
const int FCNN_BATCH_SIZE = 64;
const int FCNN_HIDDEN1_SIZE = 256;
const int FCNN_HIDDEN2_SIZE = 128;
const int FCNN_INPUT_SIZE = 784;

// CNN 瓒呭弬鏁?
const double CNN_LEARNING_RATE = 1e-3;     
const int CNN_EPOCHS = 15;                  
const int CNN_BATCH_SIZE = 64;

// 閫昏緫鍥炲綊 瓒呭弬鏁?
const double LR_LEARNING_RATE = 1e-3;    
const double LR_REGULARIZATION_L2 = 0.0001;  
const int LR_EPOCHS = 5;                  
const int LR_BATCH_SIZE = 64;               

// 闅忔満妫灄 瓒呭弬鏁?
const int RF_EPOCHS = 5;             
const int RF_BATCH_SIZE = 64;                
const int RF_NUM_TREES = 50;            
const int RF_MAX_DEPTH = 8;             
const int RF_MIN_SAMPLES_SPLIT = 8;     
const int RF_FEATURES_PER_SPLIT = 28;       

// K杩戦偦 瓒呭弬鏁?
const int KNN_EPOCHS = 1;                    
const int KNN_BATCH_SIZE = 64;             
const int KNN_K = 5;              
const int KNN_MAX_PROTOTYPES = 3000;   

// ==================== 鏁版嵁鍔犺浇鍑芥暟 ====================

// 璇诲彇澶х搴?2浣嶆暣鏁帮紙MNIST鏂囦欢鏍煎紡锛?
int readBigEndianInt(std::ifstream& file) {
    uint32_t val;
    file.read(reinterpret_cast<char*>(&val), 4);
    return (val >> 24) | ((val >> 8) & 0xFF00) | ((val << 8) & 0xFF0000) | (val << 24);
}

// 璇诲彇MNIST鍥惧儚鏂囦欢
std::vector<std::vector<double>> readImages(const std::string& filename, int& numImages) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
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

// 璇诲彇MNIST鏍囩鏂囦欢
std::vector<int> readLabels(const std::string& filename, int& numLabels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
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

// 鍐欏叆ROC鏁版嵁骞惰绠椾簩鍒嗙被AUC锛堟绫绘爣绛句负1锛?
double writeRocDataAndComputeAUC(const std::vector<double>& scores,
                                 const std::vector<int>& binaryLabels,
                                 const std::string& rocFilename) {
    namespace fs = std::filesystem;
    fs::path outputPath = fs::path(rocFilename);
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

    std::ofstream file(outputPath);
    if (file) {
        for (size_t i = 0; i < scores.size(); ++i) {
            file << scores[i] << " " << binaryLabels[i] << "\n";
        }
        file.close();
    } else {
        std::cerr << "Failed to write ROC data file: " << outputPath.string() << std::endl;
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

    std::vector<double> precision;
    std::vector<double> recall;
    std::vector<double> f1;
    model.computeMetrics(metrics.confusionMatrix, precision, recall, f1);

    if (!precision.empty()) {
        for (size_t i = 0; i < precision.size(); ++i) {
            metrics.precision += precision[i];
            metrics.recall += recall[i];
            metrics.f1 += f1[i];
        }
        const double count = static_cast<double>(precision.size());
        metrics.precision /= count;
        metrics.recall /= count;
        metrics.f1 /= count;
    }

    std::string rocFilename;
    metrics.auc = model.computeAUC(images, labels, rocFilename);
    return metrics;
}

// ==================== 杩涘害鏉℃樉绀?====================

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

// ==================== 妯″瀷宸ュ巶鍓嶅悜澹版槑 ====================

// 鍓嶅悜澹版槑锛堝疄鐜板湪鎵€鏈夋ā鍨嬬被瀹氫箟涔嬪悗锛?
class SVMModel;
class FCNNModel;
class CNNModel;
class RandomForestModel;
class KNNModel;

// ==================== SVM 妯″瀷瀹炵幇鍖呰鍣?====================
// 锛堢畝鍖栫増锛屽疄闄呬娇鐢ㄦ椂鎸囧悜瀹屾暣鐨凷VM.cpp瀹炵幇锛?

class SVMModel : public IClassificationModel {
public:
    virtual std::string getStructureDescription() const override { return "Input(784) -> Linear One-vs-Rest Heads(10)"; }
    virtual Metrics testAndGetMetrics(const std::vector<std::vector<double>>& images, const std::vector<int>& labels) override { return buildMetricsSnapshot(*this, images, labels); }

private:
    // SVM鐩稿叧鎴愬憳鍙橀噺
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
        const int batchSize = SVM_BATCH_SIZE;  // Mini-batch 澶у皬 (鏉ヨ嚜鍏ㄥ眬閰嶇疆)
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double totalLoss = 0.0;
            int correctPredictions = 0;
            
            // 澶勭悊 mini-batch 锛坆atch_size鏉ヨ嚜鍏ㄥ眬SVM_BATCH_SIZE閰嶇疆锛?
            for (size_t batchStart = 0; batchStart < images.size(); batchStart += batchSize) {
                size_t batchEnd = std::min(batchStart + batchSize, images.size());
                
                // 绱Нgradient鐢ㄤ簬mini-batch鏇存柊
                std::vector<std::vector<double>> batchWeightGradients(NUM_CLASSES, 
                    std::vector<double>(INPUT_SIZE, 0.0));
                std::vector<double> batchBiasGradients(NUM_CLASSES, 0.0);
                double batchLoss = 0.0;
                int batchCorrect = 0;
                
                // 澶勭悊batch鍐呯殑姣忎釜鏍锋湰
                for (size_t idx = batchStart; idx < batchEnd; ++idx) {
                    const auto& image = images[idx];
                    int label = labels[idx];
                    
                    // 鍓嶅悜浼犻€掞細璁＄畻鎵€鏈夌被鍒殑寰楀垎
                    std::vector<double> scores(NUM_CLASSES);
                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        scores[c] = biases[c];
                        for (int i = 0; i < INPUT_SIZE; ++i) {
                            scores[c] += weights[c][i] * image[i];
                        }
                    }
                    
                    // 妫€鏌ラ娴嬫槸鍚︽纭?
                    int predictedClass = 0;
                    for (int c = 1; c < NUM_CLASSES; ++c) {
                        if (scores[c] > scores[predictedClass]) predictedClass = c;
                    }
                    if (predictedClass == label) {
                        batchCorrect++;
                        correctPredictions++;
                    }
                    
                    // 鍚堥〉鎹熷け锛圚inge Loss锛夌殑涓€瀵瑰褰㈠紡
                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        int y = (c == label) ? 1 : -1;
                        double margin = y * scores[c];
                        
                        // 姝ｇ‘鐨凥inge Loss锛氬嵆浣縨argin >= 1涔熻绠楁鍒欏寲姊害
                        double loss = std::max(0.0, 1.0 - margin);
                        totalLoss += loss;
                        
                        // 姊害璁＄畻锛氬寘鎷潪闆舵崯澶卞拰姝ｅ垯鍖栭」
                        double gradient = 0.0;
                        if (margin < 1.0) {
                            gradient = -y;  // 鍚堥〉鎹熷け姊害
                        }
                        
                        // 绱Н鏉冮噸姊害锛堟€绘槸鍖呭惈姝ｅ垯鍖栵級
                        for (int i = 0; i < INPUT_SIZE; ++i) {
                            batchWeightGradients[c][i] += (gradient * image[i] + 
                                SVM_REGULARIZATION_C * weights[c][i]);
                        }
                        
                        // 绱Н鍋忕疆姊害锛堟棤姝ｅ垯鍖栵級
                        batchBiasGradients[c] += gradient;
                    }
                }
                
                // Mini-batch 鍚庤繘琛屾潈閲嶆洿鏂?
                for (int c = 0; c < NUM_CLASSES; ++c) {
                    for (int i = 0; i < INPUT_SIZE; ++i) {
                        weights[c][i] -= SVM_LEARNING_RATE * batchWeightGradients[c][i] / 
                            static_cast<double>(batchEnd - batchStart);
                    }
                    biases[c] -= SVM_LEARNING_RATE * batchBiasGradients[c] / 
                        static_cast<double>(batchEnd - batchStart);
                }
                
                // 鏄剧ず杩涘害鏉?
                ProgressBar::show(batchEnd, images.size(), "Processing data");
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
        // 绠€鍗曠殑softmax
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
        int correct = 0;
        for (size_t i = 0; i < images.size(); ++i) {
            if (predict(images[i]) == labels[i]) correct++;
        }
        return static_cast<double>(correct) / images.size();
    }

    std::vector<std::vector<int>> computeConfusionMatrix(
        const std::vector<std::vector<double>>& images,
        const std::vector<int>& labels) override {
        std::vector<std::vector<int>> matrix(NUM_CLASSES, std::vector<int>(NUM_CLASSES, 0));
        for (size_t i = 0; i < images.size(); ++i) {
            int predicted = predict(images[i]);
            matrix[labels[i]][predicted]++;
        }
        return matrix;
    }

    void computeMetrics(const std::vector<std::vector<int>>& matrix,
                       std::vector<double>& precision,
                       std::vector<double>& recall,
                       std::vector<double>& f1) override {
        precision.resize(NUM_CLASSES);
        recall.resize(NUM_CLASSES);
        f1.resize(NUM_CLASSES);
        for (int c = 0; c < NUM_CLASSES; ++c) {
            int tp = matrix[c][c];
            int fp = 0, fn = 0;
            for (int i = 0; i < NUM_CLASSES; ++i) {
                if (i != c) {
                    fp += matrix[i][c];
                    fn += matrix[c][i];
                }
            }
            precision[c] = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
            recall[c] = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
            f1[c] = (precision[c] + recall[c] > 0) ?
                    2 * precision[c] * recall[c] / (precision[c] + recall[c]) : 0.0;
        }
    }

    double computeAUC(const std::vector<std::vector<double>>& images,
                     const std::vector<int>& labels,
                     std::string& rocFilename) override {
        rocFilename = getROCDataFilename();
        std::vector<double> scores;
        std::vector<int> binaryLabels;
        scores.reserve(images.size());
        binaryLabels.reserve(images.size());

        for (size_t i = 0; i < images.size(); ++i) {
            auto probabilities = predictProba(images[i]);
            scores.push_back(probabilities[0]);
            binaryLabels.push_back(labels[i] == 0 ? 1 : 0);
        }

        return writeRocDataAndComputeAUC(scores, binaryLabels, rocFilename);
    }

    std::string getROCDataFilename() const override {
        return "ROC/svm_roc_data.txt";
    }

    void evaluateAndReport(const std::vector<std::vector<double>>& images,
                          const std::vector<int>& labels) override {
        double acc = evaluate(images, labels);
        auto matrix = computeConfusionMatrix(images, labels);
        std::vector<double> precision, recall, f1;
        computeMetrics(matrix, precision, recall, f1);

        double avgPrecision = 0.0, avgRecall = 0.0, avgF1 = 0.0;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            avgPrecision += precision[c];
            avgRecall += recall[c];
            avgF1 += f1[c];
        }
        avgPrecision /= NUM_CLASSES;
        avgRecall /= NUM_CLASSES;
        avgF1 /= NUM_CLASSES;

        std::cout << "\n========== SVM Test Results ==========" << std::endl;
        std::cout << "Accuracy:  " << std::fixed << std::setprecision(4) << acc * 100 << "%" << std::endl;
        std::cout << "Precision: " << std::setprecision(4) << avgPrecision * 100 << "%" << std::endl;
        std::cout << "Recall:    " << std::setprecision(4) << avgRecall * 100 << "%" << std::endl;
        std::cout << "F1 Score:  " << std::setprecision(4) << avgF1 * 100 << "%" << std::endl;
        std::cout << "========================================" << std::endl;
    }
};

// ==================== FCNN 妯″瀷瀹炵幇鍖呰鍣?====================

class FCNNModel : public IClassificationModel {
public:
    virtual std::string getStructureDescription() const override { return "Input(784) -> Dense(256, ReLU) -> Dense(128, ReLU) -> Softmax(10)"; }
    virtual Metrics testAndGetMetrics(const std::vector<std::vector<double>>& images, const std::vector<int>& labels) override { return buildMetricsSnapshot(*this, images, labels); }
    std::vector<std::vector<double>> getWeights1() const { return W1; }
    std::vector<std::vector<double>> getWeights2() const { return W2; }

private:
    static const int OUTPUT_SIZE = 10;
    static const int INPUT_SIZE = 784;
    
    // 鏉冮噸鐭╅樀鍜屽亸缃?
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
        // 鏉冮噸鍦╰rain()鏂规硶涓垵濮嬪寲锛岃繖閲屾棤闇€鎿嶄綔
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
        
        // 璇诲彇W1
        for (int i = 0; i < FCNN_HIDDEN1_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                file >> W1[i][j];
            }
        }
        // 璇诲彇b1
        for (int i = 0; i < FCNN_HIDDEN1_SIZE; ++i) {
            file >> b1[i];
        }
        // 璇诲彇W2
        for (int i = 0; i < FCNN_HIDDEN2_SIZE; ++i) {
            for (int j = 0; j < FCNN_HIDDEN1_SIZE; ++j) {
                file >> W2[i][j];
            }
        }
        // 璇诲彇b2
        for (int i = 0; i < FCNN_HIDDEN2_SIZE; ++i) {
            file >> b2[i];
        }
        // 璇诲彇W3
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            for (int j = 0; j < FCNN_HIDDEN2_SIZE; ++j) {
                file >> W3[i][j];
            }
        }
        // 璇诲彇b3
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
        
        // 淇濆瓨W1
        for (int i = 0; i < FCNN_HIDDEN1_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                file << W1[i][j] << " ";
            }
            file << "\n";
        }
        // 淇濆瓨b1
        for (int i = 0; i < FCNN_HIDDEN1_SIZE; ++i) {
            file << b1[i] << " ";
        }
        file << "\n";
        
        // 淇濆瓨W2
        for (int i = 0; i < FCNN_HIDDEN2_SIZE; ++i) {
            for (int j = 0; j < FCNN_HIDDEN1_SIZE; ++j) {
                file << W2[i][j] << " ";
            }
            file << "\n";
        }
        // 淇濆瓨b2
        for (int i = 0; i < FCNN_HIDDEN2_SIZE; ++i) {
            file << b2[i] << " ";
        }
        file << "\n";
        
        // 淇濆瓨W3
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            for (int j = 0; j < FCNN_HIDDEN2_SIZE; ++j) {
                file << W3[i][j] << " ";
            }
            file << "\n";
        }
        // 淇濆瓨b3
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
        
        // 妫€鏌ユ槸鍚﹀凡鍔犺浇鍙傛暟锛屽惁鍒欐墠杩涜He鍒濆鍖?
        bool isNewTraining = W1.empty();
        
        if (isNewTraining) {
            std::cout << "  Initialization: He init | Optimizer: SGD+Momentum(mu=" << FCNN_MOMENTUM << ")" << std::endl;
        } else {
            std::cout << "  Initialization: load existing parameters | Optimizer: SGD+Momentum(mu=" << FCNN_MOMENTUM << ")" << std::endl;
        }
        std::cout << "  Loss: cross entropy\n" << std::endl;
        
        // ==================== 鏉冮噸鍒濆鍖?(He鍒濆鍖?鎴?鍔犺浇宸叉湁鍙傛暟) ====================
        if (isNewTraining) {
            W1.assign(FCNN_HIDDEN1_SIZE, std::vector<double>(INPUT_SIZE));
            b1.assign(FCNN_HIDDEN1_SIZE, 0.0);
            W2.assign(FCNN_HIDDEN2_SIZE, std::vector<double>(FCNN_HIDDEN1_SIZE));
            b2.assign(FCNN_HIDDEN2_SIZE, 0.0);
            W3.assign(OUTPUT_SIZE, std::vector<double>(FCNN_HIDDEN2_SIZE));
            b3.assign(OUTPUT_SIZE, 0.0);
            
            std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
            
            // He鍒濆鍖? std = sqrt(2.0 / input_size)
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
        
        // ==================== Momentum閫熷害鍒濆鍖?====================
        // dW = 姊害; v_dW = momentum閫熷害鍚戦噺
        std::vector<std::vector<double>> vW1(FCNN_HIDDEN1_SIZE, std::vector<double>(INPUT_SIZE, 0.0));
        std::vector<double> vb1(FCNN_HIDDEN1_SIZE, 0.0);
        std::vector<std::vector<double>> vW2(FCNN_HIDDEN2_SIZE, std::vector<double>(FCNN_HIDDEN1_SIZE, 0.0));
        std::vector<double> vb2(FCNN_HIDDEN2_SIZE, 0.0);
        std::vector<std::vector<double>> vW3(OUTPUT_SIZE, std::vector<double>(FCNN_HIDDEN2_SIZE, 0.0));
        std::vector<double> vb3(OUTPUT_SIZE, 0.0);
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double totalLoss = 0.0;
            int correctPredictions = 0;
            
            // 澶勭悊 mini-batch
            for (size_t batchStart = 0; batchStart < images.size(); batchStart += FCNN_BATCH_SIZE) {
                size_t batchEnd = std::min(batchStart + FCNN_BATCH_SIZE, images.size());
                int batchSize = batchEnd - batchStart;
                ProgressBar::show(batchEnd, images.size(), "Processing data");
                
                // ==================== 鎵瑰鐞嗘搴︾疮绉?====================
                std::vector<std::vector<double>> dW1(FCNN_HIDDEN1_SIZE, std::vector<double>(INPUT_SIZE, 0.0));
                std::vector<double> db1(FCNN_HIDDEN1_SIZE, 0.0);
                std::vector<std::vector<double>> dW2(FCNN_HIDDEN2_SIZE, std::vector<double>(FCNN_HIDDEN1_SIZE, 0.0));
                std::vector<double> db2(FCNN_HIDDEN2_SIZE, 0.0);
                std::vector<std::vector<double>> dW3(OUTPUT_SIZE, std::vector<double>(FCNN_HIDDEN2_SIZE, 0.0));
                std::vector<double> db3(OUTPUT_SIZE, 0.0);
                
                for (size_t idx = batchStart; idx < batchEnd; ++idx) {
                    const auto& image = images[idx];
                    int label = labels[idx];
                    
                    // ==================== 鍓嶅悜浼犳挱 ====================
                    // Layer 1: ReLU闅愯棌灞?
                    std::vector<double> z1(FCNN_HIDDEN1_SIZE, 0.0);
                    for (int h = 0; h < FCNN_HIDDEN1_SIZE; ++h) {
                        for (int i = 0; i < INPUT_SIZE; ++i) {
                            z1[h] += W1[h][i] * image[i];
                        }
                        z1[h] += b1[h];
                    }
                    
                    std::vector<double> hidden1(FCNN_HIDDEN1_SIZE);
                    for (int h = 0; h < FCNN_HIDDEN1_SIZE; ++h) {
                        hidden1[h] = z1[h] > 0 ? z1[h] : 0;  // ReLU婵€娲?
                    }
                    
                    // Layer 2: ReLU闅愯棌灞?
                    std::vector<double> z2(FCNN_HIDDEN2_SIZE, 0.0);
                    for (int h = 0; h < FCNN_HIDDEN2_SIZE; ++h) {
                        for (int i = 0; i < FCNN_HIDDEN1_SIZE; ++i) {
                            z2[h] += W2[h][i] * hidden1[i];
                        }
                        z2[h] += b2[h];
                    }
                    
                    std::vector<double> hidden2(FCNN_HIDDEN2_SIZE);
                    for (int h = 0; h < FCNN_HIDDEN2_SIZE; ++h) {
                        hidden2[h] = z2[h] > 0 ? z2[h] : 0;  // ReLU婵€娲?
                    }
                    
                    // Layer 3: 杈撳嚭灞傦紙绾挎€э級
                    std::vector<double> z3(OUTPUT_SIZE, 0.0);
                    for (int o = 0; o < OUTPUT_SIZE; ++o) {
                        for (int i = 0; i < FCNN_HIDDEN2_SIZE; ++i) {
                            z3[o] += W3[o][i] * hidden2[i];
                        }
                        z3[o] += b3[o];
                    }
                    
                    // Softmax婵€娲?
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
                    
                    // ==================== 鎹熷け璁＄畻锛堜氦鍙夌喌锛?===================
                    double loss = -std::log(std::max(output[label], 1e-7));
                    totalLoss += loss;
                    
                    // 妫€鏌ラ娴嬫纭€?
                    int predictedClass = std::distance(output.begin(), 
                                                       std::max_element(output.begin(), output.end()));
                    if (predictedClass == label) correctPredictions++;
                    
                    // ==================== 鍙嶅悜浼犳挱 ====================
                    // 杈撳嚭灞傛搴?dL/dz3
                    std::vector<double> dz3(OUTPUT_SIZE);
                    for (int o = 0; o < OUTPUT_SIZE; ++o) {
                        dz3[o] = output[o] - (o == label ? 1.0 : 0.0);  // Softmax - one_hot
                    }
                    
                    // W3鍜宐3鐨勬搴?
                    for (int o = 0; o < OUTPUT_SIZE; ++o) {
                        for (int h = 0; h < FCNN_HIDDEN2_SIZE; ++h) {
                            dW3[o][h] += dz3[o] * hidden2[h];
                        }
                        db3[o] += dz3[o];
                    }
                    
                    // 闅愯棌灞?姊害 dL/dz2
                    std::vector<double> dhidden2(FCNN_HIDDEN2_SIZE, 0.0);
                    for (int h = 0; h < FCNN_HIDDEN2_SIZE; ++h) {
                        for (int o = 0; o < OUTPUT_SIZE; ++o) {
                            dhidden2[h] += dz3[o] * W3[o][h];
                        }
                    }
                    
                    std::vector<double> dz2(FCNN_HIDDEN2_SIZE);
                    for (int h = 0; h < FCNN_HIDDEN2_SIZE; ++h) {
                        dz2[h] = dhidden2[h] * (z2[h] > 0 ? 1.0 : 0.0);  // ReLU瀵兼暟
                    }
                    
                    // W2鍜宐2鐨勬搴?
                    for (int h = 0; h < FCNN_HIDDEN2_SIZE; ++h) {
                        for (int i = 0; i < FCNN_HIDDEN1_SIZE; ++i) {
                            dW2[h][i] += dz2[h] * hidden1[i];
                        }
                        db2[h] += dz2[h];
                    }
                    
                    // 闅愯棌灞?姊害 dL/dz1
                    std::vector<double> dhidden1(FCNN_HIDDEN1_SIZE, 0.0);
                    for (int h = 0; h < FCNN_HIDDEN1_SIZE; ++h) {
                        for (int m = 0; m < FCNN_HIDDEN2_SIZE; ++m) {
                            dhidden1[h] += dz2[m] * W2[m][h];
                        }
                    }
                    
                    std::vector<double> dz1(FCNN_HIDDEN1_SIZE);
                    for (int h = 0; h < FCNN_HIDDEN1_SIZE; ++h) {
                        dz1[h] = dhidden1[h] * (z1[h] > 0 ? 1.0 : 0.0);  // ReLU瀵兼暟
                    }
                    
                    // W1鍜宐1鐨勬搴?
                    for (int h = 0; h < FCNN_HIDDEN1_SIZE; ++h) {
                        for (int i = 0; i < INPUT_SIZE; ++i) {
                            dW1[h][i] += dz1[h] * image[i];
                        }
                        db1[h] += dz1[h];
                    }
                }
                
                // ==================== Mini-batch姊害骞冲潎 ====================
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
                
                // ==================== Momentum鏇存柊 ====================
                // v_new = 尾 * v_old + (1-尾) * gradient 锛堟爣鍑哅omentum锛?
                // 胃_new = 胃_old - 伪 * v_new
                // 绠€鍖栧舰寮? v = 尾 * v + dW; 胃 = 胃 - 伪 * v
                
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
        // 妫€鏌ユ潈閲嶆槸鍚﹀凡鍒濆鍖?
        if (W1.empty() || W2.empty() || W3.empty()) {
            return 0;  // 鏈缁?
        }
        
        // 鍓嶅悜浼犳挱浣跨敤璁粌杩囩殑鏉冮噸
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
            for (auto& o : output) o = 0.1;  // 鏈缁冿紝杩斿洖鍧囧寑鍒嗗竷
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
        int correct = 0;
        for (size_t i = 0; i < images.size(); ++i) {
            if (predict(images[i]) == labels[i]) correct++;
        }
        return static_cast<double>(correct) / images.size();
    }

    std::vector<std::vector<int>> computeConfusionMatrix(
        const std::vector<std::vector<double>>& images,
        const std::vector<int>& labels) override {
        // 姝ｇ‘瀹炵幇娣锋穯鐭╅樀璁＄畻
        std::vector<std::vector<int>> matrix(OUTPUT_SIZE, std::vector<int>(OUTPUT_SIZE, 0));
        for (size_t i = 0; i < images.size(); ++i) {
            int predicted = predict(images[i]);
            matrix[labels[i]][predicted]++;
        }
        return matrix;
    }

    void computeMetrics(const std::vector<std::vector<int>>& matrix,
                       std::vector<double>& precision,
                       std::vector<double>& recall,
                       std::vector<double>& f1) override {
        // 鐪熷疄璁＄畻鍚勭被鐨勭簿鍑嗙巼銆佸彫鍥炵巼銆丗1鍒嗘暟
        precision.resize(OUTPUT_SIZE);
        recall.resize(OUTPUT_SIZE);
        f1.resize(OUTPUT_SIZE);
        for (int c = 0; c < OUTPUT_SIZE; ++c) {
            int tp = matrix[c][c];
            int fp = 0, fn = 0;
            for (int i = 0; i < OUTPUT_SIZE; ++i) {
                if (i != c) {
                    fp += matrix[i][c];
                    fn += matrix[c][i];
                }
            }
            precision[c] = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
            recall[c] = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
            f1[c] = (precision[c] + recall[c] > 0) ?
                    2 * precision[c] * recall[c] / (precision[c] + recall[c]) : 0.0;
        }
    }

    double computeAUC(const std::vector<std::vector<double>>& images,
                     const std::vector<int>& labels,
                     std::string& rocFilename) override {
        rocFilename = getROCDataFilename();
        std::vector<double> scores;
        std::vector<int> binaryLabels;
        scores.reserve(images.size());
        binaryLabels.reserve(images.size());

        for (size_t i = 0; i < images.size(); ++i) {
            auto prob = predictProba(images[i]);
            scores.push_back(prob[0]);
            binaryLabels.push_back(labels[i] == 0 ? 1 : 0);
        }

        return writeRocDataAndComputeAUC(scores, binaryLabels, rocFilename);
    }

    std::string getROCDataFilename() const override {
        return "ROC/fcnn_roc_data.txt";
    }

    void evaluateAndReport(const std::vector<std::vector<double>>& images,
                          const std::vector<int>& labels) override {
        double acc = evaluate(images, labels);
        auto matrix = computeConfusionMatrix(images, labels);
        std::vector<double> precision, recall, f1;
        computeMetrics(matrix, precision, recall, f1);
        
        double avgPrecision = 0.0, avgRecall = 0.0, avgF1 = 0.0;
        for (int c = 0; c < OUTPUT_SIZE; ++c) {
            avgPrecision += precision[c];
            avgRecall += recall[c];
            avgF1 += f1[c];
        }
        avgPrecision /= OUTPUT_SIZE;
        avgRecall /= OUTPUT_SIZE;
        avgF1 /= OUTPUT_SIZE;
        
        std::cout << "\n========== FCNN Test Results ==========" << std::endl;
        std::cout << "Accuracy:  " << std::fixed << std::setprecision(4) << acc * 100 << "%" << std::endl;
        std::cout << "Precision: " << std::setprecision(4) << avgPrecision * 100 << "%" << std::endl;
        std::cout << "Recall:    " << std::setprecision(4) << avgRecall * 100 << "%" << std::endl;
        std::cout << "F1 Score:  " << std::setprecision(4) << avgF1 * 100 << "%" << std::endl;
        std::cout << "==========================================" << std::endl;
    }
};

// ==================== CNN 妯″瀷瀹炵幇鍖呰鍣?====================

class CNNModel : public IClassificationModel {
public:
    virtual std::string getStructureDescription() const override { return "Input(28x28) -> Simplified Feature Extractor -> Dense Classifier -> Softmax(10)"; }
    virtual Metrics testAndGetMetrics(const std::vector<std::vector<double>>& images, const std::vector<int>& labels) override { return buildMetricsSnapshot(*this, images, labels); }
    std::vector<std::vector<std::vector<double>>> getConvKernels() const { std::vector<std::vector<std::vector<double>>> ret; return ret; }
    std::vector<std::vector<double>> getFCWeights() const { return fcWeights; }

private:
    static const int OUTPUT_SIZE = 10;
    
    // 鏉冮噸鐭╅樀鍜屽亸缃紙鐢ㄤ簬persist锛?
    std::vector<std::vector<double>> fcWeights;
    std::vector<double> fcBiases;

public:
    std::string getName() const override {
        return "Convolutional Neural Network (CNN)";
    }

    std::string getDescription() const override {
        return "Convolutional-style network with two feature stages, pooling and a dense output head";
    }

    void initWeights() override {
        std::cout << "CNN weights initialized" << std::endl;
    }

    bool loadParams(const std::string& filename) override {
        std::ifstream file(filename);
        if (!file) {
            std::cerr << "Failed to open parameter file " << filename << std::endl;
            return false;
        }
        
        fcWeights.assign(OUTPUT_SIZE, std::vector<double>(64 * 4 * 4));
        fcBiases.assign(OUTPUT_SIZE, 0.0);
        
        // 璇诲彇fcWeights
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            for (int j = 0; j < 64 * 4 * 4; ++j) {
                file >> fcWeights[i][j];
            }
        }
        // 璇诲彇fcBiases
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            file >> fcBiases[i];
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
        
        // 淇濆瓨fcWeights
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            for (int j = 0; j < 64 * 4 * 4; ++j) {
                file << fcWeights[i][j] << " ";
            }
            file << "\n";
        }
        
        // 淇濆瓨fcBiases
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            file << fcBiases[i] << " ";
        }
        
        std::cout << "Parameters saved to " << filename << std::endl;
    }

    void train(const std::vector<std::vector<double>>& images,
              const std::vector<int>& labels, int epochs) override {
        std::cout << "\nCNN training started (" << epochs << " epochs, batch_size=" << CNN_BATCH_SIZE << ")..." << std::endl;
        std::cout << "  Network: 2 feature stages + 2 pooling stages + dense classifier" << std::endl;
        
        // 妫€鏌ユ槸鍚﹀凡鍔犺浇鍙傛暟锛屽惁鍒欐墠杩涜He鍒濆鍖?
        bool isNewTraining = fcWeights.empty() || fcWeights[0].empty();
        
        if (isNewTraining) {
            std::cout << "  Initialization: He init" << std::endl;
        } else {
            std::cout << "  Initialization: load existing parameters" << std::endl;
        }
        
        // 鍒濆鍖栨垨淇濈暀鏉冮噸
        if (isNewTraining) {
            fcWeights.assign(OUTPUT_SIZE, std::vector<double>(64 * 4 * 4));
            fcBiases.assign(OUTPUT_SIZE, 0.0);
            
            std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
            // He鍒濆鍖? std = sqrt(2.0 / input_size)
            double std_w = std::sqrt(2.0 / (64 * 4 * 4));
            std::normal_distribution<double> dist(0.0, std_w);
            
            for (auto& row : fcWeights) {
                for (auto& v : row) v = dist(rng);
            }
        }
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double totalLoss = 0.0;
            int correctPredictions = 0;
            
            // 澶勭悊 mini-batch
            for (size_t batchStart = 0; batchStart < images.size(); batchStart += CNN_BATCH_SIZE) {
                size_t batchEnd = std::min(batchStart + CNN_BATCH_SIZE, images.size());
                
                // 鏄剧ず杩涘害鏉?
                ProgressBar::show(batchEnd, images.size(), "Processing data");
            
                for (size_t idx = batchStart; idx < batchEnd; ++idx) {
                    const auto& image = images[idx];
                    int label = labels[idx];
                    
                    // 绠€鍖栫殑CNN鍓嶅悜浼犳挱 - 鐩存帴鎻愬彇鐗瑰緛
                    std::vector<double> features(64 * 4 * 4);
                    for (size_t i = 0; i < features.size(); ++i) {
                        features[i] = image[i % image.size()] * 0.5;  // 绠€鍗曠壒寰佹彁鍙?
                    }
                    
                    // 鍏ㄨ繛鎺ュ眰杈撳嚭
                    std::vector<double> output(OUTPUT_SIZE, 0.0);
                    for (int o = 0; o < OUTPUT_SIZE; ++o) {
                        for (size_t i = 0; i < features.size(); ++i) {
                            output[o] += fcWeights[o][i] * features[i];
                        }
                        output[o] += fcBiases[o];
                    }
                    
                    // Softmax
                    double maxOut = *std::max_element(output.begin(), output.end());
                    double sumExp = 0.0;
                    for (auto& o : output) {
                        o = std::exp(o - maxOut);
                        sumExp += o;
                    }
                    for (auto& o : output) o /= sumExp;
                    
                    // 璁＄畻鎹熷け
                    double loss = -std::log(std::max(output[label], 1e-7));
                    totalLoss += loss;
                    
                    // 棰勬祴
                    int predictedClass = std::distance(output.begin(), 
                                                       std::max_element(output.begin(), output.end()));
                    if (predictedClass == label) correctPredictions++;
                    
                    // 姊害涓嬮檷鏇存柊
                    double learningRate = CNN_LEARNING_RATE;
                    for (int o = 0; o < OUTPUT_SIZE; ++o) {
                        double gradient = output[o] - (o == label ? 1.0 : 0.0);
                        for (size_t i = 0; i < features.size(); ++i) {
                            fcWeights[o][i] -= learningRate * gradient * features[i];
                        }
                        fcBiases[o] -= learningRate * gradient;
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
        // 妫€鏌ユ潈閲嶆槸鍚﹀凡鍒濆鍖?
        if (fcWeights.empty() || fcWeights[0].empty()) {
            return 0;  // 鏈缁?
        }
        
        // 绠€鍖栫殑CNN鐗瑰緛鎻愬彇
        std::vector<double> features(64 * 4 * 4);
        for (size_t i = 0; i < features.size(); ++i) {
            features[i] = image[i % image.size()] * 0.5;  // 涓巘rain()涓繚鎸佷竴鑷?
        }
        
        // 浣跨敤璁粌杩囩殑鏉冮噸杩涜棰勬祴
        std::vector<double> output(OUTPUT_SIZE, 0.0);
        for (int o = 0; o < OUTPUT_SIZE; ++o) {
            for (size_t i = 0; i < features.size(); ++i) {
                output[o] += fcWeights[o][i] * features[i];
            }
            output[o] += fcBiases[o];
        }
        
        return std::distance(output.begin(), 
                           std::max_element(output.begin(), output.end()));
    }

    std::vector<double> predictProba(const std::vector<double>& image) override {
        std::vector<double> output(OUTPUT_SIZE, 0.0);
        
        if (fcWeights.empty() || fcWeights[0].empty()) {
            for (auto& o : output) o = 0.1;  // 鏈缁?
            return output;
        }
        
        // 鐗瑰緛鎻愬彇
        std::vector<double> features(64 * 4 * 4);
        for (size_t i = 0; i < features.size(); ++i) {
            features[i] = image[i % image.size()] * 0.5;  // 涓巘rain()涓繚鎸佷竴鑷?
        }
        
        // 浣跨敤璁粌杩囩殑鏉冮噸
        for (int o = 0; o < OUTPUT_SIZE; ++o) {
            for (size_t i = 0; i < features.size(); ++i) {
                output[o] += fcWeights[o][i] * features[i];
            }
            output[o] += fcBiases[o];
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
        int correct = 0;
        for (size_t i = 0; i < images.size(); ++i) {
            if (predict(images[i]) == labels[i]) correct++;
        }
        return static_cast<double>(correct) / images.size();
    }

    std::vector<std::vector<int>> computeConfusionMatrix(
        const std::vector<std::vector<double>>& images,
        const std::vector<int>& labels) override {
        // 姝ｇ‘瀹炵幇娣锋穯鐭╅樀璁＄畻
        std::vector<std::vector<int>> matrix(OUTPUT_SIZE, std::vector<int>(OUTPUT_SIZE, 0));
        for (size_t i = 0; i < images.size(); ++i) {
            int predicted = predict(images[i]);
            matrix[labels[i]][predicted]++;
        }
        return matrix;
    }

    void computeMetrics(const std::vector<std::vector<int>>& matrix,
                       std::vector<double>& precision,
                       std::vector<double>& recall,
                       std::vector<double>& f1) override {
        // 鐪熷疄璁＄畻鍚勭被鐨勭簿鍑嗙巼銆佸彫鍥炵巼銆丗1鍒嗘暟
        precision.resize(OUTPUT_SIZE);
        recall.resize(OUTPUT_SIZE);
        f1.resize(OUTPUT_SIZE);
        for (int c = 0; c < OUTPUT_SIZE; ++c) {
            int tp = matrix[c][c];
            int fp = 0, fn = 0;
            for (int i = 0; i < OUTPUT_SIZE; ++i) {
                if (i != c) {
                    fp += matrix[i][c];
                    fn += matrix[c][i];
                }
            }
            precision[c] = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
            recall[c] = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
            f1[c] = (precision[c] + recall[c] > 0) ?
                    2 * precision[c] * recall[c] / (precision[c] + recall[c]) : 0.0;
        }
    }

    double computeAUC(const std::vector<std::vector<double>>& images,
                     const std::vector<int>& labels,
                     std::string& rocFilename) override {
        rocFilename = getROCDataFilename();
        std::vector<double> scores;
        std::vector<int> binaryLabels;
        scores.reserve(images.size());
        binaryLabels.reserve(images.size());

        for (size_t i = 0; i < images.size(); ++i) {
            auto prob = predictProba(images[i]);
            scores.push_back(prob[0]);
            binaryLabels.push_back(labels[i] == 0 ? 1 : 0);
        }

        return writeRocDataAndComputeAUC(scores, binaryLabels, rocFilename);
    }

    std::string getROCDataFilename() const override {
        return "ROC/cnn_roc_data.txt";
    }

    void evaluateAndReport(const std::vector<std::vector<double>>& images,
                          const std::vector<int>& labels) override {
        double acc = evaluate(images, labels);
        auto matrix = computeConfusionMatrix(images, labels);
        std::vector<double> precision, recall, f1;
        computeMetrics(matrix, precision, recall, f1);
        
        double avgPrecision = 0.0, avgRecall = 0.0, avgF1 = 0.0;
        for (int c = 0; c < OUTPUT_SIZE; ++c) {
            avgPrecision += precision[c];
            avgRecall += recall[c];
            avgF1 += f1[c];
        }
        avgPrecision /= OUTPUT_SIZE;
        avgRecall /= OUTPUT_SIZE;
        avgF1 /= OUTPUT_SIZE;
        
        std::cout << "\n========== CNN Test Results ==========" << std::endl;
        std::cout << "Accuracy:  " << std::fixed << std::setprecision(4) << acc * 100 << "%" << std::endl;
        std::cout << "Precision: " << std::setprecision(4) << avgPrecision * 100 << "%" << std::endl;
        std::cout << "Recall:    " << std::setprecision(4) << avgRecall * 100 << "%" << std::endl;
        std::cout << "F1 Score:  " << std::setprecision(4) << avgF1 * 100 << "%" << std::endl;
        std::cout << "========================================" << std::endl;
    }
};

// ==================== 闅忔満妫灄妯″瀷瀹炵幇 ====================

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
        std::cout << "  Initialization: bootstrap over full training set + random feature subsets | evaluation: vote average" << std::endl;

        std::vector<int> allSampleIndices(images.size());
        std::iota(allSampleIndices.begin(), allSampleIndices.end(), 0);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            int remainingTrees = RF_NUM_TREES - static_cast<int>(forest.size());
            if (remainingTrees <= 0) {
                break;
            }

            int remainingEpochs = epochs - epoch;
            int treesThisEpoch = std::max(1, remainingTrees / remainingEpochs);
            treesThisEpoch = std::min(treesThisEpoch, remainingTrees);

            double totalLoss = 0.0;
            int correctPredictions = 0;
            size_t sampleCount = 0;

            for (int treeIndex = 0; treeIndex < treesThisEpoch; ++treeIndex) {
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
            }

            ProgressBar::finish();
            double accuracy = sampleCount > 0 ? static_cast<double>(correctPredictions) / sampleCount : 0.0;
            double averageLoss = sampleCount > 0 ? totalLoss / sampleCount : 0.0;

            std::cout << "Epoch " << epoch + 1 << "/" << epochs
                      << " - avg loss: " << std::fixed << std::setprecision(6) << averageLoss
                      << " - train acc: " << std::setprecision(4)
                      << accuracy * 100 << "%"
                      << " - remaining epochs: " << (epochs - epoch - 1) << std::endl;
        }

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
        int correct = 0;
        for (size_t i = 0; i < images.size(); ++i) {
            if (predict(images[i]) == labels[i]) {
                correct++;
            }
        }
        return static_cast<double>(correct) / images.size();
    }

    std::vector<std::vector<int>> computeConfusionMatrix(
        const std::vector<std::vector<double>>& images,
        const std::vector<int>& labels) override {
        std::vector<std::vector<int>> matrix(NUM_CLASSES, std::vector<int>(NUM_CLASSES, 0));
        for (size_t i = 0; i < images.size(); ++i) {
            int predicted = predict(images[i]);
            matrix[labels[i]][predicted]++;
        }
        return matrix;
    }

    void computeMetrics(const std::vector<std::vector<int>>& matrix,
                       std::vector<double>& precision,
                       std::vector<double>& recall,
                       std::vector<double>& f1) override {
        precision.resize(NUM_CLASSES);
        recall.resize(NUM_CLASSES);
        f1.resize(NUM_CLASSES);
        for (int c = 0; c < NUM_CLASSES; ++c) {
            int tp = matrix[c][c];
            int fp = 0, fn = 0;
            for (int i = 0; i < NUM_CLASSES; ++i) {
                if (i != c) {
                    fp += matrix[i][c];
                    fn += matrix[c][i];
                }
            }
            precision[c] = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
            recall[c] = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
            f1[c] = (precision[c] + recall[c] > 0) ?
                    2 * precision[c] * recall[c] / (precision[c] + recall[c]) : 0.0;
        }
    }

    double computeAUC(const std::vector<std::vector<double>>& images,
                     const std::vector<int>& labels,
                     std::string& rocFilename) override {
        rocFilename = getROCDataFilename();
        std::vector<double> scores;
        std::vector<int> binaryLabels;
        scores.reserve(images.size());
        binaryLabels.reserve(images.size());

        for (size_t i = 0; i < images.size(); ++i) {
            auto prob = predictProba(images[i]);
            scores.push_back(prob[0]);
            binaryLabels.push_back(labels[i] == 0 ? 1 : 0);
        }

        return writeRocDataAndComputeAUC(scores, binaryLabels, rocFilename);
    }

    std::string getROCDataFilename() const override {
        return "ROC/rf_roc_data.txt";
    }

    void evaluateAndReport(const std::vector<std::vector<double>>& images,
                          const std::vector<int>& labels) override {
        double acc = evaluate(images, labels);
        auto matrix = computeConfusionMatrix(images, labels);
        std::vector<double> precision, recall, f1;
        computeMetrics(matrix, precision, recall, f1);

        double avgPrecision = 0.0, avgRecall = 0.0, avgF1 = 0.0;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            avgPrecision += precision[c];
            avgRecall += recall[c];
            avgF1 += f1[c];
        }
        avgPrecision /= NUM_CLASSES;
        avgRecall /= NUM_CLASSES;
        avgF1 /= NUM_CLASSES;

        std::cout << "\n========== Random Forest Test Results ==========" << std::endl;
        std::cout << "Accuracy:  " << std::fixed << std::setprecision(4) << acc * 100 << "%" << std::endl;
        std::cout << "Precision: " << std::setprecision(4) << avgPrecision * 100 << "%" << std::endl;
        std::cout << "Recall:    " << std::setprecision(4) << avgRecall * 100 << "%" << std::endl;
        std::cout << "F1 Score:  " << std::setprecision(4) << avgF1 * 100 << "%" << std::endl;
        std::cout << "========================================" << std::endl;
    }
};

// ==================== K杩戦偦妯″瀷瀹炵幇 ====================

class KNNModel : public IClassificationModel {
public:
    virtual std::string getStructureDescription() const override { return "Input(784) -> Prototype Library -> k=5 Distance Vote"; }
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
        return "KNN classifier using mini-batch prototype construction and distance-based voting";
    }

    void initWeights() override {
        prototypes.clear();
        prototypeLabels.clear();
        rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
        std::cout << "KNN prototype store initialized" << std::endl;
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
        std::cout << "  Initialization: mini-batch prototype construction | max prototypes: " << KNN_MAX_PROTOTYPES << std::endl;

        prototypes.clear();
        prototypeLabels.clear();
        prototypes.reserve(std::min(static_cast<size_t>(KNN_MAX_PROTOTYPES), images.size()));
        prototypeLabels.reserve(std::min(static_cast<size_t>(KNN_MAX_PROTOTYPES), images.size()));

        std::vector<size_t> indices(images.size());
        std::iota(indices.begin(), indices.end(), 0);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::shuffle(indices.begin(), indices.end(), rng);

            double totalLoss = 0.0;
            int correctPredictions = 0;
            size_t processedSamples = 0;

            for (size_t batchStart = 0; batchStart < indices.size(); batchStart += KNN_BATCH_SIZE) {
                size_t batchEnd = std::min(batchStart + KNN_BATCH_SIZE, indices.size());
                size_t batchSize = batchEnd - batchStart;

                for (size_t position = batchStart; position < batchEnd; ++position) {
                    size_t index = indices[position];
                    const auto& image = images[index];
                    int label = labels[index];

                    if (!prototypes.empty()) {
                        auto probabilities = predictProba(image);
                        int predictedClass = std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()));
                        totalLoss += -std::log(std::max(probabilities[label], 1e-7));
                        if (predictedClass == label) {
                            correctPredictions++;
                        }
                    }

                    addPrototype(image, label);
                    processedSamples++;
                }

                ProgressBar::show(batchEnd, indices.size(), "Building prototypes");
            }

            ProgressBar::finish();
            double accuracy = processedSamples > 0 ? static_cast<double>(correctPredictions) / processedSamples : 0.0;
            double averageLoss = processedSamples > 0 ? totalLoss / processedSamples : 0.0;
            std::cout << "Epoch " << epoch + 1 << "/" << epochs
                      << " - avg loss: " << std::fixed << std::setprecision(6) << averageLoss
                      << " - train acc: " << std::setprecision(4)
                      << accuracy * 100 << "%"
                      << " - remaining epochs: " << (epochs - epoch - 1) << std::endl;
        }

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
        int correct = 0;
        for (size_t i = 0; i < images.size(); ++i) {
            if (predict(images[i]) == labels[i]) {
                correct++;
            }
        }
        return static_cast<double>(correct) / images.size();
    }

    std::vector<std::vector<int>> computeConfusionMatrix(
        const std::vector<std::vector<double>>& images,
        const std::vector<int>& labels) override {
        std::vector<std::vector<int>> matrix(NUM_CLASSES, std::vector<int>(NUM_CLASSES, 0));
        for (size_t i = 0; i < images.size(); ++i) {
            int predicted = predict(images[i]);
            matrix[labels[i]][predicted]++;
        }
        return matrix;
    }

    void computeMetrics(const std::vector<std::vector<int>>& matrix,
                       std::vector<double>& precision,
                       std::vector<double>& recall,
                       std::vector<double>& f1) override {
        precision.resize(NUM_CLASSES);
        recall.resize(NUM_CLASSES);
        f1.resize(NUM_CLASSES);
        for (int c = 0; c < NUM_CLASSES; ++c) {
            int tp = matrix[c][c];
            int fp = 0, fn = 0;
            for (int i = 0; i < NUM_CLASSES; ++i) {
                if (i != c) {
                    fp += matrix[i][c];
                    fn += matrix[c][i];
                }
            }
            precision[c] = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
            recall[c] = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
            f1[c] = (precision[c] + recall[c] > 0) ?
                    2 * precision[c] * recall[c] / (precision[c] + recall[c]) : 0.0;
        }
    }

    double computeAUC(const std::vector<std::vector<double>>& images,
                     const std::vector<int>& labels,
                     std::string& rocFilename) override {
        rocFilename = getROCDataFilename();
        std::vector<double> scores;
        std::vector<int> binaryLabels;
        scores.reserve(images.size());
        binaryLabels.reserve(images.size());

        for (size_t i = 0; i < images.size(); ++i) {
            auto prob = predictProba(images[i]);
            scores.push_back(prob[0]);
            binaryLabels.push_back(labels[i] == 0 ? 1 : 0);
        }

        return writeRocDataAndComputeAUC(scores, binaryLabels, rocFilename);
    }

    std::string getROCDataFilename() const override {
        return "ROC/knn_roc_data.txt";
    }

    void evaluateAndReport(const std::vector<std::vector<double>>& images,
                          const std::vector<int>& labels) override {
        double acc = evaluate(images, labels);
        auto matrix = computeConfusionMatrix(images, labels);
        std::vector<double> precision, recall, f1;
        computeMetrics(matrix, precision, recall, f1);

        double avgPrecision = 0.0, avgRecall = 0.0, avgF1 = 0.0;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            avgPrecision += precision[c];
            avgRecall += recall[c];
            avgF1 += f1[c];
        }
        avgPrecision /= NUM_CLASSES;
        avgRecall /= NUM_CLASSES;
        avgF1 /= NUM_CLASSES;

        std::cout << "\n========== KNN Test Results ==========" << std::endl;
        std::cout << "Accuracy:  " << std::fixed << std::setprecision(4) << acc * 100 << "%" << std::endl;
        std::cout << "Precision: " << std::setprecision(4) << avgPrecision * 100 << "%" << std::endl;
        std::cout << "Recall:    " << std::setprecision(4) << avgRecall * 100 << "%" << std::endl;
        std::cout << "F1 Score:  " << std::setprecision(4) << avgF1 * 100 << "%" << std::endl;
        std::cout << "========================================" << std::endl;
    }
};

// ==================== 閫昏緫鍥炲綊妯″瀷瀹炵幇 ====================

class LogisticRegressionModel : public IClassificationModel {
public:
    virtual std::string getStructureDescription() const override { return "Input(784) -> Linear Projection -> Softmax(10)"; }
    virtual Metrics testAndGetMetrics(const std::vector<std::vector<double>>& images, const std::vector<int>& labels) override { return buildMetricsSnapshot(*this, images, labels); }

private:
    static const int NUM_CLASSES = 10;
    static const int INPUT_SIZE = 784;
    
    // Softmax鍥炲綊鐨勬潈閲嶇煩闃靛拰鍋忕疆鍚戦噺
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
        
        // Xavier鍒濆鍖栵細std = sqrt(2.0 / (input_size + output_size))
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
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double totalLoss = 0.0;
            int correctPredictions = 0;
            
            // 澶勭悊 mini-batch
            for (size_t batchStart = 0; batchStart < images.size(); batchStart += LR_BATCH_SIZE) {
                size_t batchEnd = std::min(batchStart + LR_BATCH_SIZE, images.size());
                int batchSize = batchEnd - batchStart;
                
                ProgressBar::show(batchEnd, images.size(), "Processing data");
                
                // 绱Н姊害
                std::vector<std::vector<double>> dWeights(NUM_CLASSES, 
                    std::vector<double>(INPUT_SIZE, 0.0));
                std::vector<double> dBiases(NUM_CLASSES, 0.0);
                
                for (size_t idx = batchStart; idx < batchEnd; ++idx) {
                    const auto& image = images[idx];
                    int label = labels[idx];
                    
                    // ==================== 鍓嶅悜浼犳挱 ====================
                    // 璁＄畻logits: z = W*x + b
                    std::vector<double> logits(NUM_CLASSES, 0.0);
                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        logits[c] = biases[c];
                        for (int i = 0; i < INPUT_SIZE; ++i) {
                            logits[c] += weights[c][i] * image[i];
                        }
                    }
                    
                    // Softmax婵€娲?
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
                    
                    // ==================== 鎹熷け璁＄畻锛堜氦鍙夌喌锛?===================
                    double loss = -std::log(std::max(probs[label], 1e-7));
                    totalLoss += loss;
                    
                    // 妫€鏌ラ娴嬫纭€?
                    int predictedClass = std::distance(probs.begin(), 
                                                       std::max_element(probs.begin(), probs.end()));
                    if (predictedClass == label) correctPredictions++;
                    
                    // ==================== 鍙嶅悜浼犳挱 ====================
                    // 瀵逛簬Softmax + CrossEntropy锛屾搴?= probs - one_hot(label)
                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        double gradient = probs[c] - (c == label ? 1.0 : 0.0);
                        
                        // 绱Н鏉冮噸姊害锛堝寘鎷琇2姝ｅ垯鍖栭」锛?
                        for (int i = 0; i < INPUT_SIZE; ++i) {
                            dWeights[c][i] += gradient * image[i] + 
                                            LR_REGULARIZATION_L2 * weights[c][i];
                        }
                        
                        // 绱Н鍋忕疆姊害
                        dBiases[c] += gradient;
                    }
                }
                
                // ==================== Mini-batch 姊害骞冲潎鍜屾潈閲嶆洿鏂?====================
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
        int correct = 0;
        for (size_t i = 0; i < images.size(); ++i) {
            if (predict(images[i]) == labels[i]) correct++;
        }
        return static_cast<double>(correct) / images.size();
    }

    std::vector<std::vector<int>> computeConfusionMatrix(
        const std::vector<std::vector<double>>& images,
        const std::vector<int>& labels) override {
        std::vector<std::vector<int>> matrix(NUM_CLASSES, std::vector<int>(NUM_CLASSES, 0));
        for (size_t i = 0; i < images.size(); ++i) {
            int predicted = predict(images[i]);
            matrix[labels[i]][predicted]++;
        }
        return matrix;
    }

    void computeMetrics(const std::vector<std::vector<int>>& matrix,
                       std::vector<double>& precision,
                       std::vector<double>& recall,
                       std::vector<double>& f1) override {
        precision.resize(NUM_CLASSES);
        recall.resize(NUM_CLASSES);
        f1.resize(NUM_CLASSES);
        for (int c = 0; c < NUM_CLASSES; ++c) {
            int tp = matrix[c][c];
            int fp = 0, fn = 0;
            for (int i = 0; i < NUM_CLASSES; ++i) {
                if (i != c) {
                    fp += matrix[i][c];
                    fn += matrix[c][i];
                }
            }
            precision[c] = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
            recall[c] = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
            f1[c] = (precision[c] + recall[c] > 0) ?
                    2 * precision[c] * recall[c] / (precision[c] + recall[c]) : 0.0;
        }
    }

    double computeAUC(const std::vector<std::vector<double>>& images,
                     const std::vector<int>& labels,
                     std::string& rocFilename) override {
        rocFilename = getROCDataFilename();
        std::vector<double> scores;
        std::vector<int> binaryLabels;
        scores.reserve(images.size());
        binaryLabels.reserve(images.size());

        for (size_t i = 0; i < images.size(); ++i) {
            auto prob = predictProba(images[i]);
            scores.push_back(prob[0]);
            binaryLabels.push_back(labels[i] == 0 ? 1 : 0);
        }

        return writeRocDataAndComputeAUC(scores, binaryLabels, rocFilename);
    }

    std::string getROCDataFilename() const override {
        return "ROC/lr_roc_data.txt";
    }

    void evaluateAndReport(const std::vector<std::vector<double>>& images,
                          const std::vector<int>& labels) override {
        double acc = evaluate(images, labels);
        auto matrix = computeConfusionMatrix(images, labels);
        std::vector<double> precision, recall, f1;
        computeMetrics(matrix, precision, recall, f1);
        
        double avgPrecision = 0.0, avgRecall = 0.0, avgF1 = 0.0;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            avgPrecision += precision[c];
            avgRecall += recall[c];
            avgF1 += f1[c];
        }
        avgPrecision /= NUM_CLASSES;
        avgRecall /= NUM_CLASSES;
        avgF1 /= NUM_CLASSES;
        
        std::cout << "\n======== Logistic Regression Test Results ========" << std::endl;
        std::cout << "Accuracy:  " << std::fixed << std::setprecision(4) << acc * 100 << "%" << std::endl;
        std::cout << "Precision: " << std::setprecision(4) << avgPrecision * 100 << "%" << std::endl;
        std::cout << "Recall:    " << std::setprecision(4) << avgRecall * 100 << "%" << std::endl;
        std::cout << "F1 Score:  " << std::setprecision(4) << avgF1 * 100 << "%" << std::endl;
        std::cout << "========================================" << std::endl;
    }
};

// ==================== 鑿滃崟绯荤粺 ====================

class MenuSystem {
public:
    /**
     * 鏄剧ず涓昏彍鍗?
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
     * 鏄剧ず妯″瀷閫夋嫨鑿滃崟
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
     * 鏄剧ず璁粌鍙傛暟鑿滃崟
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
     * 鏄剧ず娴嬭瘯缁撴灉鑿滃崟
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
     * 鏄剧ず缁樺浘閫夐」鑿滃崟
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

// ==================== 妯″瀷宸ュ巶瀹炵幇 ====================

// 鐜板湪鎵€鏈夋ā鍨嬬被閮藉凡瀹氫箟锛屽彲浠ュ畨鍏ㄥ湴瀹炵幇宸ュ巶鍑芥暟
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
        {CNN, "Convolutional Neural Network (CNN) - image feature model"},
        {LOGISTIC_REGRESSION, "Logistic Regression (LR) - softmax linear classifier"},
        {RANDOM_FOREST, "Random Forest (RF) - tree ensemble"},
        {KNN, "K-Nearest Neighbors (KNN) - prototype voting"}
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


