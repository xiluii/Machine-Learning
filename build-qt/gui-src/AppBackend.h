#ifndef APP_BACKEND_H
#define APP_BACKEND_H

#include "../../ModelInterface.h"

#include <memory>
#include <string>
#include <functional>
#include <vector>

struct RocPoint {
    double falsePositiveRate = 0.0;
    double truePositiveRate = 0.0;
};

struct ClassMetricDetail {
    int label = 0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
};

struct TrainOptions {
    ModelFactory::ModelType modelType = ModelFactory::SVM;
    int epochs = 1;
    bool continueFromExisting = false;
    ProgressCallback progressCallback;
};

struct TrainResult {
    std::string modelName;
    std::string parameterFile;
    int epochs = 0;
    double durationSeconds = 0.0;
    std::string summary;
};

struct TestOptions {
    ModelFactory::ModelType modelType = ModelFactory::SVM;
};

struct TestResult {
    std::string modelName;
    std::string parameterFile;
    std::string rocDataFile;
    Metrics metrics;
    std::vector<ClassMetricDetail> classMetrics;
    std::vector<RocPoint> rocCurve;
};

struct ModelVisualSpec {
    ModelFactory::ModelType modelType = ModelFactory::SVM;
    std::string displayName;
    std::string description;
    std::string structureText;
    std::vector<std::string> stages;
    std::vector<std::string> stageDetails;
    std::vector<std::string> notes;
    std::vector<std::string> stageAccents;
};

class AppBackend {
public:
    static std::vector<std::pair<ModelFactory::ModelType, std::string>> availableModels();
    static std::string defaultParamFile(ModelFactory::ModelType type);
    static int defaultEpochs(ModelFactory::ModelType type);
    static std::vector<ModelVisualSpec> visualSpecs();

    TrainResult trainModel(const TrainOptions& options);
    TestResult testModel(const TestOptions& options);

private:
    struct Dataset {
        std::vector<std::vector<double>> images;
        std::vector<int> labels;
    };

    Dataset loadTrainDataset();
    Dataset loadTestDataset();
    std::shared_ptr<IClassificationModel> createModel(ModelFactory::ModelType type) const;
    std::string resolvePath(const std::string& relative) const;
    std::vector<RocPoint> readRocCurve(const std::string& path) const;
};

#endif
