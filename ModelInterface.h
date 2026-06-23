#ifndef MODEL_INTERFACE_H
#define MODEL_INTERFACE_H

#include <string>
#include <vector>
#include <memory>
#include <functional>

/**
 * ================ 模型接口 ================
 * 所有机器学习模型都应继承此接口
 * 
 * 使用说明：
 * 1. 创建新模型类，继承 IClassificationModel
 * 2. 实现所有纯虚函数
 * 3. 在 ModelFactory::createModel() 中注册新模型
 * 4. 在 main() 的菜单中添加选项
 */


using ProgressCallback = std::function<void(int currentEpoch,
                                            int totalEpochs,
                                            double progress,
                                            double etaSeconds)>;

struct Metrics {
    double accuracy = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
    double auc = 0.0;
    double microPrecision = 0.0;
    double microRecall = 0.0;
    double microF1 = 0.0;
    std::vector<double> perClassPrecision;
    std::vector<double> perClassRecall;
    std::vector<double> perClassF1;
    std::vector<std::vector<int>> confusionMatrix;
};
class IClassificationModel {
public:
    ProgressCallback progressCallback;
    virtual void setProgressCallback(ProgressCallback cb) { progressCallback = cb; }
    virtual std::string getStructureDescription() const = 0;
    virtual Metrics testAndGetMetrics(const std::vector<std::vector<double>>& images, const std::vector<int>& labels) = 0;

public:
    virtual ~IClassificationModel() = default;

    // ============ 必须实现的核心方法 ============
    
    /**
     * 模型名称（用于菜单显示）
     * @return 模型的人读名称，如 "Support Vector Machine (SVM)"
     */
    virtual std::string getName() const = 0;

    /**
     * 模型描述
     * @return 模型的简要描述
     */
    virtual std::string getDescription() const = 0;

    /**
     * 初始化权重或参数
     * 使用随机初始化
     */
    virtual void initWeights() = 0;

    /**
     * 从文件加载已保存的参数
     * @param filename 参数文件路径
     * @return 成功返回true，失败返回false
     */
    virtual bool loadParams(const std::string& filename) = 0;

    /**
     * 保存训练好的参数到文件
     * @param filename 输出文件路径
     */
    virtual void saveParams(const std::string& filename) = 0;

    /**
     * 训练模型
     * @param images 训练图像数据，每个向量是一个展平的图像
     * @param labels 对应的标签
     * @param epochs 训练轮数
     */
    virtual void train(const std::vector<std::vector<double>>& images,
                      const std::vector<int>& labels,
                      int epochs) = 0;

    /**
     * 单个样本预测
     * @param image 输入的单个图像（展平向量）
     * @return 预测的类别标签 (0-9)
     */
    virtual int predict(const std::vector<double>& image) = 0;

    /**
     * 获取单个样本的预测概率分布
     * @param image 输入图像
     * @return 各类别的概率分布向量 (大小为10，对应数字0-9)
     */
    virtual std::vector<double> predictProba(const std::vector<double>& image) = 0;

    /**
     * 批量评估准确率
     * @param images 测试图像集合
     * @param labels 对应的真实标签
     * @return 准确率 (0.0-1.0)
     */
    virtual double evaluate(const std::vector<std::vector<double>>& images,
                           const std::vector<int>& labels) = 0;

    // ============ 评估指标计算 ============
    
    /**
     * 计算混淆矩阵
     * @param images 测试图像
     * @param labels 真实标签
     * @return 10x10的混淆矩阵，matrix[i][j]表示真实为i预测为j的样本数
     */
    virtual std::vector<std::vector<int>> computeConfusionMatrix(
        const std::vector<std::vector<double>>& images,
        const std::vector<int>& labels) = 0;

    /**
     * 计算精确率、召回率、F1分数
     * @param confusionMatrix 混淆矩阵
     * @param precision 输出：各类别的精确率
     * @param recall 输出：各类别的召回率
     * @param f1 输出：各类别的F1分数
     */
    virtual void computeMetrics(
        const std::vector<std::vector<int>>& confusionMatrix,
        std::vector<double>& precision,
        std::vector<double>& recall,
        std::vector<double>& f1) = 0;

    /**
     * 计算AUC并生成ROC数据
     * @param images 测试图像
     * @param labels 真实标签
     * @param rocFilename 输出ROC数据的文件名
     * @return 宏平均AUC值 (0.0-1.0)
     */
    virtual double computeAUC(
        const std::vector<std::vector<double>>& images,
        const std::vector<int>& labels,
        std::string& rocFilename) = 0;

    /**
     * 获取模型特定的ROC数据文件名
     * 每个模型应返回不同的文件名以区分（如 svm_roc_data.txt, fcnn_roc_data.txt）
     * @return ROC数据文件名
     */
    virtual std::string getROCDataFilename() const = 0;

    /**
     * 完整的测试和评估流程
     * 自动计算所有指标并输出结果
     * @param images 测试图像
     * @param labels 真实标签
     */
    virtual void evaluateAndReport(
        const std::vector<std::vector<double>>& images,
        const std::vector<int>& labels) = 0;

protected:
    void reportProgress(int current, int total, double progress, double etaSeconds) const {
        if (progressCallback) {
            progressCallback(current, total, progress, etaSeconds);
        }
    }
};

// ============ 模型工厂 ============
/**
 * 模型工厂类
 * 负责创建和管理模型实例
 * 
 * 扩展方法：在任何地方调用 ModelFactory::createModel(modelType)
 */
class ModelFactory {
public:
    enum ModelType {
        SVM = 1,
        FCNN = 2,
        CNN = 3,
        LOGISTIC_REGRESSION = 4,
        RANDOM_FOREST = 5,
        KNN = 6
    };

    static std::shared_ptr<IClassificationModel> createModel(ModelType type);
    static std::vector<std::pair<int, std::string>> getAvailableModels();
    static std::string getModelTypeName(ModelType type);
};

#endif // MODEL_INTERFACE_H
