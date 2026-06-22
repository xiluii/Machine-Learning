#include "AppBackend.h"

#include <QtCharts/QBarCategoryAxis>
#include <QtCharts/QBarSeries>
#include <QtCharts/QBarSet>
#include <QtCharts/QChart>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>

#include <QApplication>
#include <QComboBox>
#include <QCursor>
#include <QFile>
#include <QFrame>
#include <QGraphicsOpacityEffect>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QMainWindow>
#include <QMouseEvent>
#include <QPainter>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPropertyAnimation>
#include <QPushButton>
#include <QRadioButton>
#include <QPainterPath>
#include <QResizeEvent>
#include <QScrollArea>
#include <QSpinBox>
#include <QSplitter>
#include <QTabWidget>
#include <QTimer>
#include <QToolTip>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrent>

#include <cmath>

class LoadingSpinner : public QWidget {
public:
    explicit LoadingSpinner(QWidget* parent = nullptr) : QWidget(parent) {
        setFixedSize(64, 64);
        timer_.setInterval(16);
        connect(&timer_, &QTimer::timeout, this, [this]() { angle_ = (angle_ + 8) % 360; update(); });
    }
    void start() { show(); timer_.start(); }
    void stop() { timer_.stop(); hide(); }
protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.translate(rect().center());
        painter.rotate(angle_);
        QPen track(QColor(255,255,255,35), 6);
        track.setCapStyle(Qt::RoundCap);
        painter.setPen(track);
        painter.drawArc(QRect(-20, -20, 40, 40), 0, 360 * 16);
        QConicalGradient gradient(0,0,-90);
        gradient.setColorAt(0.0, QColor("#7c3aed"));
        gradient.setColorAt(0.4, QColor("#22d3ee"));
        gradient.setColorAt(1.0, QColor(124,58,237,30));
        QPen pen(QBrush(gradient), 6);
        pen.setCapStyle(Qt::RoundCap);
        painter.setPen(pen);
        painter.drawArc(QRect(-20, -20, 40, 40), 0, 220 * 16);
    }
private:
    QTimer timer_;
    int angle_ = 0;
};

class MetricCard : public QFrame {
public:
    explicit MetricCard(const QString& title, QWidget* parent = nullptr) : QFrame(parent) {
        setObjectName("panel");
        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(16, 14, 16, 14);
        auto* titleLabel = new QLabel(title, this);
        titleLabel->setStyleSheet("color:#94a3b8;");
        valueLabel_ = new QLabel("--", this);
        valueLabel_->setStyleSheet("font-size:24px;font-weight:700;color:#f8fafc;");
        subtitleLabel_ = new QLabel(this);
        subtitleLabel_->setStyleSheet("color:#64748b;font-size:11px;");
        layout->addWidget(titleLabel);
        layout->addWidget(valueLabel_);
        layout->addWidget(subtitleLabel_);
    }
    void setValue(const QString& v) { valueLabel_->setText(v); }
    void setSubtitle(const QString& v) { subtitleLabel_->setText(v); }
    void setDetailToolTip(const QString& v) { setToolTip(v); valueLabel_->setToolTip(v); subtitleLabel_->setToolTip(v); }
private:
    QLabel* valueLabel_ = nullptr;
    QLabel* subtitleLabel_ = nullptr;
};

class ConfusionMatrixWidget : public QWidget {
public:
    explicit ConfusionMatrixWidget(QWidget* parent = nullptr) : QWidget(parent) { setMinimumHeight(320); }
    void setMatrix(const std::vector<std::vector<int>>& matrix) { matrix_ = matrix; update(); }
protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.fillRect(rect(), QColor("#101726"));
        QRectF frame = rect().adjusted(24,24,-24,-24);
        painter.setPen(QColor(255,255,255,24));
        painter.setBrush(QColor(255,255,255,10));
        painter.drawRoundedRect(frame, 14, 14);
        if (matrix_.empty()) {
            painter.setPen(QColor("#94a3b8"));
            painter.drawText(frame, Qt::AlignCenter, "No confusion matrix");
            return;
        }
        int rows = static_cast<int>(matrix_.size());
        int cols = static_cast<int>(matrix_.front().size());
        qreal labelBand = 32.0;
        QRectF gridRect = frame.adjusted(labelBand, labelBand, -8, -8);
        qreal cellW = gridRect.width() / cols;
        qreal cellH = gridRect.height() / rows;
        int maxValue = 1;
        for (const auto& row : matrix_) for (int value : row) maxValue = std::max(maxValue, value);
        painter.setPen(QColor("#cbd5e1"));
        painter.drawText(QRectF(frame.left(), frame.top() - 2, frame.width(), 24), Qt::AlignCenter, "Confusion Matrix");
        for (int c = 0; c < cols; ++c) {
            painter.setPen(QColor("#94a3b8"));
            painter.drawText(QRectF(gridRect.left() + c * cellW, frame.top() + 2, cellW, labelBand - 8), Qt::AlignCenter, QString::number(c));
        }
        for (int r = 0; r < rows; ++r) {
            painter.setPen(QColor("#94a3b8"));
            painter.drawText(QRectF(frame.left() + 4, gridRect.top() + r * cellH, labelBand - 8, cellH), Qt::AlignCenter, QString::number(r));
        }
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                double ratio = static_cast<double>(matrix_[r][c]) / maxValue;
                QColor fill = QColor::fromRgbF(0.12 + ratio * 0.33, 0.18 + ratio * 0.44, 0.30 + ratio * 0.55, 0.92);
                QRectF cell(gridRect.left() + c * cellW, gridRect.top() + r * cellH, cellW - 2, cellH - 2);
                painter.fillRect(cell, fill);
                painter.setPen(QColor(255,255,255,22));
                painter.drawRect(cell);
                painter.setPen(QColor("#f8fafc"));
                painter.drawText(cell, Qt::AlignCenter, QString::number(matrix_[r][c]));
            }
        }
    }
private:
    std::vector<std::vector<int>> matrix_;
};

class StructureDiagramWidget : public QWidget {
public:
    explicit StructureDiagramWidget(QWidget* parent = nullptr) : QWidget(parent) {
        setMinimumHeight(440);
        setMouseTracking(true);
    }
    void setSpec(const ModelVisualSpec& spec) {
        spec_ = spec;
        hoverRegions_.clear();
        update();
    }
protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.fillRect(rect(), QColor("#181a1e"));
        QRectF frame = rect().adjusted(24,24,-24,-24);
        painter.setPen(QColor("#353941"));
        painter.setBrush(QColor("#202225"));
        painter.drawRoundedRect(frame, 18, 18);
        painter.setPen(QColor("#e7e9ec"));
        QFont f = painter.font(); f.setPointSize(14); f.setBold(true); painter.setFont(f);
        painter.drawText(frame.adjusted(20,16,-20,0), QString::fromStdString(spec_.displayName));
        f.setPointSize(9); f.setBold(false); painter.setFont(f);
        painter.setPen(QColor("#9ea3aa"));
        painter.drawText(frame.adjusted(20,44,-20,0), QString::fromStdString(spec_.structureText));
        QRectF paperRect = frame.adjusted(20, 78, -20, -22);
        painter.setPen(QColor("#42464e"));
        painter.setBrush(QColor("#f6f4ef"));
        painter.drawRoundedRect(paperRect, 12, 12);

        hoverRegions_.clear();
        switch (spec_.modelType) {
        case ModelFactory::CNN:
            drawCnnDiagram(painter, paperRect);
            break;
        case ModelFactory::FCNN:
            drawFcnnDiagram(painter, paperRect);
            break;
        case ModelFactory::SVM:
            drawSvmDiagram(painter, paperRect);
            break;
        case ModelFactory::LOGISTIC_REGRESSION:
            drawLogisticDiagram(painter, paperRect);
            break;
        case ModelFactory::RANDOM_FOREST:
            drawForestDiagram(painter, paperRect);
            break;
        case ModelFactory::KNN:
            drawKnnDiagram(painter, paperRect);
            break;
        default:
            drawFallbackDiagram(painter, paperRect);
            break;
        }
    }
    void mouseMoveEvent(QMouseEvent* event) override {
        for (const auto& region : hoverRegions_) {
            if (region.first.contains(event->position())) {
                QString detail;
                const int i = region.second;
                if (i < static_cast<int>(spec_.stageDetails.size())) {
                    detail = QString::fromStdString(spec_.stageDetails[i]);
                } else {
                    detail = QString::fromStdString(spec_.stages[i]);
                }
                QToolTip::showText(event->globalPosition().toPoint(), detail, this);
                return;
            }
        }
        QToolTip::hideText();
        QWidget::mouseMoveEvent(event);
    }
    void leaveEvent(QEvent* event) override {
        QToolTip::hideText();
        QWidget::leaveEvent(event);
    }
private:
    void addHoverRegion(const QRectF& rect, int index) {
        hoverRegions_.push_back({rect, index});
    }

    QColor accentColor(int index, const QString& fallback = "#7c6ee6") const {
        if (index >= 0 && index < static_cast<int>(spec_.stageAccents.size())) {
            return QColor(QString::fromStdString(spec_.stageAccents[index]));
        }
        return QColor(fallback);
    }

    void drawArrow(QPainter& painter, const QPointF& start, const QPointF& end, const QColor& color = QColor("#585d66")) {
        painter.save();
        painter.setPen(QPen(color, 2.0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
        painter.drawLine(start, end);
        const qreal arrowSize = 8.0;
        painter.drawLine(end, QPointF(end.x() - arrowSize, end.y() - arrowSize * 0.55));
        painter.drawLine(end, QPointF(end.x() - arrowSize, end.y() + arrowSize * 0.55));
        painter.restore();
    }

    void drawStageLabel(QPainter& painter, const QRectF& rect, const QString& title, const QString& meta) {
        painter.save();
        QRectF titleRect(rect.left() - 6, rect.bottom() + 10, rect.width() + 12, 18);
        QRectF metaRect(rect.left() - 6, rect.bottom() + 28, rect.width() + 12, 16);
        painter.setPen(QColor("#2f3137"));
        QFont titleFont = painter.font();
        titleFont.setPointSize(10);
        titleFont.setBold(true);
        painter.setFont(titleFont);
        painter.drawText(titleRect, Qt::AlignHCenter, title);
        QFont metaFont = painter.font();
        metaFont.setPointSize(8);
        metaFont.setBold(false);
        painter.setFont(metaFont);
        painter.setPen(QColor("#666b74"));
        painter.drawText(metaRect, Qt::AlignHCenter, meta);
        painter.restore();
    }

    void drawFeatureStack(QPainter& painter, const QRectF& anchor, const QSizeF& size, int depth, const QColor& color, int stageIndex, const QString& topLabel) {
        painter.save();
        QRectF hoverRect;
        for (int i = 0; i < depth; ++i) {
            QRectF plane(anchor.left() + i * 12.0, anchor.top() - i * 10.0, size.width(), size.height());
            QLinearGradient gradient(plane.topLeft(), plane.bottomRight());
            gradient.setColorAt(0.0, QColor("#ffffff"));
            gradient.setColorAt(1.0, color.lighter(170));
            painter.setPen(QPen(QColor("#8f949d"), 2));
            painter.setBrush(gradient);
            painter.drawRect(plane);
            hoverRect = hoverRect.united(plane);
        }
        addHoverRegion(hoverRect.adjusted(-4, -4, 4, 4), stageIndex);
        painter.setPen(QColor("#2f3137"));
        QFont labelFont = painter.font();
        labelFont.setPointSize(9);
        labelFont.setBold(true);
        painter.setFont(labelFont);
        painter.drawText(QRectF(anchor.left() - 6, anchor.top() - 40, size.width() + depth * 12.0 + 12, 22), Qt::AlignHCenter, topLabel);
        painter.restore();
    }

    void drawDenseBar(QPainter& painter, const QRectF& rect, const QColor& color, int stageIndex, const QString& label, const QString& meta) {
        painter.save();
        QLinearGradient gradient(rect.topLeft(), rect.bottomLeft());
        gradient.setColorAt(0.0, QColor("#ffffff"));
        gradient.setColorAt(1.0, color.lighter(165));
        painter.setPen(QPen(QColor("#8f949d"), 2));
        painter.setBrush(gradient);
        painter.drawRoundedRect(rect, 4, 4);
        addHoverRegion(rect.adjusted(-4, -4, 4, 4), stageIndex);
        drawStageLabel(painter, rect, label, meta);
        painter.restore();
    }

    void drawNodeLayer(QPainter& painter, const QRectF& bounds, int nodes, const QColor& color, int stageIndex, const QString& label, const QString& meta) {
        painter.save();
        const qreal spacing = bounds.height() / (nodes + 1);
        const qreal radius = std::min<qreal>(11.0, spacing * 0.26);
        for (int i = 0; i < nodes; ++i) {
            QPointF center(bounds.center().x(), bounds.top() + spacing * (i + 1));
            painter.setPen(QPen(QColor("#858a93"), 1.5));
            painter.setBrush(color);
            painter.drawEllipse(center, radius, radius);
        }
        addHoverRegion(bounds.adjusted(-6, -6, 6, 6), stageIndex);
        drawStageLabel(painter, bounds, label, meta);
        painter.restore();
    }

    void connectLayers(QPainter& painter, const QRectF& left, const QRectF& right, int leftNodes, int rightNodes, const QColor& lineColor) {
        painter.save();
        painter.setPen(QPen(lineColor, 0.85));
        const qreal leftSpacing = left.height() / (leftNodes + 1);
        const qreal rightSpacing = right.height() / (rightNodes + 1);
        for (int i = 0; i < leftNodes; ++i) {
            QPointF from(left.center().x(), left.top() + leftSpacing * (i + 1));
            for (int j = 0; j < rightNodes; ++j) {
                QPointF to(right.center().x(), right.top() + rightSpacing * (j + 1));
                painter.drawLine(from, to);
            }
        }
        painter.restore();
    }

    void drawTreeGlyph(QPainter& painter, const QPointF& root, qreal scale, const QColor& edgeColor, const QColor& activeColor) {
        painter.save();
        painter.setPen(QPen(edgeColor, 1.6, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
        QPointF a = root;
        QPointF b(root.x() - 28 * scale, root.y() + 34 * scale);
        QPointF c(root.x() + 28 * scale, root.y() + 34 * scale);
        QPointF d(root.x() - 42 * scale, root.y() + 68 * scale);
        QPointF e(root.x() - 10 * scale, root.y() + 68 * scale);
        QPointF f(root.x() + 10 * scale, root.y() + 68 * scale);
        QPointF g(root.x() + 42 * scale, root.y() + 68 * scale);
        painter.drawLine(a, b); painter.drawLine(a, c);
        painter.drawLine(b, d); painter.drawLine(b, e);
        painter.drawLine(c, f); painter.drawLine(c, g);
        painter.setBrush(Qt::white);
        for (const auto& p : {d, e, f, g}) painter.drawEllipse(p, 5.0 * scale, 5.0 * scale);
        painter.setBrush(activeColor);
        for (const auto& p : {a, c, f}) painter.drawEllipse(p, 5.0 * scale, 5.0 * scale);
        painter.setBrush(Qt::white);
        painter.drawEllipse(b, 5.0 * scale, 5.0 * scale);
        painter.restore();
    }

    void drawCnnDiagram(QPainter& painter, const QRectF& paperRect) {
        painter.save();
        const qreal w = paperRect.width();
        const qreal h = paperRect.height();
        const qreal startX = paperRect.left() + w * 0.055;
        const qreal inputBaseY = paperRect.top() + h * 0.22;
        const qreal stage2Y = paperRect.top() + h * 0.40;
        const qreal stage3Y = paperRect.top() + h * 0.53;
        const qreal stage4Y = paperRect.top() + h * 0.63;

        drawFeatureStack(painter, QRectF(startX, inputBaseY, 0, 0), QSizeF(w * 0.105, h * 0.55), 2, QColor("#d9dde3"), 0, "28 x 28");
        drawFeatureStack(painter, QRectF(paperRect.left() + w * 0.24, stage2Y, 0, 0), QSizeF(w * 0.082, h * 0.43), 2, accentColor(1), 1, "14 x 14");
        drawFeatureStack(painter, QRectF(paperRect.left() + w * 0.39, stage3Y, 0, 0), QSizeF(w * 0.062, h * 0.31), 3, accentColor(1).lighter(115), 1, "7 x 7");
        drawFeatureStack(painter, QRectF(paperRect.left() + w * 0.56, stage4Y, 0, 0), QSizeF(w * 0.038, h * 0.18), 2, accentColor(2), 2, "head");

        QRectF fc1(paperRect.left() + w * 0.72, paperRect.top() + h * 0.18, w * 0.034, h * 0.58);
        QRectF fc2(paperRect.left() + w * 0.80, paperRect.top() + h * 0.22, w * 0.034, h * 0.54);
        QRectF out(paperRect.left() + w * 0.91, paperRect.top() + h * 0.39, w * 0.028, h * 0.33);
        drawDenseBar(painter, fc1, QColor("#ece8ff"), 2, "FC 256", "classifier");
        drawDenseBar(painter, fc2, QColor("#ece8ff"), 2, "FC 128", "projection");
        drawDenseBar(painter, out, QColor("#ecfff8"), 3, "Softmax", "10 classes");

        drawArrow(painter, QPointF(paperRect.left() + w * 0.19, paperRect.top() + h * 0.55), QPointF(paperRect.left() + w * 0.225, paperRect.top() + h * 0.55));
        drawArrow(painter, QPointF(paperRect.left() + w * 0.35, paperRect.top() + h * 0.60), QPointF(paperRect.left() + w * 0.385, paperRect.top() + h * 0.60));
        drawArrow(painter, QPointF(paperRect.left() + w * 0.52, paperRect.top() + h * 0.66), QPointF(paperRect.left() + w * 0.555, paperRect.top() + h * 0.66));
        drawArrow(painter, QPointF(paperRect.left() + w * 0.63, paperRect.top() + h * 0.66), QPointF(fc1.left() - w * 0.018, paperRect.top() + h * 0.66));
        drawArrow(painter, QPointF(fc1.right() + 10, fc1.center().y()), QPointF(fc2.left() - 10, fc2.center().y()));
        drawArrow(painter, QPointF(fc2.right() + 10, fc2.center().y()), QPointF(out.left() - 10, out.center().y()));

        painter.setPen(QColor("#2f3137"));
        QFont serif("Times New Roman", 10);
        painter.setFont(serif);
        painter.drawText(QRectF(fc1.left() - w * 0.01, paperRect.top() + h * 0.04, w * 0.18, h * 0.12), Qt::AlignLeft | Qt::AlignVCenter, "fully\nconnected");
        painter.restore();
    }

    void drawFcnnDiagram(QPainter& painter, const QRectF& paperRect) {
        painter.save();
        QRectF inputLayer(paperRect.left() + 88, paperRect.top() + 44, 56, paperRect.height() - 138);
        QRectF hidden1(paperRect.left() + 282, paperRect.top() + 62, 70, paperRect.height() - 172);
        QRectF hidden2(paperRect.left() + 486, paperRect.top() + 88, 70, paperRect.height() - 214);
        QRectF outputLayer(paperRect.right() - 132, paperRect.top() + 110, 56, paperRect.height() - 250);

        connectLayers(painter, inputLayer, hidden1, 8, 6, QColor(104, 89, 180, 55));
        connectLayers(painter, hidden1, hidden2, 6, 5, QColor(104, 89, 180, 70));
        connectLayers(painter, hidden2, outputLayer, 5, 4, QColor(104, 89, 180, 85));
        drawNodeLayer(painter, inputLayer, 8, accentColor(0), 0, "Input", "784 features");
        drawNodeLayer(painter, hidden1, 6, accentColor(1), 1, "Dense 256", "ReLU");
        drawNodeLayer(painter, hidden2, 5, accentColor(2), 2, "Dense 128", "ReLU");
        drawNodeLayer(painter, outputLayer, 4, accentColor(3), 3, "Softmax", "10 logits");
        painter.restore();
    }

    void drawSvmDiagram(QPainter& painter, const QRectF& paperRect) {
        painter.save();
        const qreal w = paperRect.width();
        const qreal h = paperRect.height();
        QRectF chartRect(paperRect.left() + w * 0.06, paperRect.top() + h * 0.12, w * 0.40, h * 0.62);
        QRectF projRect(paperRect.left() + w * 0.59, paperRect.top() + h * 0.12, w * 0.28, h * 0.62);

        painter.setPen(QPen(QColor("#8f949d"), 1.3));
        painter.setBrush(QColor("#fbfbfd"));
        painter.drawRect(chartRect);
        painter.drawRect(projRect);
        addHoverRegion(chartRect.adjusted(-4, -4, 4, 4), 0);
        addHoverRegion(projRect.adjusted(-4, -4, 4, 4), 1);

        painter.setPen(QPen(QColor("#d6d9de"), 1.0));
        for (int i = 1; i < 8; ++i) {
            qreal x = chartRect.left() + i * chartRect.width() / 8.0;
            qreal y = chartRect.top() + i * chartRect.height() / 8.0;
            painter.drawLine(QPointF(x, chartRect.top()), QPointF(x, chartRect.bottom()));
            painter.drawLine(QPointF(chartRect.left(), y), QPointF(chartRect.right(), y));
        }

        const QPointF center(chartRect.center().x(), chartRect.center().y());
        painter.setPen(QPen(QColor("#1f9d55"), 2.4));
        painter.setBrush(Qt::NoBrush);
        painter.drawEllipse(center, chartRect.width() * 0.20, chartRect.width() * 0.20);

        painter.setPen(Qt::NoPen);
        for (int i = 0; i < 48; ++i) {
            qreal angle = (6.2831853 / 48.0) * i;
            qreal radius = chartRect.width() * (0.28 + (i % 5) * 0.012);
            QPointF p(center.x() + std::cos(angle) * radius, center.y() + std::sin(angle) * radius);
            painter.setBrush(QColor("#8b1e3f"));
            painter.drawEllipse(p, 3.0, 3.0);
        }
        for (int i = 0; i < 26; ++i) {
            qreal angle = (6.2831853 / 26.0) * i;
            qreal radius = chartRect.width() * (0.12 + (i % 4) * 0.01);
            QPointF p(center.x() + std::cos(angle) * radius, center.y() + std::sin(angle) * radius);
            painter.setBrush(QColor("#262f7a"));
            painter.drawEllipse(p, 3.0, 3.0);
        }

        painter.setPen(QPen(QColor("#e5e7eb"), 1.0));
        for (int i = 1; i < 6; ++i) {
            qreal x = projRect.left() + i * projRect.width() / 6.0;
            qreal y = projRect.top() + i * projRect.height() / 6.0;
            painter.drawLine(QPointF(x, projRect.top()), QPointF(x, projRect.bottom()));
            painter.drawLine(QPointF(projRect.left(), y), QPointF(projRect.right(), y));
        }
        painter.setPen(QPen(QColor("#15803d"), 2.2));
        QPointF p1(projRect.left() + projRect.width() * 0.10, projRect.top() + projRect.height() * 0.65);
        QPointF p2(projRect.right() - projRect.width() * 0.08, projRect.top() + projRect.height() * 0.58);
        QPointF p3(projRect.center().x(), projRect.top() + projRect.height() * 0.50);
        painter.drawLine(p1, p2);
        painter.drawLine(QPointF(p1.x(), p1.y() - 18), QPointF(p2.x(), p2.y() - 18));
        painter.drawLine(QPointF(p1.x(), p1.y() + 18), QPointF(p2.x(), p2.y() + 18));
        painter.setPen(Qt::NoPen);
        for (int i = 0; i < 36; ++i) {
            qreal x = projRect.left() + projRect.width() * (0.12 + (i % 9) * 0.09);
            qreal y = projRect.top() + projRect.height() * (0.16 + ((i * 7) % 10) * 0.065);
            painter.setBrush(i % 3 == 0 ? QColor("#262f7a") : QColor("#8b1e3f"));
            painter.drawEllipse(QPointF(x, y), 2.6, 2.6);
        }

        drawArrow(painter, QPointF(chartRect.right() + w * 0.02, chartRect.center().y()), QPointF(projRect.left() - w * 0.02, projRect.center().y()));
        drawStageLabel(painter, chartRect, "Input space", "support boundaries");
        drawStageLabel(painter, projRect, "Projected margin", "linear separator");
        painter.restore();
    }

    void drawLogisticDiagram(QPainter& painter, const QRectF& paperRect) {
        painter.save();
        QRectF inputRect(paperRect.left() + 72, paperRect.top() + 120, 132, 108);
        QRectF linearRect(paperRect.center().x() - 48, paperRect.top() + 96, 96, 158);
        QRectF outputRect(paperRect.right() - 178, paperRect.top() + 108, 112, 132);

        drawDenseBar(painter, inputRect, QColor("#f1eee7"), 0, "Input", "784 features");
        drawDenseBar(painter, linearRect, accentColor(1), 1, "Linear", "W x + b");
        drawDenseBar(painter, outputRect, accentColor(2), 2, "Softmax", "10 probabilities");
        drawArrow(painter, QPointF(inputRect.right() + 20, inputRect.center().y()), QPointF(linearRect.left() - 18, linearRect.center().y()));
        drawArrow(painter, QPointF(linearRect.right() + 18, linearRect.center().y()), QPointF(outputRect.left() - 18, outputRect.center().y()));
        painter.restore();
    }

    void drawForestDiagram(QPainter& painter, const QRectF& paperRect) {
        painter.save();
        const qreal w = paperRect.width();
        const qreal h = paperRect.height();
        QRectF sourceRect(paperRect.left() + w * 0.32, paperRect.top() + h * 0.04, w * 0.16, h * 0.10);
        painter.setPen(QPen(QColor("#5b7f1a"), 2));
        painter.setBrush(QColor("#75b527"));
        painter.drawRoundedRect(sourceRect, 4, 4);
        painter.setPen(Qt::white);
        QFont bold = painter.font();
        bold.setBold(true);
        bold.setPointSize(12);
        painter.setFont(bold);
        painter.drawText(sourceRect, Qt::AlignCenter, "Digits");
        addHoverRegion(sourceRect.adjusted(-4, -4, 4, 4), 0);

        const qreal treeY = paperRect.top() + h * 0.28;
        const qreal x1 = paperRect.left() + w * 0.16;
        const qreal x2 = paperRect.left() + w * 0.49;
        const qreal x3 = paperRect.left() + w * 0.80;
        painter.setPen(QPen(QColor("#1f2937"), 1.8));
        drawArrow(painter, QPointF(sourceRect.center().x(), sourceRect.bottom()), QPointF(x2, treeY - 26), QColor("#1f2937"));
        drawArrow(painter, QPointF(sourceRect.left() + 4, sourceRect.bottom()), QPointF(x1, treeY - 26), QColor("#1f2937"));
        drawArrow(painter, QPointF(sourceRect.right() - 4, sourceRect.bottom()), QPointF(x3, treeY - 26), QColor("#1f2937"));
        painter.setPen(QColor("#2f3137"));
        painter.setFont(QFont("Segoe UI", 10, QFont::Bold));
        painter.drawText(QRectF(x1 - 40, treeY - 46, 80, 20), Qt::AlignCenter, "Tree 1");
        painter.drawText(QRectF(x2 - 40, treeY - 46, 80, 20), Qt::AlignCenter, "Tree 2");
        painter.drawText(QRectF(x3 - 40, treeY - 46, 80, 20), Qt::AlignCenter, "Tree n");
        drawTreeGlyph(painter, QPointF(x1, treeY), 0.95, QColor("#9ca3af"), QColor("#77c043"));
        drawTreeGlyph(painter, QPointF(x2, treeY), 0.95, QColor("#9ca3af"), QColor("#77c043"));
        drawTreeGlyph(painter, QPointF(x3, treeY), 0.95, QColor("#9ca3af"), QColor("#77c043"));
        addHoverRegion(QRectF(x1 - 58, treeY - 10, 116, 120), 2);
        addHoverRegion(QRectF(x2 - 58, treeY - 10, 116, 120), 2);
        addHoverRegion(QRectF(x3 - 58, treeY - 10, 116, 120), 2);

        QRectF label1(x1 - 36, treeY + 112, 72, 42);
        QRectF label2(x2 - 36, treeY + 112, 72, 42);
        QRectF label3(x3 - 42, treeY + 112, 84, 42);
        painter.setPen(QPen(QColor("#f2c94c"), 5));
        painter.setBrush(QColor("#fffaf0"));
        painter.drawEllipse(label1.adjusted(-10, -8, 10, 8));
        painter.drawEllipse(label2.adjusted(-10, -8, 10, 8));
        painter.setPen(QPen(QColor("#a3e635"), 1.5));
        painter.setBrush(QColor("#fef3c7"));
        painter.drawRoundedRect(label1, 4, 4);
        painter.drawRoundedRect(label2, 4, 4);
        painter.drawRoundedRect(label3, 4, 4);
        painter.setPen(QColor("#1f2937"));
        painter.setFont(QFont("Segoe UI", 9, QFont::Bold));
        painter.drawText(label1, Qt::AlignCenter, "2");
        painter.drawText(label2, Qt::AlignCenter, "2");
        painter.drawText(label3, Qt::AlignCenter, "7");

        QRectF voteRect(paperRect.left() + w * 0.28, paperRect.top() + h * 0.74, w * 0.36, h * 0.10);
        painter.setPen(QPen(QColor("#5b7f1a"), 2));
        painter.setBrush(QColor("#7cc840"));
        painter.drawRoundedRect(voteRect, voteRect.height() * 0.5, voteRect.height() * 0.5);
        painter.setPen(Qt::white);
        painter.setFont(QFont("Segoe UI", 11, QFont::Bold));
        painter.drawText(voteRect, Qt::AlignCenter, "Majority Voting");
        addHoverRegion(voteRect.adjusted(-4, -4, 4, 4), 3);

        drawArrow(painter, QPointF(label1.center().x(), label1.bottom()), QPointF(voteRect.left() + w * 0.05, voteRect.top()), QColor("#1f2937"));
        drawArrow(painter, QPointF(label2.center().x(), label2.bottom()), QPointF(voteRect.center().x(), voteRect.top()), QColor("#1f2937"));
        drawArrow(painter, QPointF(label3.center().x(), label3.bottom()), QPointF(voteRect.right() - w * 0.05, voteRect.top()), QColor("#1f2937"));

        QRectF finalRect(paperRect.left() + w * 0.46, paperRect.top() + h * 0.88, w * 0.10, h * 0.07);
        painter.setPen(QPen(QColor("#8b1024"), 1.6));
        painter.setBrush(QColor("#a60f2d"));
        painter.drawRoundedRect(finalRect, 4, 4);
        painter.setPen(Qt::white);
        painter.drawText(finalRect, Qt::AlignCenter, "2");
        drawArrow(painter, QPointF(voteRect.center().x(), voteRect.bottom()), QPointF(finalRect.center().x(), finalRect.top()), QColor("#1f2937"));
        painter.restore();
    }

    void drawKnnDiagram(QPainter& painter, const QRectF& paperRect) {
        painter.save();
        painter.fillRect(paperRect, QColor("#2f3130"));
        painter.setPen(QPen(QColor(255, 255, 255, 18), 1.0));
        for (int i = 0; i <= 14; ++i) {
            qreal x = paperRect.left() + i * paperRect.width() / 14.0;
            painter.drawLine(QPointF(x, paperRect.top()), QPointF(x, paperRect.bottom()));
        }
        for (int i = 0; i <= 10; ++i) {
            qreal y = paperRect.top() + i * paperRect.height() / 10.0;
            painter.drawLine(QPointF(paperRect.left(), y), QPointF(paperRect.right(), y));
        }
        painter.setPen(QPen(QColor("#f97316"), 2));
        painter.drawLine(QPointF(paperRect.left(), paperRect.bottom()), QPointF(paperRect.left(), paperRect.top() + 6));
        painter.drawLine(QPointF(paperRect.left(), paperRect.bottom()), QPointF(paperRect.right() - 6, paperRect.bottom()));

        auto mapPoint = [&](qreal x, qreal y) {
            return QPointF(paperRect.left() + paperRect.width() * x, paperRect.top() + paperRect.height() * (1.0 - y));
        };
        std::vector<QPointF> leftCluster = {
            mapPoint(0.07, 0.14), mapPoint(0.09, 0.10), mapPoint(0.15, 0.23), mapPoint(0.18, 0.34),
            mapPoint(0.19, 0.46), mapPoint(0.23, 0.23), mapPoint(0.30, 0.52), mapPoint(0.31, 0.29),
            mapPoint(0.24, 0.62), mapPoint(0.40, 0.46)
        };
        std::vector<QPointF> rightCluster = {
            mapPoint(0.57, 0.69), mapPoint(0.65, 0.61), mapPoint(0.70, 0.52), mapPoint(0.72, 0.88),
            mapPoint(0.79, 0.80), mapPoint(0.85, 0.24), mapPoint(0.92, 0.30), mapPoint(0.86, 0.71),
            mapPoint(0.86, 0.36)
        };
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor("#fb923c"));
        for (const auto& p : leftCluster) painter.drawEllipse(p, 6.0, 6.0);
        painter.setBrush(QColor("#ff6f4d"));
        for (const auto& p : rightCluster) painter.drawEllipse(p, 7.0, 7.0);

        QPointF query = mapPoint(0.43, 0.55);
        painter.setBrush(QColor("#4fc3f7"));
        painter.drawEllipse(query, 6.5, 6.5);
        QRectF knnCircle(query.x() - paperRect.width() * 0.215, query.y() - paperRect.width() * 0.215, paperRect.width() * 0.43, paperRect.width() * 0.43);
        painter.setPen(QPen(QColor("#f8ea16"), 3));
        painter.setBrush(Qt::NoBrush);
        painter.drawEllipse(knnCircle);
        addHoverRegion(knnCircle.adjusted(-6, -6, 6, 6), 1);
        addHoverRegion(QRectF(query.x() - 12, query.y() - 12, 24, 24), 0);
        drawStageLabel(painter, QRectF(paperRect.left() + 10, paperRect.bottom() - 56, paperRect.width() * 0.32, 18), "Local neighborhood", "distance vote");
        painter.restore();
    }

    void drawFallbackDiagram(QPainter& painter, const QRectF& paperRect) {
        painter.save();
        if (spec_.stages.empty()) {
            painter.restore();
            return;
        }
        const int count = static_cast<int>(spec_.stages.size());
        const qreal gap = 18.0;
        const qreal width = (paperRect.width() - 40 - gap * (count - 1)) / count;
        const qreal top = paperRect.top() + 96;
        for (int i = 0; i < count; ++i) {
            QRectF box(paperRect.left() + 20 + i * (width + gap), top + (i % 2 == 0 ? 0 : 28), width, 110);
            drawDenseBar(painter, box, accentColor(i), i, QString::fromStdString(spec_.stages[i]), QString("Stage %1").arg(i + 1));
            if (i + 1 < count) {
                drawArrow(painter, QPointF(box.right() + 6, box.center().y()), QPointF(box.right() + gap - 6, box.center().y()));
            }
        }
        painter.restore();
    }

    ModelVisualSpec spec_;
    std::vector<std::pair<QRectF, int>> hoverRegions_;
};

class MainWindow : public QMainWindow {
public:
    MainWindow() {
        setWindowTitle("MNIST Model Workbench");
        resize(1440, 920);

        auto* central = new QWidget(this);
        auto* root = new QVBoxLayout(central);
        root->setContentsMargins(20,20,20,20);
        root->setSpacing(14);

        auto* title = new QLabel("MNIST Workbench", central);
        title->setStyleSheet("font-size:28px;font-weight:700;color:#f8fafc;");
        auto* subtitle = new QLabel("Native C++ training, testing and model visualization desktop application.", central);
        subtitle->setStyleSheet("color:#94a3b8;");

        auto* tabs = new QTabWidget(central);
        tabs->addTab(buildTrainPage(), "Train");
        tabs->addTab(buildTestPage(), "Test");
        tabs->addTab(buildVisualizePage(), "Visualize");

        root->addWidget(title);
        root->addWidget(subtitle);
        root->addWidget(tabs, 1);
        setCentralWidget(central);
    }

private:
    QWidget* buildTrainPage() {
        auto* page = new QWidget(this);
        auto* root = new QVBoxLayout(page);
        root->setSpacing(14);

        auto* card = new QFrame(page);
        card->setObjectName("panel");
        auto* form = new QGridLayout(card);
        form->setContentsMargins(18,18,18,18);
        form->setHorizontalSpacing(14);
        form->setVerticalSpacing(12);

        auto* modelCombo = new QComboBox(card);
        for (const auto& item : AppBackend::availableModels()) {
            modelCombo->addItem(QString::fromStdString(ModelFactory::getModelTypeName(item.first) + " | " + item.second),
                                static_cast<int>(item.first));
        }
        auto* epochsSpin = new QSpinBox(card);
        epochsSpin->setRange(1, 200);
        auto* createNew = new QRadioButton("Create new parameter state", card);
        auto* continueExisting = new QRadioButton("Continue existing parameter file", card);
        createNew->setChecked(true);
        auto* paramLabel = new QLabel(card);
        auto* descLabel = new QLabel(card);
        descLabel->setWordWrap(true);
        descLabel->setStyleSheet("color:#a5b4cc;");
        auto* progress = new QProgressBar(card);
        progress->setTextVisible(false);
        progress->setRange(0,1);
        progress->setValue(0);
        auto* runButton = new QPushButton("Start Training", card);
        runButton->setObjectName("primary");

        form->addWidget(new QLabel("Model", card), 0, 0);
        form->addWidget(modelCombo, 0, 1, 1, 2);
        form->addWidget(new QLabel("Epochs", card), 1, 0);
        form->addWidget(epochsSpin, 1, 1);
        form->addWidget(runButton, 1, 2);
        form->addWidget(createNew, 2, 0, 1, 2);
        form->addWidget(continueExisting, 2, 2);
        form->addWidget(paramLabel, 3, 0, 1, 3);
        form->addWidget(descLabel, 4, 0, 1, 3);
        form->addWidget(progress, 5, 0, 1, 3);

        auto* outputCard = new QFrame(page);
        outputCard->setObjectName("panel");
        auto* outputLayout = new QVBoxLayout(outputCard);
        outputLayout->setContentsMargins(18,18,18,18);
        auto* output = new QPlainTextEdit(outputCard);
        output->setReadOnly(true);
        outputLayout->addWidget(new QLabel("Execution Summary", outputCard));
        outputLayout->addWidget(output);

        auto updateSummary = [modelCombo, epochsSpin, paramLabel, descLabel]() {
            auto type = static_cast<ModelFactory::ModelType>(modelCombo->currentData().toInt());
            epochsSpin->setValue(AppBackend::defaultEpochs(type));
            paramLabel->setText(QString("Parameter file: %1").arg(QString::fromStdString(AppBackend::defaultParamFile(type))));
            for (const auto& spec : AppBackend::visualSpecs()) {
                if (spec.modelType == type) {
                    descLabel->setText(QString::fromStdString(spec.description + "\n" + spec.structureText));
                    break;
                }
            }
        };
        updateSummary();
        QObject::connect(modelCombo, &QComboBox::currentIndexChanged, page, updateSummary);

        auto* watcher = new QFutureWatcher<TrainResult>(page);
        QObject::connect(runButton, &QPushButton::clicked, page, [=]() {
            if (watcher->isRunning()) return;
            TrainOptions options;
            options.modelType = static_cast<ModelFactory::ModelType>(modelCombo->currentData().toInt());
            options.epochs = epochsSpin->value();
            options.continueFromExisting = continueExisting->isChecked();
            progress->setRange(0,0);
            runButton->setEnabled(false);
            output->appendPlainText(QString("Starting %1 ...").arg(modelCombo->currentText()));
            watcher->setFuture(QtConcurrent::run([options]() {
                AppBackend backend;
                return backend.trainModel(options);
            }));
        });
        QObject::connect(watcher, &QFutureWatcher<TrainResult>::finished, page, [=]() {
            progress->setRange(0,1);
            progress->setValue(1);
            runButton->setEnabled(true);
            output->appendPlainText(QString::fromStdString(watcher->result().summary));
            output->appendPlainText("");
        });

        root->addWidget(card);
        root->addWidget(outputCard, 1);
        return page;
    }

    QWidget* buildTestPage() {
        auto* page = new QWidget(this);
        auto* outer = new QVBoxLayout(page);
        outer->setContentsMargins(0,0,0,0);
        auto* scroll = new QScrollArea(page);
        scroll->setWidgetResizable(true);
        scroll->setFrameShape(QFrame::NoFrame);
        scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        auto* content = new QWidget(scroll);
        auto* root = new QVBoxLayout(content);
        root->setSpacing(14);

        auto* controls = new QFrame(content);
        controls->setObjectName("panel");
        auto* controlsLayout = new QHBoxLayout(controls);
        controlsLayout->setContentsMargins(18,18,18,18);

        auto* modelCombo = new QComboBox(controls);
        for (const auto& item : AppBackend::availableModels()) {
            modelCombo->addItem(QString::fromStdString(ModelFactory::getModelTypeName(item.first) + " | " + item.second),
                                static_cast<int>(item.first));
        }
        auto* paramLabel = new QLabel(controls);
        auto* runButton = new QPushButton("Run Test", controls);
        runButton->setObjectName("primary");
        controlsLayout->addWidget(new QLabel("Model", controls));
        controlsLayout->addWidget(modelCombo, 1);
        controlsLayout->addWidget(paramLabel, 2);
        controlsLayout->addWidget(runButton);

        auto* cardsRow = new QHBoxLayout();
        auto* accCard = new MetricCard("Accuracy", content);
        auto* precCard = new MetricCard("Precision", content);
        auto* recCard = new MetricCard("Recall", content);
        auto* f1Card = new MetricCard("F1", content);
        auto* aucCard = new MetricCard("AUC", content);
        cardsRow->addWidget(accCard);
        cardsRow->addWidget(precCard);
        cardsRow->addWidget(recCard);
        cardsRow->addWidget(f1Card);
        cardsRow->addWidget(aucCard);

        auto* grid = new QGridLayout();
        auto* rocPanel = new QFrame(content);
        rocPanel->setObjectName("panel");
        auto* rocLayout = new QVBoxLayout(rocPanel);
        rocLayout->setContentsMargins(18,18,18,18);
        auto* rocChartView = new QChartView(rocPanel);
        rocLayout->addWidget(new QLabel("ROC Curve", rocPanel));
        rocLayout->addWidget(rocChartView);

        auto* matrixPanel = new QFrame(content);
        matrixPanel->setObjectName("panel");
        auto* matrixLayout = new QVBoxLayout(matrixPanel);
        matrixLayout->setContentsMargins(18,18,18,18);
        auto* matrixWidget = new ConfusionMatrixWidget(matrixPanel);
        matrixLayout->addWidget(new QLabel("Confusion Matrix", matrixPanel));
        matrixLayout->addWidget(matrixWidget);

        auto* metricsPanel = new QFrame(content);
        metricsPanel->setObjectName("panel");
        auto* metricsLayout = new QVBoxLayout(metricsPanel);
        metricsLayout->setContentsMargins(18,18,18,18);
        auto* metricsChartView = new QChartView(metricsPanel);
        metricsChartView->setMinimumHeight(500);
        metricsLayout->addWidget(new QLabel("Per-Class Metrics", metricsPanel));
        metricsLayout->addWidget(metricsChartView);

        grid->addWidget(rocPanel, 0, 0);
        grid->addWidget(matrixPanel, 0, 1);
        grid->addWidget(metricsPanel, 1, 0, 1, 2);

        auto* overlay = new QWidget(page);
        overlay->hide();
        overlay->setStyleSheet("background: rgba(2,6,23,0.78); border-radius:18px;");
        auto* overlayLayout = new QVBoxLayout(overlay);
        overlayLayout->setAlignment(Qt::AlignCenter);
        auto* spinner = new LoadingSpinner(overlay);
        auto* overlayLabel = new QLabel("Testing model in native C++ pipeline ...", overlay);
        overlayLabel->setStyleSheet("font-size:15px;font-weight:600;color:#e2e8f0;");
        overlayLayout->addWidget(spinner, 0, Qt::AlignCenter);
        overlayLayout->addWidget(overlayLabel, 0, Qt::AlignCenter);

        page->installEventFilter(page);
        auto updateOverlay = [page, overlay]() { overlay->setGeometry(page->rect()); };
        updateOverlay();

        auto updateParam = [modelCombo, paramLabel]() {
            auto type = static_cast<ModelFactory::ModelType>(modelCombo->currentData().toInt());
            paramLabel->setText(QString::fromStdString(AppBackend::defaultParamFile(type)));
        };
        updateParam();
        QObject::connect(modelCombo, &QComboBox::currentIndexChanged, page, updateParam);

        auto* watcher = new QFutureWatcher<TestResult>(page);
        QObject::connect(runButton, &QPushButton::clicked, page, [=]() {
            if (watcher->isRunning()) return;
            TestOptions options;
            options.modelType = static_cast<ModelFactory::ModelType>(modelCombo->currentData().toInt());
            updateOverlay();
            overlay->show();
            overlay->raise();
            spinner->start();
            runButton->setEnabled(false);
            watcher->setFuture(QtConcurrent::run([options]() {
                AppBackend backend;
                return backend.testModel(options);
            }));
        });
        QObject::connect(watcher, &QFutureWatcher<TestResult>::finished, page, [=]() {
            spinner->stop();
            overlay->hide();
            runButton->setEnabled(true);
            const TestResult result = watcher->result();
            auto percent = [](double value) { return QString::number(value * 100.0, 'f', 2) + "%"; };
            accCard->setValue(percent(result.metrics.accuracy));
            accCard->setSubtitle("overall");
            accCard->setDetailToolTip(QString("Overall classification accuracy.\nValue: %1").arg(percent(result.metrics.accuracy)));
            precCard->setValue(percent(result.metrics.precision));
            precCard->setSubtitle("macro average");
            precCard->setDetailToolTip(QString("Macro-averaged precision across 10 classes.\nValue: %1").arg(percent(result.metrics.precision)));
            recCard->setValue(percent(result.metrics.recall));
            recCard->setSubtitle("macro average");
            recCard->setDetailToolTip(QString("Macro-averaged recall across 10 classes.\nValue: %1").arg(percent(result.metrics.recall)));
            f1Card->setValue(percent(result.metrics.f1));
            f1Card->setSubtitle("macro average");
            f1Card->setDetailToolTip(QString("Macro-averaged F1 score across 10 classes.\nValue: %1").arg(percent(result.metrics.f1)));
            aucCard->setValue(QString::number(result.metrics.auc, 'f', 4));
            aucCard->setSubtitle(QString::fromStdString(result.rocDataFile));
            aucCard->setDetailToolTip(QString("Binary ROC AUC for class 0 vs all other classes.\nValue: %1\nSource: %2")
                                      .arg(QString::number(result.metrics.auc, 'f', 4),
                                           QString::fromStdString(result.rocDataFile)));

            auto* rocSeries = new QLineSeries();
            for (const auto& point : result.rocCurve) rocSeries->append(point.falsePositiveRate, point.truePositiveRate);
            auto* diagonal = new QLineSeries();
            diagonal->append(0.0, 0.0); diagonal->append(1.0, 1.0);
            auto* rocChart = new QChart();
            rocChart->setBackgroundVisible(false);
            rocChart->setPlotAreaBackgroundVisible(false);
            rocChart->setTitle(QString::fromStdString(result.modelName));
            rocChart->addSeries(rocSeries);
            rocChart->addSeries(diagonal);
            auto* axisX = new QValueAxis(); axisX->setRange(0.0, 1.0); axisX->setTitleText("False Positive Rate");
            auto* axisY = new QValueAxis(); axisY->setRange(0.0, 1.0); axisY->setTitleText("True Positive Rate");
            rocChart->addAxis(axisX, Qt::AlignBottom);
            rocChart->addAxis(axisY, Qt::AlignLeft);
            rocSeries->attachAxis(axisX); rocSeries->attachAxis(axisY);
            diagonal->attachAxis(axisX); diagonal->attachAxis(axisY);
            rocChartView->setChart(rocChart);
            rocChartView->setToolTip(QString("ROC curve for class 0 vs all other classes.\nAUC: %1")
                                     .arg(QString::number(result.metrics.auc, 'f', 4)));

            matrixWidget->setMatrix(result.metrics.confusionMatrix);
            matrixWidget->setToolTip("Confusion matrix. Rows are true labels, columns are predicted labels.");

            auto* precisionSet = new QBarSet("Precision");
            auto* recallSet = new QBarSet("Recall");
            auto* f1Set = new QBarSet("F1");
            QStringList labels;
            for (const auto& item : result.classMetrics) {
                *precisionSet << item.precision;
                *recallSet << item.recall;
                *f1Set << item.f1;
                labels << QString::number(item.label);
            }
            auto* series = new QBarSeries();
            series->append(precisionSet); series->append(recallSet); series->append(f1Set);
            series->setLabelsVisible(true);
            series->setLabelsFormat("@value");
            series->setLabelsPosition(QAbstractBarSeries::LabelsOutsideEnd);
            QObject::connect(series, &QBarSeries::hovered, metricsChartView, [=](bool status, int index, QBarSet* barset) {
                if (!status || index < 0 || index >= static_cast<int>(result.classMetrics.size()) || barset == nullptr) {
                    QToolTip::hideText();
                    return;
                }
                const auto& item = result.classMetrics[index];
                QToolTip::showText(QCursor::pos(),
                                   QString("Class %1\n%2: %3\nPrecision: %4\nRecall: %5\nF1: %6")
                                       .arg(item.label)
                                       .arg(barset->label())
                                       .arg(QString::number(barset->at(index), 'f', 4))
                                       .arg(QString::number(item.precision, 'f', 4))
                                       .arg(QString::number(item.recall, 'f', 4))
                                       .arg(QString::number(item.f1, 'f', 4)),
                                   metricsChartView);
            });
            auto* metricsChart = new QChart();
            metricsChart->setBackgroundVisible(false);
            metricsChart->setPlotAreaBackgroundVisible(false);
            metricsChart->setTitle("Per-Class Precision / Recall / F1");
            metricsChart->addSeries(series);
            auto* categoryAxis = new QBarCategoryAxis(); categoryAxis->append(labels);
            auto* valueAxis = new QValueAxis(); valueAxis->setRange(0.0, 1.0); valueAxis->setTitleText("Score"); valueAxis->setTickCount(6);
            metricsChart->addAxis(categoryAxis, Qt::AlignBottom);
            metricsChart->addAxis(valueAxis, Qt::AlignLeft);
            series->attachAxis(categoryAxis); series->attachAxis(valueAxis);
            metricsChart->setMargins(QMargins(18, 18, 18, 18));
            metricsChartView->setChart(metricsChart);
            metricsChartView->setToolTip("Hover over bars to inspect per-class precision, recall and F1.");
        });

        root->addWidget(controls);
        root->addLayout(cardsRow);
        root->addLayout(grid, 1);
        scroll->setWidget(content);
        outer->addWidget(scroll);
        return page;
    }

    QWidget* buildVisualizePage() {
        auto* page = new QWidget(this);
        auto* root = new QVBoxLayout(page);

        auto* splitter = new QSplitter(page);
        auto* listPanel = new QFrame(splitter);
        listPanel->setObjectName("panel");
        auto* listLayout = new QVBoxLayout(listPanel);
        listLayout->setContentsMargins(18,18,18,18);
        listLayout->addWidget(new QLabel("Implemented Models", listPanel));
        auto* modelList = new QListWidget(listPanel);
        listLayout->addWidget(modelList);

        auto* detailPanel = new QFrame(splitter);
        detailPanel->setObjectName("panel");
        auto* detailLayout = new QVBoxLayout(detailPanel);
        detailLayout->setContentsMargins(18,18,18,18);
        auto* descLabel = new QLabel(detailPanel);
        descLabel->setStyleSheet("color:#a5b4cc;");
        descLabel->setWordWrap(true);
        auto* diagram = new StructureDiagramWidget(detailPanel);
        auto* notesLabel = new QLabel(detailPanel);
        notesLabel->setStyleSheet("color:#a5b4cc;");
        notesLabel->setTextFormat(Qt::RichText);
        notesLabel->setWordWrap(true);
        auto* hoverHint = new QLabel("Hover over each stage to inspect detailed structural information.", detailPanel);
        hoverHint->setStyleSheet("color:#8f949c;font-size:12px;");
        detailLayout->addWidget(descLabel);
        detailLayout->addWidget(diagram, 1);
        detailLayout->addWidget(hoverHint);
        detailLayout->addWidget(notesLabel);

        const auto specs = AppBackend::visualSpecs();
        for (const auto& spec : specs) {
            modelList->addItem(QString::fromStdString(spec.displayName));
        }
        QObject::connect(modelList, &QListWidget::currentRowChanged, page, [=](int row) {
            if (row < 0 || row >= specs.size()) return;
            const auto& spec = specs[row];
            descLabel->setText(QString::fromStdString(spec.description + "\n" + spec.structureText));
            diagram->setSpec(spec);
            QString notes;
            for (const auto& note : spec.notes) notes += QString("&bull; %1<br>").arg(QString::fromStdString(note));
            notesLabel->setText(notes);
            auto* effect = new QGraphicsOpacityEffect(diagram);
            diagram->setGraphicsEffect(effect);
            auto* animation = new QPropertyAnimation(effect, "opacity", diagram);
            animation->setDuration(240);
            animation->setStartValue(0.35);
            animation->setEndValue(1.0);
            animation->start(QAbstractAnimation::DeleteWhenStopped);
        });
        if (!specs.empty()) modelList->setCurrentRow(0);

        splitter->addWidget(listPanel);
        splitter->addWidget(detailPanel);
        splitter->setSizes({260, 900});
        root->addWidget(splitter);
        return page;
    }
};

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    QFile styleFile(":/app_styles.qss");
    if (styleFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        app.setStyleSheet(styleFile.readAll());
    }
    MainWindow window;
    window.show();
    return app.exec();
}
