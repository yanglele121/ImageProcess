#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include<QLabel>
#include <QMainWindow>
#include<opencv2/opencv.hpp>
#include<highgui.hpp>
using namespace cv;
QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    QImage Src;
    QImage fitpixmap;
    QImage pixStatic;
    QImage fitpixmapTemp;
    QImage backimg;
    Mat targetsegmat;

public slots:
    void on_actionOpenFileSlot();
    void on_actionCaptureSlot();
    void on_actionSaveSlot();
    void on_actionSaveAsSlot();
    void receiveImage(QImage pic);
    void startsalt();
    void startGauss();
    void startrand();
    void startAverSmooth();
    void startMidSmooth();
    void startGaussSmooth();
    void startPinyi();
    void startRotate();
    void startCut();
    void resetPic();
    void startlight();
    void startroberts();
    void startSobel();
    void startprewitt();
    void startlaplacian();
    void mouseDoubleClickEvent(QMouseEvent *event);
    void startrigion();
    void startIterator();
    void starttargetseg();
    void replaceback();
    void startfoggy();
    void startfudiao();
    void startedge();
    static void onMouse(int event, int x, int y, int flags, void*);
    void startgetinfo();
//工具函数
public:
    Mat ChangeToMat(QImage image);
    QImage ChangeToQImage(Mat mat);
    void ShowTolabel(QImage img);
    void ShowTolabel1(QImage img);
    bool fileisempty();
    void static MouseEvent(int event, int x, int y, int flags, void*);
    int index=0;






private slots:
    void on_actionpicclearSlot();

private:
    Ui::MainWindow *ui;
    QString fileName;
    bool pressed=false;

    QPoint seed1={-1,-1},seed2={-1,-1};


};
#endif // MAINWINDOW_H
