#ifndef CAMERADLG_H
#define CAMERADLG_H

#include <QWidget>
#include<opencv2/opencv.hpp>
#include<highgui.hpp>
#include<qtimer.h>
using namespace cv;
namespace Ui {
class cameradlg;
}

class cameradlg : public QWidget
{
    Q_OBJECT

public:
    explicit cameradlg(QWidget *parent = nullptr);
    ~cameradlg();

public:
    void showwin();

public slots:
    void openCamara() ;//打开摄像头
    void readFrame();//实时显示
    //void stopCamera(); //暂停摄像头
    void takePhoto();  //拍照
    void closeCamera();//关闭摄像头

signals:
    void sendImage(QImage pic);
public:
    VideoCapture cam;
    Mat frame;
    QImage fitpixmap;
    QImage imageStatic;
    QTimer    *timer;
private:
    Ui::cameradlg *ui;


};

#endif // CAMERADLG_H
