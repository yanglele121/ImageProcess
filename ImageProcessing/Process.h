#ifndef TEST_H
#define TEST_H

#include <QObject>
#include<opencv2/opencv.hpp>
#include<QImage>
using namespace cv;
class processer : public QObject
{
    Q_OBJECT
public:
    explicit processer(QObject *parent = nullptr);

public:
    Mat addJiaoyan(Mat mat,int saltnum);
    double generateGaussianNoise(double mu, double sigma);
    Mat addGauss(Mat mat,double mean,double sigma);
    Mat addRand(Mat mat);
    Mat Average_Smooth(Mat mat,int M);
    Mat Mid_Smooth(Mat mat,int ksize);
    Mat separateGaussianFilter(Mat mat,int ksize, double sigma);
    Mat imgTranslation(Mat mat, int xOffset, int yOffset);
    Mat rotate(Mat mat,double Angle);
    Mat enRich(Mat mat,int beta=1,int alpha=1);
    Mat roberts(Mat mat);
    Mat sobel(Mat mat);
    Mat prewitt(Mat mat);
    Mat laplacian(Mat mat);
    QImage regiongrowth(QImage img,QRgb seed1,QRgb seed2);
    Mat AreaGrow(Mat mat,QPoint seed1,QPoint seed2);
    Mat IterateThrehold(Mat mat);
    Mat targetseg(Mat mat);
    Mat replaceback(Mat mat,Mat background);
    Mat foggy(Mat mat);
    Mat fudiao(Mat mat);
    Mat addedge(Mat mat,Mat edge);
//    void MouseEvent(int event, int x, int y, int flags, void*);
//    Mat getinfo();

public:
    Mat static temp;
signals:

};

#endif // TEST_H
