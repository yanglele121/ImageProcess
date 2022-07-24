#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\imgproc\types_c.h>
#include <opencv2\objdetect\objdetect_c.h>
#include "QDebug"
#include<QFileDialog>
#include<QPixmap>
#include<QSize>
#include<QMessageBox>
#include<cameradlg.h>
#include<QDebug>
#include<Process.h>
#include<string.h>
#include<QMouseEvent>
using namespace cv;
using namespace std;
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->setWindowTitle("图改改");
    this->setStyleSheet("color:rgb(64,65,66)");
    ui->EdgeOptionBar->setStyleSheet("color:white");
    ui->splitter_2->setStretchFactor(0, 1);
    ui->splitter_2->setStretchFactor(1, 3);
    ui->splitter->setStretchFactor(0,4);
    ui->splitter->setStretchFactor(1,1);
    ui->splitter_3->setStretchFactor(0,3);
    ui->splitter_3->setStretchFactor(1,1);
    ui->stackedWidget->setCurrentWidget(ui->page_0);
    //默认展示subpage第1页
    ui->stackedWidget_2->setCurrentIndex(1);
    QImage bkg,bkg1;
    bkg.load("://pic/main.png");
    bkg1=bkg.scaled(ui->Picstatic->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->Picstatic->setPixmap(QPixmap::fromImage(bkg1));
    ui->Picstatic->setAlignment(Qt::AlignCenter);

//    QImage bg,bg1;
//    bg.load("://pic/libary.jpg");
//    bg1=bg.scaled(ui->Picstatic->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
//    ui->morelabel->setPixmap(QPixmap::fromImage(bg1));
//    ui->morelabel->setAlignment(Qt::AlignCenter);

    connect(ui->TurntoPage0,&QPushButton::clicked,this,[=](){
        ui->stackedWidget->setCurrentIndex(0);
    });
    connect(ui->TurntoPage1,&QPushButton::clicked,this,[=](){
        ui->stackedWidget->setCurrentIndex(1);
    });
    connect(ui->actionOpenFile,&QAction::triggered,this,&MainWindow::on_actionOpenFileSlot);
    connect(ui->actionCapture,&QAction::triggered,this,&MainWindow::on_actionCaptureSlot);
    connect(ui->addnoisebtn,&QPushButton::clicked,this,[=](){
        ui->stackedWidget_2->setCurrentIndex(0);
    });
    connect(ui->reducenoisebtn,&QPushButton::clicked,this,[=](){
        ui->stackedWidget_2->setCurrentIndex(2);
    });
    connect(ui->geotransbtn,&QPushButton::clicked,this,[=](){
        ui->stackedWidget_2->setCurrentIndex(3);
    });
    connect(ui->startsalt,&QPushButton::clicked,this,&MainWindow::startsalt);
    connect(ui->startGauss,&QPushButton::clicked,this,&MainWindow::startGauss);
    connect(ui->startrand,&QPushButton::clicked,this,&MainWindow::startrand);
    connect(ui->startAverSmooth,&QPushButton::clicked,this,&MainWindow::startAverSmooth);
    connect(ui->startmid,&QPushButton::clicked,this,&MainWindow::startMidSmooth);
    connect(ui->startGaussSmooth,&QPushButton::clicked,this,&MainWindow::startGaussSmooth);
    connect(ui->startmove,&QPushButton::clicked,this,&MainWindow::startPinyi);
    connect(ui->startrotate,&QPushButton::clicked,this,&MainWindow::startRotate);
    connect(ui->imagecropbtn,&QPushButton::clicked,this,&MainWindow::startCut);
    connect(ui->actionSave,&QAction::triggered,this,&MainWindow::on_actionSaveSlot);
    connect(ui->actionSaveAs,&QAction::triggered,this,&MainWindow::on_actionSaveAsSlot);
    connect(ui->resetbtn_1,&QPushButton::clicked,this,&MainWindow::resetPic);
    connect(ui->adjustbtn,&QPushButton::clicked,this,[=](){
        ui->stackedWidget_2->setCurrentIndex(4);
    });
    connect(ui->lightSlider,&QSlider::valueChanged,this,&MainWindow::startlight);
    connect(ui->compareSlider,&QSlider::valueChanged,this,&MainWindow::startlight);
    connect(ui->richSlider,&QSlider::valueChanged,this,&MainWindow::startlight);
    connect(ui->resetbtn_2,&QPushButton::clicked,this,&MainWindow::resetPic);
    connect(ui->startedge,&QPushButton::clicked,this,[=](){
        ui->stackedWidget_2->setCurrentIndex(5);
    });
    connect(ui->startroberts,&QPushButton::clicked,this,&MainWindow::startroberts);
    connect(ui->startsobel,&QPushButton::clicked,this,&MainWindow::startSobel);
    connect(ui->startprewitt,&QPushButton::clicked,this,&MainWindow::startprewitt);
    connect(ui->startlaplacian,&QPushButton::clicked,this,&MainWindow::startlaplacian);
    connect(ui->resetbtn_3,&QPushButton::clicked,this,&MainWindow::resetPic);
    connect(ui->target_seg,&QPushButton::clicked,this,[=](){
        ui->stackedWidget_2->setCurrentIndex(6);
    });
    connect(ui->startregiongrowth,&QPushButton::clicked,this,&MainWindow::startrigion);
    connect(ui->startIterator,&QPushButton::clicked,this,&MainWindow::startIterator);
    connect(ui->startseg,&QPushButton::clicked,this,&MainWindow::starttargetseg);
    connect(ui->startreplace,&QPushButton::clicked,this,&MainWindow::replaceback);
    connect(ui->resetbtn_4,&QPushButton::clicked,this,&MainWindow::resetPic);
    connect(ui->actionpicclear,&QAction::triggered,this,&MainWindow::on_actionpicclearSlot);
    connect(ui->startfoggy,&QPushButton::clicked,this,&MainWindow::startfoggy);
    connect(ui->startfudiao,&QPushButton::clicked,this,&MainWindow::startfudiao);
    connect(ui->startaddedge,&QPushButton::clicked,this,&MainWindow::startedge);
    connect(ui->resetbtn_5,&QPushButton::clicked,this,&MainWindow::resetPic);
    connect(ui->startnums,&QPushButton::clicked,this,[=](){
        ui->stackedWidget_2->setCurrentIndex(7);
    });
    connect(ui->specialarea,&QPushButton::clicked,this,&MainWindow::startgetinfo);
    connect(ui->resetbtn_6,&QPushButton::clicked,this,&MainWindow::resetPic);
    connect(ui->actionInfo,&QAction::triggered,this,[=](){
        ui->stackedWidget->setCurrentIndex(2);
    });
    connect(ui->actionAbout,&QAction::triggered,this,[=](){
        QMessageBox::information(this," ","Version:图改改1.0.1\n©杨乐乐 2022",QMessageBox::Yes);
    });
    connect(ui->actionFeedback,&QAction::triggered,this,[=](){
        QMessageBox::information(this," ","反馈请发送至邮箱:yanglele121@163.com",QMessageBox::Ok);
    });
}

MainWindow::~MainWindow()
{
    delete ui;
}

//以下两个函数实现QImage和Mat的互转
Mat MainWindow::ChangeToMat(QImage image){
    image = image.convertToFormat(QImage::Format_RGB888);
    cv::Mat tmp(image.height(),image.width(),CV_8UC3,(uchar*)image.bits(),image.bytesPerLine());
    cv::Mat result; // deep copy just in case (my lack of knowledge with open cv)
    cvtColor(tmp, result,CV_BGR2RGB);
    return result;
}

QImage MainWindow::ChangeToQImage(Mat mat){
    cv::cvtColor(mat, mat, CV_BGR2RGB);
    QImage qim((const unsigned char*)mat.data, mat.cols, mat.rows, mat.step,
    QImage::Format_RGB888);
    return qim;
}

//图片展示操作
void MainWindow::ShowTolabel(QImage img){
    //ui->showPic->clear();
    fitpixmap=img.scaled(ui->showPic->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    //fitpixmap=img;
    ui->showPic->setPixmap(QPixmap::fromImage(fitpixmap));
    ui->showPic->setAlignment(Qt::AlignCenter);
}
void MainWindow::ShowTolabel1(QImage img){
    //ui->showPic->clear();
    fitpixmap=img;
    ui->showPic->setPixmap(QPixmap::fromImage(fitpixmap));
    ui->showPic->setAlignment(Qt::AlignCenter);
}
bool MainWindow::fileisempty(){
    if(fitpixmap.isNull()){
        QMessageBox::warning(this,"文件读取错误","未打开任何图片",QMessageBox::Ok);
        return true;
    }
    else{
        return false;
    }
}
//打开文件
void MainWindow::on_actionOpenFileSlot()
{
    fileName = QFileDialog::getOpenFileName(this, "选择图片","image","*.png *.bmp *.jpg");
    //qDebug()<<fileName<<endl;
    if(!fileName.isEmpty()){
        Src.load(fileName);
        pixStatic=Src;//备份
        fitpixmap=Src.scaled(ui->showPic->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        fitpixmapTemp=fitpixmap;
        ui->showPic->setPixmap(QPixmap::fromImage(fitpixmap));
        ui->showPic->setAlignment(Qt::AlignCenter);
        ui->textshow2->setText(QString("长度：%1\n宽度：%2\n通道数：%3").arg(fitpixmap.height()).arg(fitpixmap.width()).arg(fitpixmap.depth()/8));
    }
}
//打开摄像头
void MainWindow::on_actionCaptureSlot()
{
    cameradlg* dlg=new cameradlg;
    dlg->showwin();
    connect(dlg,SIGNAL(sendImage(QImage)),this,SLOT(receiveImage(QImage)));
}


//接收摄像头拍摄图片
void MainWindow::receiveImage(QImage pic){
    Src=pic;
    pixStatic=Src;
    fitpixmap=Src.scaled(ui->showPic->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
     ui->textshow2->setText(QString("长度：%1\n宽度：%2\n通道数：%3").arg(fitpixmap.height()).arg(fitpixmap.width()).arg(fitpixmap.depth()/8));
    fitpixmapTemp=fitpixmap;
    ui->showPic->setPixmap(QPixmap::fromImage(fitpixmap));
    ui->showPic->setAlignment(Qt::AlignCenter);
}

//保存
void MainWindow::on_actionSaveSlot()
{
    if(fileisempty()){
        return;
    }
   int ret=QMessageBox::information(this,"提醒","保存对图片的更改?",QMessageBox::Yes,QMessageBox::No);
   if(ret==QMessageBox::Yes){
//           QString filename1 = QFileDialog::getSaveFileName(this,tr("Save Image"),"",tr("Images (*.png *.bmp *.jpg)")); //选择路径
           string fileAsSave = fileName.toStdString();
           imwrite(fileAsSave,ChangeToMat(fitpixmap));
   }
}
//另存
void MainWindow::on_actionSaveAsSlot()
{
    if(fileisempty()){
    return;
    }
   QString filename1 = QFileDialog::getSaveFileName(this,tr("Save Image As"),"",tr("Images (*.png *.bmp *.jpg)")); //选择路径
   if(filename1.isNull()){
       return;
   }
   string fileAsSave = filename1.toStdString();
   imwrite(fileAsSave,ChangeToMat(fitpixmap));
}
void MainWindow::on_actionpicclearSlot()
{
    ui->showPic->clear();
}
//执行椒盐噪声
void MainWindow::startsalt(){
    if(fileisempty()){
        return;
    }
    int num=ui->saltnumEdit->text().toInt();
    processer *work =new processer();
    Mat inputImage=ChangeToMat(fitpixmap);
    //imshow("input",inputImage);
    Mat outImage=work->addJiaoyan(inputImage,num);
    //imshow("test",outImage);
    QImage img=ChangeToQImage(outImage);
    ui->textshow->setText("椒盐噪声也称为脉冲噪声，是图像中经常见到的一种噪声，它是一种随机出现的白点或者黑点，可能是亮的区域有黑色像素或是在暗的区域有白色像素（或是两者皆有）。");
    ShowTolabel(img);
    delete  work;
}
//执行高斯噪声
void MainWindow::startGauss(){
    if(fileisempty()){
        return;
    }
     QImage orimg=ui->showPic->pixmap()->toImage();
    double mean=ui->meanEdit->text().toDouble();
    double sigma=ui->sigmaEdit->text().toDouble();
    processer*work=new processer();
    Mat inputImage=ChangeToMat(orimg);
    //imshow("test",inputImage);
    Mat outImage=work->addGauss(inputImage,mean,sigma);
    //imshow("out0",outImage);
    QImage img=ChangeToQImage(outImage);
    ui->textshow->setText("高斯噪声是指它的概率密度函数服从高斯分布（即正态分布）的一类噪声。如果一个噪声，它的幅度分布服从高斯分布，而它的功率谱密度又是均匀分布的，则称它为高斯白噪声。高斯白噪声的二阶矩不相关，一阶矩为常数，是指先后信号在时间上的相关性。高斯白噪声包括热噪声和散粒噪声。在通信信道测试和建模中，高斯噪声被用作加性白噪声以产生加性白高斯噪声。");
    ShowTolabel(img);
    delete  work;
}
//执行随机噪声
void MainWindow::startrand(){
    if(fileisempty()){
        return;
    }
    processer*work=new processer();
    QImage orimg=ui->showPic->pixmap()->toImage();
    Mat inputImage=ChangeToMat(orimg);
    Mat outImage=work->addRand(inputImage);
    QImage img=ChangeToQImage(outImage);
    ui->textshow->setText("随机噪声是一种由时间上随机产生的大量起伏骚扰积累而造成的，其值在给定瞬间内不能预测的噪声");
    ShowTolabel(img);
    delete  work;
}
//执行简单随机滤波
void MainWindow::startAverSmooth(){
    if(fileisempty()){
        return;
    }
     QImage orimg=ui->showPic->pixmap()->toImage();
    int M=ui->mubanEdit->text().toInt();
    processer*work=new processer();
    Mat inputImage=ChangeToMat(orimg);
    //imshow("test0",inputImage);
    Mat outImage=work->Average_Smooth(inputImage,M);
    //imshow("test",outImage);
    QImage img=ChangeToQImage(outImage);
    ui->textshow->setText("均值滤波是典型的线性滤波算法，它是指在图像上对目标像素给一个模板，该模板包括了其周围的临近像素（以目标像素为中心的周围8个像素，构成一个滤波模板，即包括目标像素本身），再用模板中的全体像素的平均值来代替原来像素值。");
    ShowTolabel(img);
    delete  work;
}
//中值滤波
void MainWindow::startMidSmooth(){
    if(fileisempty()){
        return;
    }
     QImage orimg=ui->showPic->pixmap()->toImage();
    int minSize=ui->minSizeEdit->text().toInt();
    //qDebug()<<minSize<<endl;
    //int maxSize=ui->maxSizeEdit->text().toInt();
    processer*work=new processer();
    Mat inputImage=ChangeToMat(orimg);
    //imshow("test",inputImage);
    Mat outImage=work->Mid_Smooth(inputImage,minSize);
    QImage img=ChangeToQImage(outImage);
    ui->textshow->setText("中值滤波法是一种非线性平滑技术，它将每一像素点的灰度值设置为该点某邻域窗口内的所有像素点灰度值的中值。");
    ShowTolabel(img);
    delete  work;
}
void MainWindow::startGaussSmooth()
{
    if(fileisempty()){
        return;
    }
     QImage orimg=ui->showPic->pixmap()->toImage();
    int ksize=ui->ksizeEdit->text().toInt();
    double sigma=ui->sigmaEdit_2->text().toDouble();
    processer*work=new processer();
    Mat inputImage=ChangeToMat(orimg);
    Mat outImage=work->separateGaussianFilter(inputImage,ksize,sigma);
    QImage img=ChangeToQImage(outImage);
    ui->textshow->setText("高斯滤波是一种线性平滑滤波，适用于消除高斯噪声，广泛应用于图像处理的减噪过程。");
    ShowTolabel(img);
    delete  work;
}

//平移

void MainWindow::startPinyi(){
    if(fileisempty()){
    return;
    }
    QImage orimg=ui->showPic->pixmap()->toImage();
    int x,y;
    x=ui->dxEdit->text().toInt();
    y=ui->dyEdit->text().toInt();
    processer*work=new processer();
    Mat inImage=ChangeToMat(orimg);

    Mat outImage=work->imgTranslation(inImage,x,y);
    QImage img=ChangeToQImage(outImage);
    ui->textshow->setText("图像平移就是将图像中的所有像素点按照给定的平移量进行水平（x方向）或垂直（y方向）移动。");
    ShowTolabel1(img);
    delete  work;
}
void MainWindow::startRotate(){
    if(fileisempty()){
        return;
    }
    QImage orimg=ui->showPic->pixmap()->toImage();
    double angle=ui->anglenum->text().toDouble();
    processer*work=new processer();
    Mat inImage=ChangeToMat(orimg);
    Mat outImage=work->rotate(inImage,angle);
    QImage img=ChangeToQImage(outImage);
    ui->textshow->setText("图像旋转是指图像以某一点为中心旋转一定的角度，形成一幅新的图像的过程。当然这个点通常就是图像的中心。既然是按照中心旋转，自然会有这样一个属性：旋转前和旋转后的点离中心的位置不变。");
    ShowTolabel1(img);
    delete  work;
}

Mat  matStatic;
Rect roirect;
Point startPoint;
Point endPoint;
//实现裁剪
void MainWindow::MouseEvent(int event, int x, int y, int flags, void*)
{

    if (event == CV_EVENT_LBUTTONDOWN)
    {
        startPoint = Point(x, y);
    }
    else if (event==CV_EVENT_MOUSEMOVE && (flags&CV_EVENT_FLAG_LBUTTON))
    {
        endPoint=Point(x,y);
        Mat tempImage = matStatic.clone();
        //rectangle(src,原点,终点,linecolor,linewidth,linetype,0);
        //rectangle(src,Rect(原点,width,height),linecolor,linewidth,linetype,0);
        rectangle(tempImage,startPoint,endPoint,Scalar(250,0,100),2,8,0);
        //rectangle(matStatic, startPoint, endPoint, Scalar(255, 0, 0), 2, 8, 0);
        imshow("OriginalImage",tempImage);
    }
    roirect.width = abs(endPoint.x - startPoint.x);
    roirect.height = abs(endPoint.y - startPoint.y);
    if (roirect.width > 0 && roirect.height > 0)
    {
        roirect.x = min(startPoint.x, endPoint.x);
        roirect.y = min(startPoint.y, endPoint.y);
        Mat roiMat = matStatic(Rect(roirect.x, roirect.y, roirect.width, roirect.height));
        imshow("result", roiMat);
    }
}
void MainWindow::startCut(){
    if(fileisempty()){
        return;
    }
    QImage orimg=ui->showPic->pixmap()->toImage();
    matStatic=ChangeToMat(orimg);
    namedWindow("OriginalImage");
    imshow("OriginalImage", matStatic);
    setMouseCallback("OriginalImage",MouseEvent,0);
    waitKey(0);
    ui->textshow->setText("根据矩形框裁剪出所需区域");
}


void MainWindow::resetPic(){
    ShowTolabel1(fitpixmapTemp);
    ui->meanEdit->setText("0");
    ui->sigmaEdit->setText("0");
    ui->saltnumEdit->setText("0");
    ui->mubanEdit->setText("0");
    ui->minSizeEdit->setText("0");
    ui->ksizeEdit->setText("0");
    ui->sigmaEdit_2->setText("0");
    ui->dxEdit->setText("0");
    ui->dyEdit->setText("0");
    ui->anglenum->setValue(0.00);
    ui->lightSlider->setValue(0);
    ui->lightnum->setNum(0);
    ui->compareSlider->setValue(1);
    ui->comparenum->setNum(1);
    ui->richSlider->setValue(0);
    ui->richnum->setText("0%");
    ui->seed1->setText("");
    ui->seed2->setText("");
    ui->textshow->clear();
    ui->rectnum->clear();
    ui->circlenum->clear();
    ui->centrenum->clear();
    ui->lengthnum->clear();
}

void MainWindow::startlight(){
    if(fileisempty()){
    return;
    }
    int beta=0;
    float alpha=1.0;
    //QImage temp=fitpixmap;
    beta=ui->lightSlider->value();
    alpha=(1.0*(ui->compareSlider->value()+100))/100.0;
    ui->lightnum->setNum(beta);
    ui->comparenum->setNum(alpha);
    QColor oldColor;
    QImage img=fitpixmapTemp;
    QImage newImage=QImage(img.width(), img.height(), QImage::Format_ARGB32);
    int r=0,g=0,b=0,a=0;

    //uchar *line = img.scanLine(0);
    //uchar *pixel = line;

    int width=img.width();//获取图像宽度
    int height=img.height();//获取图像高度
     for (int i=0;i<height;i++)
     {
         for (int j=0;j<width;j++)
         {
             oldColor = QColor(img.pixel(j,i));
             r = oldColor.red()*alpha + beta;
             g = oldColor.green()*alpha + beta;
             b = oldColor.blue() *alpha+ beta;

             r = r>255?255:r;
             g = g>255?255:g;
             b = b>255?255:b;

             r =  r  < 0 ? 0 : r;
             g = g < 0 ? 0 : g;
             b = b < 0 ? 0 : b;

             a = qAlpha(img.pixel(j,i));
             newImage.setPixel(j,i, qRgba(r,g,b,a));

         }
     }
     //饱和度调节
     float adjust=0;
     adjust=1.0f*(ui->richSlider->value())/100.0;
     QString data = QString("%1").arg(adjust*100);
     ui->richnum->setText(data+"%");
      r=0,b=0,g=0,a=0;
     for (int i = 0; i < height; ++i) {
         for (int j = 0; j < width; ++j) {
             oldColor = QColor(newImage.pixel(j,i));

             float lum = oldColor.blue() * 0.299f +oldColor.green() * 0.587f + oldColor.red() * 0.114f;
             float maskB = std::max(0.0f, std::min(oldColor.blue() - lum, 255.0f)) / 255.0f;
             float maskG = std::max(0.0f, std::min(oldColor.green() - lum, 255.0f)) / 255.0f;
             float maskR = std::max(0.0f, std::min(oldColor.red() - lum, 255.0f)) / 255.0f;
             float lumMask = (1.0f - (maskB * 0.299f + maskG * 0.587f + maskR * 0.114f)) * adjust;

             r=oldColor.red()*(1.0f+lumMask)-lum*lumMask;
             g=oldColor.green()*(1.0f+lumMask)-lum*lumMask;
             b=oldColor.blue()*(1.0f+lumMask)-lum*lumMask;

             r = r>255?255:r;
             g = g>255?255:g;
             b = b>255?255:b;

             r =  r  < 0 ? 0 : r;
             g = g < 0 ? 0 : g;
             b = b < 0 ? 0 : b;

             a = qAlpha(img.pixel(j,i));
             newImage.setPixel(j,i, qRgba(r,g,b,a));
         }
     }

    ui->textshow->setText("图象亮度是指画面的明亮程度，单位是堪德拉每平米(cd/m2)或称nits。图象亮度是从白色表面到黑色表面的感觉连续体，由反射系数决定，亮度侧重物体，重在“反射”。\n"
                          "图像对比度指的是一幅图像中明暗区域最亮的白和最暗的黑之间不同亮度层级的测量，即指一幅图像灰度反差的大小。差异范围越大代表对比越大，差异范围越小代表对比越小，好的对比率120:1就可容易地显示生动、丰富的色彩，当对比率高达300:1时，便可支持各阶的颜色。\n"
                          "色彩的饱和度(saturation)指色彩的鲜艳程度，也称作纯度。在hue-saturation-value(HSV)色彩模型下，饱和度是色彩的3个属性之一，另外两个属性为色相(hue)和明度(value)；在此模型下色相的取值范围为0°到360°，饱和度和明度取值范围为0到100%。在色彩学中，原色饱和度最高，随着饱和度降低，色彩变得暗淡直至成为无彩色，即失去色相的色彩。作为信息的载体，色彩不仅依附于设计形式，还作为一个主体来完成信息传达的过程。");
        ShowTolabel1(newImage);
}

void MainWindow::startroberts(){
    if(fileisempty()){
        return;
    }
     QImage orimg=ui->showPic->pixmap()->toImage();
    processer *work=new processer();
    Mat inputImage=ChangeToMat(orimg);
    Mat outImage=work->roberts(inputImage);
    QImage img=ChangeToQImage(outImage);
    ui->textshow->setText("Roberts算子又称为交叉微分算法，它是基于交叉差分的梯度算法，通过局部差分计算检测边缘线条。常用来处理具有陡峭的低噪声图像，当图像边缘接近于正45度或负45度时，该算法处理效果更理想。其缺点是对边缘的定位不太准确，提取的边缘线条较粗。");
    ShowTolabel1(img);
    delete  work;
}

void MainWindow::startSobel(){
    if(fileisempty()){
        return;
    }
     QImage orimg=ui->showPic->pixmap()->toImage();
    processer* work=new processer();
    Mat inputImage=ChangeToMat(orimg);
    Mat outImage=work->sobel(inputImage);
    QImage img=ChangeToQImage(outImage);
    ui->textshow->setText("其主要用于边缘检测，在技术上它是以离散型的差分算子，用来运算图像亮度函数的梯度的近似值，缺点是Sobel算子并没有将图像的主题与背景严格地区分开来，换言之就是Sobel算子并没有基于图像灰度进行处理，由于Sobel算子并没有严格地模拟人的视觉生理特征，所以提取的图像轮廓有时并不能令人满意，算法具体实现很简单，就是3*3的两个不同方向上的模板运算，这里不再写出。");
    ShowTolabel1(img);
    delete  work;
}
void MainWindow::startprewitt(){
    if(fileisempty()){
        return;
    }
     QImage orimg=ui->showPic->pixmap()->toImage();
    processer* work=new processer();
    Mat inputImage=ChangeToMat(orimg);
    Mat outImage=work->prewitt(inputImage);
    QImage img=ChangeToQImage(outImage);
    ui->textshow->setText("该算子与Sobel算子类似，只是权值有所变化，但两者实现起来功能还是有差距的，据经验得知Sobel要比Prewitt更能准确检测图像边缘。");
    ShowTolabel1(img);
    delete  work;
}

void MainWindow::startlaplacian(){
    if(fileisempty()){
        return;
    }
     QImage orimg=ui->showPic->pixmap()->toImage();
    processer* work=new processer();
    Mat inputImage=ChangeToMat(orimg);
    Mat outImage=work->laplacian(inputImage);
    QImage img=ChangeToQImage(outImage);
    ui->textshow->setText("拉普拉斯算子是一种二阶微分算子，若只考虑边缘点的位置而不考虑周围的灰度差时可用该算子进行检测。对于阶跃状边缘，其二阶导数在边缘点出现零交叉，并且边缘点两旁的像素的二阶导数异号。");
    ShowTolabel1(img);
    delete  work;
}

void MainWindow::mouseDoubleClickEvent(QMouseEvent *event){
if (!pressed)//成员变量bool Pressed = false;   //鼠标是否被摁压
    {

        QPoint m_nMovePos;
        m_nMovePos = event->globalPos();
        //leftButtomRightPoint = event->                          //获取全局位置
         QImage orimg=ui->showPic->pixmap()->toImage();
        m_nMovePos = ui->showPic->mapFromGlobal(m_nMovePos);
        int xoffset = (ui->showPic->contentsRect().width()-ui->showPic->pixmap()->rect().width())/2;
        int yoffset = (ui->showPic->contentsRect().height()-ui->showPic->pixmap()->rect().height())/2;

        int seedx = m_nMovePos.x()-xoffset;
        int seedy = m_nMovePos.y()-yoffset;
        if(seedx<0||seedy<0){
            QMessageBox::warning(this,"警告","请选取在图像内的种子点！",QMessageBox::Ok);
        }
        else{
            QRgb color=orimg.pixel(seedx,seedy);
            int R=qRed(color),G=qGreen(color),B=qBlue(color);
            if(index%2==0){
            ui->seed1->setText(QString("坐标：(%1,%2)\nRGB：(%3,%4,%5)").arg(seedx).arg(seedy).arg(R).arg(G).arg(B));
            seed1={seedx,seedy};
            }
            else{
            ui->seed2->setText(QString("坐标：(%1,%2)\nRGB：(%3,%4,%5)").arg(seedx).arg(seedy).arg(R).arg(G).arg(B));
            seed2={seedx,seedy};
            }
            index++;
        }
        return QWidget::mouseDoubleClickEvent(event);
    }
}

void MainWindow::startrigion(){
    if(fileisempty()){
        return;
    }
    if(index==0){
      QMessageBox::critical(this,"错误","请选取种子点！",QMessageBox::Ok);
      return;
    }
     QImage orimg=ui->showPic->pixmap()->toImage();
    processer* work=new processer();
    Mat inputImage=ChangeToMat(orimg);
    //cvtColor(inputImage,inputImage1,CV_BGR2Luv);
    //imshow("test",inputImage1);
    //imshow("1",inputImage);
    Mat outImage=work->AreaGrow(inputImage,seed1,seed2);

    //cvtColor(outImage,outImage,CV_Luv2BGR);
    //imshow("2",outImage);
    QImage img=ChangeToQImage(outImage);
    ui->textshow->setText("区域生长（region growing）是指将成组的像素或区域发展成更大区域的过程。从种子点的集合开始，从这些点的区域增长是通过将与每个种子点有相似属性像强度、灰度级、纹理颜色等的相邻像素合并到此区域。");
    ShowTolabel1(img);
    delete  work;
}

void MainWindow::startIterator(){
    if(fileisempty()){
        return;
    }
     QImage orimg=ui->showPic->pixmap()->toImage();
    processer *work=new processer();
    Mat inputImage=ChangeToMat(orimg);
    Mat outImage=work->IterateThrehold(inputImage);
    QImage img=ChangeToQImage(outImage);
    ui->textshow->setText("阈值分割法是一种基于区域的图像分割技术，原理是把图像像素点分为若干类。图像阈值化分割是一种传统的最常用的图像分割方法，因其实现简单、计算量小、性能较稳定而成为图像分割中最基本和应用最广泛的分割技术。");
    ShowTolabel1(img);
    delete  work;
}

void MainWindow::starttargetseg(){
    if(fileisempty()){
        return;
    }
    QMessageBox::information(this,"操作提醒","长按鼠标左键框选待裁剪区域点击键盘空格完成操作",QMessageBox::Yes);
     QImage orimg=ui->showPic->pixmap()->toImage();
    processer *work=new processer();
    Mat inputImage=ChangeToMat(orimg);
    Mat outImage=work->targetseg(inputImage);
    targetsegmat=outImage;
    //imshow("target",targetsegmat);
    QImage img=ChangeToQImage(outImage);
    ui->textshow->setText("Grabcut是基于图割(graph cut)实现的图像分割算法，它需要用户输入一个bounding box作为分割目标位置，实现对目标与背景的分离的分割,框选目标后键入空格或回车键进行分割");
    ShowTolabel1(img);
    delete  work;
}

void MainWindow::replaceback(){
    if(fileisempty()){
        return;
    }
    QString fileName1 = QFileDialog::getOpenFileName(this, "选择新背景图片","image","*.png *.bmp *.jpg");
    //qDebug()<<fileName<<endl;
    if(!fileName1.isEmpty()){
        QImage Src1;
        Src1.load(fileName1);
        Mat newback=ChangeToMat(Src1);
        processer *work=new processer();
        //Mat obj=targetsegmat.clone();
        Mat outImage=work->replaceback(targetsegmat,newback);
        //imshow("out",outImage);
        QImage img=ChangeToQImage(outImage);
        ui->textshow->setText("实现置换背景");
        ShowTolabel1(img.rgbSwapped());
        delete  work;
    }else{
        QMessageBox::critical(this,"错误","请先选择背景图",QMessageBox::Ok);
        return;
    }

}

void MainWindow::startfoggy(){
    if(fileisempty()){
        return;
    }
     QImage orimg=ui->showPic->pixmap()->toImage();
    Mat inputImage=ChangeToMat(orimg);
    processer*work=new processer();
    Mat outImage=work->foggy(inputImage);
    QImage img=ChangeToQImage(outImage);
    ui->textshow->setText("通过调整像素亮度和对比度来实现雾化效果，本算法实现较差");
    ShowTolabel1(img);
    delete  work;
}

void MainWindow::startfudiao(){
    if(fileisempty()){
        return;
    }
     QImage orimg=ui->showPic->pixmap()->toImage();
    Mat inputImage=ChangeToMat(orimg);
    processer*work=new processer();
    Mat outImage=work->fudiao(inputImage);
    QImage img=ChangeToQImage(outImage);
    ui->textshow->setText("通过勾画图像的轮廓，并且降低周围的像素值，从而产生一张具有立体感的浮雕效果图片。这里我们通过相邻元素相减的方法得到轮廓与边缘的差，从而获得凹凸的立体感觉。");
    ShowTolabel1(img);
    delete  work;
}

void MainWindow::startedge(){
    if(fileisempty()){
        return;
    }
     QImage orimg=ui->showPic->pixmap()->toImage();
    QString fileName1 = QFileDialog::getOpenFileName(this, "选择边框图片","image","*.png *.bmp *.jpg");
    //qDebug()<<fileName<<endl;
    if(!fileName1.isEmpty()){
        QImage Src1;
        Src1.load(fileName1);
        Mat edge=ChangeToMat(Src1);
        //imshow("edge",edge);
        Mat inputImage=ChangeToMat(orimg);
        processer *work=new processer();
        //Mat obj=targetsegmat.clone();
        Mat outImage=work->addedge(inputImage,edge);
        GaussianBlur(outImage,outImage, Size(3, 3), 1);
        QImage img=ChangeToQImage(outImage);
        ui->textshow->setText("将边框图像中的纯色部分替换成图片对应部分");
        ShowTolabel1(img);
        delete  work;
    }else{
        QMessageBox::critical(this,"错误","请先选择边框图",QMessageBox::Ok);
        return;
    }
}
Mat result;

void  MainWindow::onMouse(int event, int x, int y, int flags, void*)
{
    Mat roiMat;
    if (event == CV_EVENT_LBUTTONDOWN)
    {
        startPoint = Point(x, y);
    }
    else if (event==CV_EVENT_MOUSEMOVE && (flags&CV_EVENT_FLAG_LBUTTON))
    {
        endPoint=Point(x,y);
        Mat tempImage = matStatic.clone();
        //rectangle(src,原点,终点,linecolor,linewidth,linetype,0);
        //rectangle(src,Rect(原点,width,height),linecolor,linewidth,linetype,0);
        rectangle(tempImage,startPoint,endPoint,Scalar(250,0,100),2,8,0);
        //rectangle(matStatic, startPoint, endPoint, Scalar(255, 0, 0), 2, 8, 0);
        imshow("OriginalImage",tempImage);
    }
    roirect.width = abs(endPoint.x - startPoint.x);
    roirect.height = abs(endPoint.y - startPoint.y);
    if (roirect.width > 0 && roirect.height > 0)
    {
        roirect.x = min(startPoint.x, endPoint.x);
        roirect.y = min(startPoint.y, endPoint.y);
        roiMat = matStatic(Rect(roirect.x, roirect.y, roirect.width, roirect.height));
        if(event==CV_EVENT_MBUTTONDOWN){
            result=roiMat;
            //imshow("result",result);
            return;
        }
    }
}

RNG g_rng(12345);
void MainWindow::startgetinfo(){
    if(fileisempty()){
        return;
    }
     QImage orimg=ui->showPic->pixmap()->toImage();
     QMessageBox::information(this,"操作提醒","长按鼠标左键框选轮廓区域后，点击鼠标中键再点击键盘空格键完成操作",QMessageBox::Yes);
    matStatic=ChangeToMat(orimg);
    namedWindow("OriginalImage");
    imshow("OriginalImage", matStatic);
    setMouseCallback("OriginalImage",onMouse,0);
    waitKey(0);
    //imshow("result",result);
    Mat src_image=result;
    //高斯滤波去噪声
    Mat blur_image;
    GaussianBlur(src_image, blur_image, Size(3, 3), 0, 0);
    //imshow("GaussianBlur", blur_image);

    //灰度变换与二值化
    Mat gray_image, binary_image;
    cvtColor(blur_image, gray_image, COLOR_BGR2GRAY);
    threshold(gray_image, binary_image, 100, 255, THRESH_BINARY);
    //imshow("binary", binary_image);

    //形态学闭操作(粘合断开的区域)
    Mat morph_image;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
    morphologyEx(binary_image, morph_image, MORPH_CLOSE, kernel, Point(-1, -1), 1);
    //imshow("morphology", morph_image);

    //查找所有外轮廓
    vector< vector<Point> > contours;
    vector<Vec4i> hireachy;
    findContours(binary_image, contours, hireachy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());

    //定义结果图
    Mat result_image = Mat::zeros(src_image.size(), CV_8UC3);

    //drawContours(result_image, contours, -1, Scalar(0, 0, 255), 1, 8, hireachy);//画出所有轮廓

    //初始化周长、面积、圆形度、周径比
    double len = 0, area = 0, roundness = 0;
            //lenratio = 0;
    float rectangularity;

    //循环找出所有符合条件的轮廓
    QString rectnumt,circlet,centrenumt,lenthnumt;
    int n=contours.size();
    for (int t = 0; t < n; t++)
    {
        Scalar color = Scalar(g_rng.uniform(0, 255),
            g_rng.uniform(0, 255), g_rng.uniform(0, 255));//任意值
        //条件：过滤掉小的干扰轮廓
        Rect rect = boundingRect(contours[t]);		//垂直边界最小矩形
        if (rect.width < 10)
            continue;
        //画出找到的轮廓
        drawContours(result_image, contours, t, color, 1, 8, hireachy);

        //绘制轮廓的最小外结矩形
        RotatedRect minrect = minAreaRect(contours[t]);	//最小外接矩形
        int minrectmianji = minrect.size.height * minrect.size.width;
        Point2f P[4];			//四个顶点坐标
        minrect.points(P);
        for (int j = 0; j <= 3; j++)
        {
            line(result_image, P[j], P[(j + 1) % 4], color, 1);
        }
        //cout << "最小外接矩形尺寸" << minrect.size << endl;//最小外接矩形尺寸
        //cout << "最小外接矩形面积" << minrectmianji << endl;//最小外接矩形尺寸

        //绘制轮廓的最小外结圆
        Point2f center; float radius;
        minEnclosingCircle(contours[t], center, radius);		//最小外接圆
        circle(result_image, center, radius, color, 1);

        //计算面积、周长、圆形度、周径比
        area = contourArea(contours[t]);//计算轮廓面积
        //qDebug()<<area<<endl;
        int sumx=0,sumy=0;
        for(auto it=contours[t].begin();it!=contours[t].end();it++){
            sumx+=it->x;
            sumy+=it->y;
        }
        int x=sumx/area;
        int y=sumy/area;//重心
        //qDebug()<<x<<endl;
        len = arcLength(contours[t], true);//计算轮廓周长
        roundness = (4 * CV_PI * area) / (len * len);//圆形度
        if (minrectmianji == 0)rectangularity = 0;
        else rectangularity = area / minrectmianji;


        rectnumt.append(QString("轮廓:%1，矩形度:%2\n").arg(t).arg(rectangularity));
        circlet.append(QString("轮廓:%3，圆形度:%4\n").arg(t).arg(roundness));
        centrenumt.append(QString("轮廓:%5，重心:(%6,%7)\n").arg(t).arg(x).arg(y));
        lenthnumt.append(QString("轮廓:%8，周长:%9\n").arg(t).arg(len));
    }
        ui->rectnum->setText(rectnumt);
        ui->circlenum->setText(circlet);
        ui->centrenum->setText(centrenumt);
        ui->lengthnum->setText(lenthnumt);
        QImage img=ChangeToQImage(result_image);
        ui->textshow->setText("获取区域轮廓图得到目标区域的描述参数,框选出目标后先点击鼠标中键再点击键盘空格或回车键得到结果");
        ShowTolabel1(img);
}
