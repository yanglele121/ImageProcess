#include "cameradlg.h"
#include "ui_cameradlg.h"
#include<opencv2/opencv.hpp>
#include<highgui.hpp>
#include<qtimer.h>
#include<QTimer>
#include<opencv2/videoio.hpp>
#include<opencv2/core.hpp>
#include<QDebug>
using namespace cv;

cameradlg::cameradlg(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::cameradlg)
{
    ui->setupUi(this);

    timer   = new QTimer(this);


    connect (timer,SIGNAL(timeout()),this,SLOT(readFrame()));
    connect(ui->openbtn,&QPushButton::clicked,this,&cameradlg::openCamara);

    connect(ui->takebtn,&QPushButton::clicked,this,&cameradlg::takePhoto);
    connect(ui->stopbtn,&QPushButton::clicked,this,&cameradlg::closeCamera);

}

void cameradlg::showwin(){
    this->show();
}
cameradlg::~cameradlg()
{
    delete ui;
}
//打开摄像头
void cameradlg::openCamara(){
    this->cam.open(0);
    this->timer->start(33);
}

//将读取的图像显示在指定位置
void cameradlg::readFrame(){


        this->cam >> frame;

         //将抓取到的帧，转换为QImage格式。QImage::Format_RGB888不同的摄像头用不同的格式。
        QImage image((const uchar*)frame.data, frame.cols, frame.rows,frame.step, QImage::Format_RGB888);
        image=image.rgbSwapped();
        fitpixmap=image.scaled(ui->vedio->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        ui->vedio->setPixmap(QPixmap::fromImage(fitpixmap));  // 将图片显示到label上
        ui->vedio->setAlignment(Qt::AlignCenter);

}
void cameradlg::takePhoto()
{
    this->cam >> frame;

     //将抓取到的帧，转换为QImage格式。QImage::Format_RGB888不同的摄像头用不同的格式。
    QImage image((const uchar*)frame.data, frame.cols, frame.rows,frame.step, QImage::Format_RGB888);
    image=image.rgbSwapped();
    fitpixmap=image.scaled(ui->vedio->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->photo->setPixmap(QPixmap::fromImage(fitpixmap));  // 将图片显示到label上
    ui->photo->setAlignment(Qt::AlignCenter);
    imageStatic=image;
    if(!imageStatic.isNull()){
       emit sendImage(imageStatic);
    }
}

void cameradlg::closeCamera()
{
    cam.release();
    timer->stop();
}



