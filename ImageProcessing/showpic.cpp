#include "showpic.h"
#include<QDebug>
#include<QMouseEvent>
showPic::showPic(QWidget *parent) : QLabel(parent)
{

}

//void showPic::enterEvent(QEvent *event)
//{
//qDebug()<<"鼠标进入";
//}

//鼠标离开事件
//void showPic::leaveEvent(QEvent *)
//{
//qDebug()<<"鼠标离开";
//}
//void showPic::mousePressEvent(QMouseEvent *ev)
//{
//    QString str=QString("鼠标按下了 x= %1 y=%2 globalx=%3 globaly=%4").arg(ev->x()).arg(ev->y()).arg(ev->globalX()).arg(ev->globalY());
//    qDebug()<<str;
//}

//鼠标释放


//void showPic::mouseReleaseEvent(QMouseEvent *ev)
//{
//    qDebug()<<"鼠标释放";
//}

//鼠标移动
//void showPic::mouseMoveEvent(QMouseEvent *ev)
//{
//    qDebug()<<"鼠标移动";
//}

//void showPic::mouseDoubleClickEvent(QMouseEvent *ev){
//    x=ev->x();
//    y=ev->y();
//    //qDebug()<<x<<y<<endl;
//    emit sendseed(x,y);
//}
