#ifndef SHOWPIC_H
#define SHOWPIC_H

#include <QLabel>

class showPic:public QLabel
{
    Q_OBJECT
public:
    explicit showPic(QWidget *parent = nullptr);
    //鼠标进入事件
//    //void enterEvent(QEvent *event);
//    //鼠标离开事件
//    // void leaveEvent(QEvent *);
//        //鼠标按下
//       virtual void mouseDoubleClickEvent(QMouseEvent *ev);

//       //鼠标释放
//      // virtual void mouseReleaseEvent(QMouseEvent *ev);

//       //鼠标移动
//    //   virtual void mouseMoveEvent(QMouseEvent *ev);

//signals:
//    void sendseed(int x,int y);
//public:
//    int x,y;

};

#endif // SHOWPIC_H
