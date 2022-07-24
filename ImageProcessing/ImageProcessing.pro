QT       += core gui


greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    Process.cpp \
    cameradlg.cpp \
    main.cpp \
    mainwindow.cpp \
    showpic.cpp

HEADERS += \
    Process.h \
    cameradlg.h \
    mainwindow.h \
    showpic.h

FORMS += \
    cameradlg.ui \
    mainwindow.ui
INCLUDEPATH +=D:\Qt\opencv3.4.0\OpenCV-MinGW-Build-OpenCV-3.4.5\include \
              D:\Qt\opencv3.4.0\OpenCV-MinGW-Build-OpenCV-3.4.5\include \
              D:\Qt\opencv3.4.0\OpenCV-MinGW-Build-OpenCV-3.4.5\include\opencv2

LIBS +=D:\Qt\opencv3.4.0\OpenCV-MinGW-Build-OpenCV-3.4.5\x86\mingw\bin\libopencv_*.dll

            RC_ICONS = logo1.ico

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

RESOURCES += \
    image.qrc
