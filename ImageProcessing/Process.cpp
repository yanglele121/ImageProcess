#include "Process.h"
#include<QDebug>
#include<limits>
#include<stack>
#include<QPoint>
using namespace std;
processer::processer(QObject *parent) : QObject(parent)
{
    //qDebug()<<"构造完成"<<endl;
}
//添加椒盐噪声
Mat processer::addJiaoyan(Mat mat,int saltnum){
    srand(time(0));
    Mat res;
    int w=mat.cols;
    int h=mat.rows;
    int row,col;
    mat.copyTo(res);
    //imshow("res",res);
    for(int i=0;i<saltnum;i++){
        row=rand()%h;
        col=rand()%w;
        if(rand()%2==0){
            res.at<Vec3b>(row,col)[0]=0;
            res.at<Vec3b>(row,col)[1]=0;
            res.at<Vec3b>(row,col)[2]=0;
        }
        else{
            res.at<Vec3b>(row,col)[0]=255;
            res.at<Vec3b>(row,col)[1]=255;
            res.at<Vec3b>(row,col)[2]=255;
        }
    }
    return res;
}

//添加高斯噪声
//生成高斯噪声
double processer:: generateGaussianNoise(double mu, double sigma)
{
    //定义小值
    const double epsilon = numeric_limits<double>::min();
    static double z0, z1;
    static bool flag = false;
    flag = !flag;
    //flag为假构造高斯随机变量X
    if (!flag)
        return z1 * sigma + mu;
    double u1, u2;
    //构造随机变量
    do
    {
        u1 = rand() * (1.0 / RAND_MAX);
        u2 = rand() * (1.0 / RAND_MAX);
    } while (u1 <= epsilon);
    //flag为真构造高斯随机变量
    z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI*u2);
    z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI*u2);
    return z0*sigma + mu;
}


//为图像添加高斯噪声
Mat processer::addGauss(Mat mat,double mean,double sigma)
{
    Mat dstImage = mat.clone();
    int channel = mat.channels();
    int rowsNumber = dstImage.rows;
    int colsNumber = dstImage.cols*channel;
    //判断图像的连续性
    if (dstImage.isContinuous())
    {
        colsNumber *= rowsNumber;
        rowsNumber = 1;
    }
    for (int i = 0; i < rowsNumber; i++)
       {
           for (int j = 0; j < colsNumber; j++)
           {
               //添加高斯噪声
               int val = dstImage.ptr<uchar>(i)[j] + generateGaussianNoise(mean, sigma) * 64;
               if (val < 0)
                   val = 0;
               if (val>255)
                   val = 255;
               dstImage.ptr<uchar>(i)[j] = (uchar)val;
           }
       }
       return dstImage;

}
//添加随机噪声
Mat processer::addRand(Mat mat){
    srand(time(NULL));
    Mat dstImage=mat.clone();
    int channel = mat.channels();
    int rowsNumber = dstImage.rows;
    int colsNumber = dstImage.cols*channel;
    for(int i=0;i<rowsNumber;i++){
        for(int j=0;j<colsNumber;j++){
            long temp=rand();
            int val=dstImage.ptr<uchar>(i)[j]*244/256+temp/1024;
            if(val<0){
                val=0;
            }
            if(val>255){
                val=255;
            }
            dstImage.ptr<uchar>(i)[j]=(uchar)val;
        }
    }
    return dstImage;
}
//去噪

//简单邻域均值滤波
Mat processer::Average_Smooth(Mat mat, int M){
    Mat dstImage=mat.clone();
    int h=mat.rows,w=mat.cols;
    int Start1,Start2;
    Start1=Start2=M/2;
    for(int i=Start1;i<h-Start1;i++){
        for(int j=Start2;j<w-Start2;j++){
            int sumR=0,sumB=0,sumG=0;
            for(int m=i-Start1;m<=i+Start1;m++){
                for(int n=j-Start2;n<=j+Start2;n++){
                    sumR+=mat.at<Vec3b>(m,n)[2];
                    sumG+=mat.at<Vec3b>(m,n)[1];
                    sumB+=mat.at<Vec3b>(m,n)[0];
                }
            }
            dstImage.at<Vec3b>(i,j)[0]=sumB/(M*M);
            dstImage.at<Vec3b>(i,j)[1]=sumG/(M*M);
            dstImage.at<Vec3b>(i,j)[2]=sumR/(M*M);
        }
    }
    return dstImage;

}

void Sort(int* a, int n)		//把数组a递增排序
{
    //for (int gap = n / 2; gap > 0; gap /= 2)//希尔排序
    //	for (int i = gap; i < n; ++i)
    //		for (int j = i - gap; j >= 0 && a[j] > a[j + gap]; j -= gap)
    //			swap(a[j], a[j + gap]);

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++)
        {
            if (a[i] > a[j]) {
                int tmp = a[i];
                a[i] = a[j];
                a[j] = tmp;
            }
        }
    }

}

Mat processer::Mid_Smooth(Mat mat, int ksize){
    // 判断原图像是否为空
        Mat dstImg;
        // 判断核的大小是否为奇数
        CV_Assert(ksize % 2 == 1);

        // 获取通道数
        int channels = mat.channels();

        // 清空目标图像，对原图像进行边界填充
        Mat tmp;
        dstImg = dstImg.zeros(mat.size(), mat.type());
        int *kernel = new int[ksize*ksize];
        copyMakeBorder(mat, tmp, ksize / 2, ksize / 2, ksize / 2, ksize / 2, BORDER_REPLICATE);
        for (int i = ksize / 2; i < mat.rows + ksize / 2; i++)		//对填充后的图像从有图像区域开始滤波
        {
            for (int j = ksize / 2; j < mat.cols + ksize / 2; j++)
            {
                for (int c = 0; c < channels; c++)
                {
                    // 将核大小的图像填入数组
                    for (int m = 0; m < ksize*ksize; m++) {
                        if (tmp.channels() == 1) {
                            kernel[m] = tmp.ptr<uchar>(i - ksize / 2 + m / ksize, j - ksize / 2 + m % ksize)[c];
                        }
                        else if (tmp.channels() == 3) {
                            kernel[m] = tmp.ptr<Vec3b>(i - ksize / 2 + m / ksize, j - ksize / 2 + m % ksize)->val[c];
                        }
                        else
                        {
                            delete[]kernel;
                            return dstImg;
                        }

                    }

                    // 排序
                    Sort(kernel, ksize*ksize);
                    // 将中值写入目标图像
                    if (tmp.channels() == 1) {
                        dstImg.ptr<uchar>(i - ksize / 2, j - ksize / 2)[c] = kernel[(ksize*ksize) / 2];
                    }
                    else if (tmp.channels() == 3) {
                        dstImg.ptr<Vec3b>(i - ksize / 2, j - ksize / 2)->val[c] = kernel[(ksize*ksize) / 2];
                    }

                }
            }
        }
        delete[]kernel;
        return dstImg;
}
//分离实现高斯滤波
//O(m*n*k)
Mat processer::separateGaussianFilter(Mat mat,int ksize, double sigma)
{
    Mat dstImage;
    assert(mat.channels()==1 || mat.channels() == 3); //只处理单通道或者三通道图像
    //生成一维的
    double *matrix = new double[ksize];
    double sum = 0;
    int origin = ksize / 2;
    for(int i = 0; i < ksize; i++)
    {
        double g = exp(-(i-origin) * (i-origin) / (2 * sigma * sigma));
        sum += g;
        matrix[i] = g;
    }
    for(int i = 0; i < ksize; i++)
    {
        matrix[i] /= sum;
    }
    int border = ksize / 2;
    copyMakeBorder(mat, dstImage, border, border, border, border, BORDER_CONSTANT);
    int channels = dstImage.channels();
    int rows = dstImage.rows - border;
    int cols = dstImage.cols - border;
    //水平方向
    for(int i = border; i < rows; i++)
    {
        for(int j = border; j < cols; j++)
        {
            double sum[3] = {0};
            for(int k = -border; k<=border; k++)
            {
                if(channels == 1)
                {
                    sum[0] += matrix[border + k] * dstImage.at<uchar>(i, j+k);
                }
                else if(channels == 3)
                {
                    Vec3b rgb = dstImage.at<Vec3b>(i, j+k);
                    sum[0] += matrix[border+k] * rgb[0];
                    sum[1] += matrix[border+k] * rgb[1];
                    sum[2] += matrix[border+k] * rgb[2];
                }
            }
            for(int k = 0; k < channels; k++)
            {
                if(sum[k] < 0) sum[k] = 0;
                else if(sum[k] > 255) sum[k] = 255;
            }
            if(channels == 1)
                dstImage.at<Vec3b>(i, j) = static_cast<uchar>(sum[0]);
            else if(channels == 3){
                Vec3b rgb = {static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2])};
                dstImage.at<Vec3b>(i, j) = rgb;
            }
        }
    }
    //竖直方向
    for(int i = border; i < rows; i++)
    {
        for(int j = border; j < cols; j++)
        {
            double sum[3] = {0};
            for(int k = -border; k<=border; k++)
            {
                if(channels == 1)
                {
                    sum[0] += matrix[border + k] * dstImage.at<uchar>(i+k, j);
                }else if(channels == 3)
                {
                    Vec3b rgb = dstImage.at<Vec3b>(i+k, j);
                    sum[0] += matrix[border+k] * rgb[0];
                    sum[1] += matrix[border+k] * rgb[1];
                    sum[2] += matrix[border+k] * rgb[2];
                }
            }
            for(int k = 0; k < channels; k++)
            {
                if(sum[k] < 0) sum[k] = 0;
                else if(sum[k] > 255) sum[k] = 255;
            }
            if(channels == 1)
                dstImage.at<Vec3b>(i, j) = static_cast<uchar>(sum[0]);
            else if(channels == 3)
            {
                Vec3b rgb = {static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2])};
                dstImage.at<Vec3b>(i, j) = rgb;
            }
        }
    }
    delete [] matrix;
    return dstImage;
}
/****************************************************
 * 接下来的几个函数主要实现几何变换：
 * 平移
 * 旋转
 ***************************************************/
Mat processer::imgTranslation(Mat mat, int xOffset, int yOffset)
{

    int nrows = mat.rows;
    int ncols = mat.cols;
    Mat dst(mat.size(), mat.type());
    //dst.zeros(nrows,ncols,mat.type());
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            //映射变换
            int x = j - xOffset;
            int y = i - yOffset;
            //边界判断
            if (x >= 0 && y >= 0 && x < ncols && y < nrows)
            {
                dst.at<Vec3b>(i, j) = mat.at<Vec3b>(y,x);
            }
        }
    }
    return dst;
}


Mat processer::rotate(Mat mat, double Angle){
    double radian = (double) (Angle /180.0 * CV_PI);
    Mat dst;
    //填充图像
    int maxBorder =(int) (max(mat.cols, mat.rows)* 1.414 ); //即为sqrt(2)*max
    int dx = (maxBorder - mat.cols)/2;
    int dy = (maxBorder - mat.rows)/2;
    copyMakeBorder(mat, dst, dy, dy, dx, dx, BORDER_CONSTANT);

    //旋转
    Point2f center( (double)(dst.cols/2) , (double) (dst.rows/2));
    Mat affine_matrix = getRotationMatrix2D( center, Angle, 1.0 );//求得旋转矩阵
    warpAffine(dst, dst, affine_matrix, dst.size());

    //计算图像旋转之后包含图像的最大的矩形
    double sinVal = abs(sin(radian));
    double cosVal = abs(cos(radian));
    Size targetSize( (int)(mat.cols * cosVal +mat.rows * sinVal),
                     (int)(mat.cols * sinVal + mat.rows * cosVal) );

    //剪掉多余边框
    int x = (dst.cols - targetSize.width) / 2;
    int y = (dst.rows - targetSize.height) / 2;
    Rect rect(x, y, targetSize.width, targetSize.height);
    dst = Mat(dst,rect);
    return dst;

}

Mat processer::enRich(Mat mat, int beta, int alpha){
    Mat dstImage;
    for( int y = 0; y < mat.rows; y++ ) {
            for( int x = 0; x < mat.cols; x++ ) {
                for( int c = 0; c < mat.channels(); c++ ) {
                    dstImage.at<Vec3b>(y,x)[c] =saturate_cast<uchar>( alpha*mat.at<Vec3b>(y,x)[c] + beta );
                }
            }
    }
    return dstImage;
}
/****************************************************
 * 接下来的几个函数主要实现边缘锐化变换：
 * roberts
 * sobel
 * prewitt
 * laplacian
 ***************************************************/
Mat processer::roberts(Mat mat){
    Mat dstImage=mat.clone();
    int lWidth=mat.cols,lHeight=mat.rows;
    // 中间变量
    int v_r_v, v_g_v, v_b_v, g_v;
    int v_r_h, v_g_h, v_b_h, g_h;

    int i,j,k,l;

    // 2X2 模版
    for (i = 0; i < lWidth; i++)		//被处理像素在i列
    {
        for (j = 0; j < lHeight; j++)	//被处理像素在j行
        {
            v_r_v = v_g_v = v_b_v = v_r_h = v_g_h = v_b_h = 0;

            for (k = i - 1; k < i + 1; k++)	//2*2模版
            {
                for (l = j - 1; l < j + 1; l++)
                {
                    // 防止内存溢出
                    if (k >= 0  && l >= 0 && k < lWidth && l < lHeight)
                    {
                        // 检测模版
                        if (k == i - 1 && l == j - 1)
                            g_v = 1;
                        else if (k == i && l == j)
                            g_v = -1;
                        else
                            g_v = 0;

                        if(k == i - 1 && l == j)
                            g_h = -1;
                        else if (k == i && l == j - 1)
                            g_h = 1;
                        else
                            g_h = 0;

                        v_r_v += mat.at<Vec3b>(l,k)[2] * g_v;
                        v_r_h += mat.at<Vec3b>(l,k)[2] * g_h;
                        v_g_v += mat.at<Vec3b>(l,k)[1] * g_v;
                        v_g_h += mat.at<Vec3b>(l,k)[1] * g_h;
                        v_b_v += mat.at<Vec3b>(l,k)[0] * g_v;
                        v_b_h += mat.at<Vec3b>(l,k)[0] * g_h;
                    }
                }
            }
            dstImage.at<Vec3b>(j,i)[2] = (int)sqrt(1.0*v_r_v * v_r_v + v_r_h * v_r_h);
            dstImage.at<Vec3b>(j,i)[1] = (int)sqrt(1.0*v_g_v * v_g_v + v_g_h * v_g_h);
            dstImage.at<Vec3b>(j,i)[0] = (int)sqrt(1.0*v_b_v * v_b_v + v_b_h * v_b_h);
        }
    }
    for(i=0;i<lWidth;i++){
        for(j=0;j<lHeight;j++){
            dstImage.at<Vec3b>(j,i)[0]=dstImage.at<Vec3b>(j,i)[0]+mat.at<Vec3b>(j,i)[0]<255?dstImage.at<Vec3b>(j,i)[0]+mat.at<Vec3b>(j,i)[0]:255;
            dstImage.at<Vec3b>(j,i)[1]=dstImage.at<Vec3b>(j,i)[1]+mat.at<Vec3b>(j,i)[1]<255?dstImage.at<Vec3b>(j,i)[1]+mat.at<Vec3b>(j,i)[1]:255;
            dstImage.at<Vec3b>(j,i)[2]=dstImage.at<Vec3b>(j,i)[2]+mat.at<Vec3b>(j,i)[2]<255?dstImage.at<Vec3b>(j,i)[2]+mat.at<Vec3b>(j,i)[2]:255;
        }
    }
    return dstImage;
}

Mat processer::sobel(Mat mat){
        Mat dstImage=mat.clone();
        int lWidth=mat.cols,lHeight=mat.rows;

        int v_r_v, v_g_v, v_b_v, g_v;
        int v_r_h, v_g_h, v_b_h, g_h;

        int i,j,k,l;

        // 3X3 模版
        for (i = 0; i < lWidth; i++)			//被处理像素在i列
        {
            for (j = 0; j < lHeight; j++)	//被处理像素在j行
            {
                v_r_v = v_g_v = v_b_v = v_r_h = v_g_h = v_b_h = 0;

                for (k = i - 1; k < i + 2; k++)	//3*3模版
                {
                    for (l = j - 1; l < j + 2; l++)
                    {
                        // 防止内存溢出
                        if (k >= 0  && l >= 0 && k < lWidth && l < lHeight)
                        {
                            // 检测模版
                            if (k == i - 1)
                            {
                                if (l == j)
                                    g_v = -2;
                                else
                                    g_v = -1;
                            }
                            if (k == i + 1)
                            {
                                if (l == j)
                                    g_v = 2;
                                else
                                    g_v = 1;
                            }
                            if (k == i)
                                g_v = 0;
                            if (l == j - 1)
                            {
                                if (k == i)
                                    g_h = 2;
                                else
                                    g_h = 1;
                            }
                            if (l == j + 1)
                            {
                                if (k == i)
                                    g_h = -2;
                                else
                                    g_h = -1;
                            }
                            if (l == j)
                                g_h = 0;

                            v_r_v += mat.at<Vec3b>(l,k)[2] * g_v;
                            v_r_h += mat.at<Vec3b>(l,k)[2] * g_h;
                            v_g_v += mat.at<Vec3b>(l,k)[1] * g_v;
                            v_g_h += mat.at<Vec3b>(l,k)[1] * g_h;
                            v_b_v += mat.at<Vec3b>(l,k)[0] * g_v;
                            v_b_h += mat.at<Vec3b>(l,k)[0] * g_h;
                        }
                    }
                }
                dstImage.at<Vec3b>(j,i)[2] = (int)sqrt(1.0*v_r_v * v_r_v + v_r_h * v_r_h);
                dstImage.at<Vec3b>(j,i)[1] = (int)sqrt(1.0*v_g_v * v_g_v + v_g_h * v_g_h);
                dstImage.at<Vec3b>(j,i)[0] = (int)sqrt(1.0*v_b_v * v_b_v + v_b_h * v_b_h);
            }
        }
        for(i=0;i<lWidth;i++){
            for(j=0;j<lHeight;j++){
                dstImage.at<Vec3b>(j,i)[0]=dstImage.at<Vec3b>(j,i)[0]+mat.at<Vec3b>(j,i)[0]<255?dstImage.at<Vec3b>(j,i)[0]+mat.at<Vec3b>(j,i)[0]:255;
                dstImage.at<Vec3b>(j,i)[1]=dstImage.at<Vec3b>(j,i)[1]+mat.at<Vec3b>(j,i)[1]<255?dstImage.at<Vec3b>(j,i)[1]+mat.at<Vec3b>(j,i)[1]:255;
                dstImage.at<Vec3b>(j,i)[2]=dstImage.at<Vec3b>(j,i)[2]+mat.at<Vec3b>(j,i)[2]<255?dstImage.at<Vec3b>(j,i)[2]+mat.at<Vec3b>(j,i)[2]:255;
            }
        }
        return dstImage;
}

Mat processer::prewitt(Mat mat){
    Mat dstImage=mat.clone();
    int lWidth=mat.cols,lHeight=mat.rows;

    int v_r_v, v_g_v, v_b_v, g_v;
    int v_r_h, v_g_h, v_b_h, g_h;

    int i,j,k,l;

        // 3X3 模版
    for (i = 0; i < lWidth; i++)			//被处理像素在i列
    {
        for (j = 0; j < lHeight; j++)	//被处理像素在j行
        {
            v_r_v = v_g_v = v_b_v = v_r_h = v_g_h = v_b_h = 0;

            for (k = i - 1; k < i + 2; k++)	//3*3模版
            {
                for (l = j - 1; l < j + 2; l++)
                {
                    // 防止内存溢出
                    if (k >= 0  && l >= 0 && k < lWidth && l < lHeight)
                    {
                        // 检测模版
                        if (k == i - 1)
                            g_v = -1;
                        if (k == i + 1)
                            g_v = 1;
                        if (k == i)
                            g_v = 0;
                        if (l == j - 1)
                            g_h = 1;
                        if (l == j + 1)
                            g_h = -1;
                        if (l == j)
                            g_h = 0;

                        v_r_v += mat.at<Vec3b>(l,k)[2] * g_v;
                        v_r_h += mat.at<Vec3b>(l,k)[2] * g_h;
                        v_g_v += mat.at<Vec3b>(l,k)[1] * g_v;
                        v_g_h += mat.at<Vec3b>(l,k)[1] * g_h;
                        v_b_v += mat.at<Vec3b>(l,k)[0] * g_v;
                        v_b_h += mat.at<Vec3b>(l,k)[0] * g_h;
                    }
                }
            }
            dstImage.at<Vec3b>(j,i)[2] = (int)sqrt(1.0*v_r_v * v_r_v + v_r_h * v_r_h);
            dstImage.at<Vec3b>(j,i)[1] = (int)sqrt(1.0*v_g_v * v_g_v + v_g_h * v_g_h);
            dstImage.at<Vec3b>(j,i)[0] = (int)sqrt(1.0*v_b_v * v_b_v + v_b_h * v_b_h);
        }
    }
    for(i=0;i<lWidth;i++){
        for(j=0;j<lHeight;j++){
            dstImage.at<Vec3b>(j,i)[0]=dstImage.at<Vec3b>(j,i)[0]+mat.at<Vec3b>(j,i)[0]<255?dstImage.at<Vec3b>(j,i)[0]+mat.at<Vec3b>(j,i)[0]:255;
            dstImage.at<Vec3b>(j,i)[1]=dstImage.at<Vec3b>(j,i)[1]+mat.at<Vec3b>(j,i)[1]<255?dstImage.at<Vec3b>(j,i)[1]+mat.at<Vec3b>(j,i)[1]:255;
            dstImage.at<Vec3b>(j,i)[2]=dstImage.at<Vec3b>(j,i)[2]+mat.at<Vec3b>(j,i)[2]<255?dstImage.at<Vec3b>(j,i)[2]+mat.at<Vec3b>(j,i)[2]:255;
        }
    }
    return dstImage;
}

Mat processer::laplacian(Mat mat){
    Mat dstImage=mat.clone();
    //qDebug()<<"right"<<endl;
    int lWidth=mat.cols,lHeight=mat.rows;

    int i,j,k,l;

    int v_r, v_g, v_b, p_g;

        // 检测模版
        int g[9]={-1, -1, -1, -1, 8, -1, -1, -1, -1};

        // 3X3 模版
        for (i = 0; i < lWidth; i++)		//被处理像素在i列
        {
            for (j = 0; j < lHeight; j++)	//被处理像素在j行
            {
                v_r = v_g = v_b = p_g = 0;

                for (k = i - 1; k < i + 2; k++)	//3*3模版
                {
                    for (l = j - 1; l < j + 2; l++)
                    {
                        // 防止内存溢出
                        if (k >= 0  && l >= 0 && k < lWidth && l < lHeight)
                        {
                                v_r += mat.at<Vec3b>(l,k)[2] * g[p_g];
                                v_g += mat.at<Vec3b>(l,k)[1] * g[p_g];
                                v_b += mat.at<Vec3b>(l,k)[0] * g[p_g];
                                p_g++;
                        }
                    }
                }

                if (v_r < 0)
                    v_r = 0;
                if (v_g < 0)
                    v_g = 0;
                if (v_b < 0)
                    v_b = 0;

               dstImage.at<Vec3b>(j,i)[2] = v_r;
               dstImage.at<Vec3b>(j,i)[1] = v_g;
               dstImage.at<Vec3b>(j,i)[0] = v_b;
            }
        }
        for(i=0;i<lWidth;i++){
            for(j=0;j<lHeight;j++){
                dstImage.at<Vec3b>(j,i)[0]=dstImage.at<Vec3b>(j,i)[0]+mat.at<Vec3b>(j,i)[0]<255?dstImage.at<Vec3b>(j,i)[0]+mat.at<Vec3b>(j,i)[0]:255;
                dstImage.at<Vec3b>(j,i)[1]=dstImage.at<Vec3b>(j,i)[1]+mat.at<Vec3b>(j,i)[1]<255?dstImage.at<Vec3b>(j,i)[1]+mat.at<Vec3b>(j,i)[1]:255;
                dstImage.at<Vec3b>(j,i)[2]=dstImage.at<Vec3b>(j,i)[2]+mat.at<Vec3b>(j,i)[2]<255?dstImage.at<Vec3b>(j,i)[2]+mat.at<Vec3b>(j,i)[2]:255;
            }
        }
       // imshow("right",dstImage);
        return dstImage;
}


Mat processer::AreaGrow(Mat mat,QPoint firstSeed,QPoint secondSeed)
{

    Mat growArea = mat.clone();    //生长区域

    int R_I = mat.at<Vec3b>(firstSeed.y(),firstSeed.x())[2];
    int G_I = mat.at<Vec3b>(firstSeed.y(),firstSeed.x())[1];
    int B_I = mat.at<Vec3b>(firstSeed.y(),firstSeed.x())[0];
    float fY_I = (9798.0f * R_I + 19235.0f * G_I + 3735.0f * B_I) / 32768.0f;

        // 计算种子点二的灰度值
    int R_II = mat.at<Vec3b>(secondSeed.y(),secondSeed.x())[2];
    int G_II = mat.at<Vec3b>(secondSeed.y(),secondSeed.x())[1];
    int B_II = mat.at<Vec3b>(secondSeed.y(),secondSeed.x())[0];
    float fY_II = (9798.0f * R_II + 19235.0f * G_II + 3735.0f * B_II) / 32768.0f;


    int lWidth = mat.cols;

    int lHeight = mat.rows;

    int i,j;
    //LONG lLineBytes = m_pDIB->RowLen();

    // 对各像素进行灰度转换
    for (i = 0; i < lHeight; i ++)
    {
        for (j = 0; j < lWidth; j ++)
        {
            //获取各颜色分量
            int R = mat.at<Vec3b>(i,j)[2];
            int G = mat.at<Vec3b>(i,j)[1];
            int B = mat.at<Vec3b>(i,j)[0];

            //计算当前点灰度值
            float Y = (9798.0 * R + 19235.0 * G + 3735.0 * B) / 32768.0;
            if (abs(Y - fY_I) < abs(Y - fY_II))
            {
                //当前点同种子一灰度值比较接近

                //将种子一的颜色赋给当前像素
                growArea.at<Vec3b>(i,j)[2]= R_I;
                growArea.at<Vec3b>(i,j)[1] = G_I;
                growArea.at<Vec3b>(i,j)[0] = B_I;
            }
            else
            {
                //当前点同种子二灰度值比较接近

                //将种子二的颜色赋给当前像素
                growArea.at<Vec3b>(i,j)[2]= R_II;
                growArea.at<Vec3b>(i,j)[1] = G_II;
                growArea.at<Vec3b>(i,j)[0] = B_II;
            }
        }
    }

   return growArea;
}

Mat processer::IterateThrehold(Mat mat){
    int i, j;
    int nNs_Y[256];
    // 变量初始化
    memset(nNs_Y, 0, sizeof(nNs_Y));

    Mat dstImage=mat.clone();
    int lWidth =mat.cols;
    int lHeight = mat.rows;
    int R,G,B,Y;

    // 对各像素进行灰度转换
    for (i = 0; i < lHeight; i ++)
    {
        for (j = 0; j < lWidth; j ++)
        {
            // 计算灰度值
             R=mat.at<Vec3b>(i,j)[2];
             G=mat.at<Vec3b>(i,j)[1];
             B=mat.at<Vec3b>(i,j)[0];
             Y = (9798 *R + 19235 *G + 3735 *B) / 32768;
            // 灰度统计计数
            nNs_Y[Y]++;
        }
    }
    int T1, T2;
        T1 = 127;
        T2 = 0;

        // 临时变量
        int Temp0, Temp1, Temp2, Temp3;
        Temp0 = Temp1 = Temp2 = Temp3 = 0;

    while (true)
    {
        // 计算下一个迭代阀值
        for (i = 0; i < T1 + 1; i++)
        {
            Temp0 += nNs_Y[i] * i;
            Temp1 += nNs_Y[i];
        }
        for (i = T1 + 1; i < 256; i++)
        {
            Temp2 += nNs_Y[i] * i;
            Temp3 += nNs_Y[i];
        }
        T2 = (Temp0 / Temp1 + Temp2 / Temp3) / 2;

        // 看迭代结果是否已收敛
        if (T1 == T2)
            break;
        else
            T1 = T2;
    }

    // 对各像素进行灰度转换
    for (i = 0; i < lHeight; i ++)
    {
        for (j = 0; j < lWidth; j ++)
        {
            // 读取像素R分量
            int R1 = mat.at<Vec3b>(i,j)[2];

            // 判断R分量是否超出范围
            if (R1 < T1)
                R1 = 0;
            else
                R1 = 255;

            // 回写处理完的R分量
            dstImage.at<Vec3b>(i,j)[2] = R1;


            // 读取像素G分量
            int G1=mat.at<Vec3b>(i,j)[1];

            // 判断G分量是否超出范围
            if (G1 < T1)
                G1 = 0;
            else
                G1 = 255;

            // 回写处理完的G分量
            dstImage.at<Vec3b>(i,j)[1] = G1;


            // 读取像素B分量
           int B1=mat.at<Vec3b>(i,j)[0];

            // 判断B分量是否超出范围
            if (B < T1)
                B = 0;
            else
                B = 255;

            // 回写处理完的B分量
            dstImage.at<Vec3b>(i,j)[0] = B1;
        }
    }
    return dstImage;
}

Mat processer::targetseg(Mat mat){
    Mat foreground = Mat::zeros(mat.size(), mat.type());//前景图
    Mat mask = Mat::zeros(mat.size(), CV_8UC1);
    imshow("OrignalImage",mat);
    waitKey(30);
    Rect rect = selectROI("OrignalImage", mat, false);
    Mat bgdModel, fgdModel;
    grabCut(mat, mask, rect, bgdModel, fgdModel, 5, GC_INIT_WITH_RECT);

    Mat result = Mat::zeros(mat.size(), CV_8UC3);
    for (int row = 0; row < result.rows; row++)
    {
        for (int col = 0; col < result.cols; col++)
        {
            //如果掩膜mask的某个位置上像素值为1或3，也就是明显前景和可能前景，就把原图像中该位置的像素值赋给结果图像
            if (mask.at<uchar>(row, col) == 1 || mask.at<uchar>(row, col) == 3)
            {
                result.at<Vec3b>(row, col) = Vec3b(255,255,255);
            }
        }
    }
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(result, result, MORPH_OPEN, kernel);
    bitwise_and(result, mat, foreground);
    return foreground;
}

Mat processer::replaceback(Mat mat,Mat background){
        resize(background, background, mat.size());
        //背景虚化
        //对背景图像进行高斯模糊
        Mat gaus_background = Mat::zeros(background.size(), background.type());
        GaussianBlur(background, gaus_background, Size(), 2, 2);
        for (int row = 0; row < gaus_background.rows; row++)
        {
            for (int col = 0; col < gaus_background.cols; col++)
            {
                if (mat.at<Vec3b>(row, col) != Vec3b(0, 0, 0))
                {
                    //将锐化图像前景和虚化背景图像混合
                    gaus_background.at<Vec3b>(row, col) = mat.at<Vec3b>(row, col);
                }
            }
        }
        //对虚化图像再进行高斯模糊，抵消前景的边缘效应和锐化效果
        GaussianBlur(gaus_background, gaus_background, Size(3, 3), 1);
        return gaus_background;
}

Mat processer::foggy(Mat mat){
    Mat img_f=mat.clone();
    //imshow("img",img_f);
    int w=mat.cols,h=mat.rows;
    int i,j;
    float A=0.6;        //亮度
    float beta=0.01;    //雾的浓度
    float size=sqrt(max(h,w));    //雾化尺寸
    float d,td;
    Point center(int(h/2),int(w/2));
    for(i=0;i<h;i++){
        for(j=0;j<w;j++){
            d=-0.04*sqrt(pow((i-center.y),2)+pow((j-center.x),2))+size;
            td=exp(-beta*d);
            img_f.at<Vec3b>(i,j)[0]=img_f.at<Vec3b>(i,j)[0]*td+A*(1-td);
            img_f.at<Vec3b>(i,j)[1]=img_f.at<Vec3b>(i,j)[1]*td+A*(1-td);
            img_f.at<Vec3b>(i,j)[2]=img_f.at<Vec3b>(i,j)[2]*td+A*(1-td);
        }
    }
    //imshow("img",img_f);
     return img_f;
}

Mat processer::fudiao(Mat mat){
    Mat dstImage=mat.clone();
    int h=mat.rows,w=mat.cols;
    int i,j;
    for(i=0;i<h;i++){
        for(j=0;j<w;j++){
            if(i-1>=0&&j-1>=0)
            {
                dstImage.at<Vec3b>(i,j)[0]=mat.at<Vec3b>(i,j)[0]-mat.at<Vec3b>(i-1,j-1)[0]+128;
                dstImage.at<Vec3b>(i,j)[1]=mat.at<Vec3b>(i,j)[1]-mat.at<Vec3b>(i-1,j-1)[1]+128;
                dstImage.at<Vec3b>(i,j)[2]=mat.at<Vec3b>(i,j)[2]-mat.at<Vec3b>(i-1,j-1)[2]+128;
            }
        }

    }
return dstImage;
}

Mat processer::addedge(Mat mat, Mat edge){
   // qDebug()<<"测试点1"<<endl;
    resize(edge,edge,mat.size());
   // qDebug()<<"测试点2"<<endl;
    Mat dstImage=mat.clone();
    int h=mat.rows,w=mat.cols;
    int i,j;
    for(i=0;i<h;i++){
        for(j=0;j<w;j++){
           if(edge.at<Vec3b>(i,j)==Vec3b(0,0,0)){
               dstImage.at<Vec3b>(i,j)=mat.at<Vec3b>(i,j);
               //qDebug()<<"内容"<<endl;
           }
           else{
               dstImage.at<Vec3b>(i,j)=edge.at<Vec3b>(i,j);
               //qDebug()<<"边框"<<endl;
           }
        }
    }
    //imshow("dst",dstImage);
    return dstImage;
}


//Mat matStatic2;
//Rect roirect1;
//Point startPoint1;
//Point endPoint1;

//void processer::MouseEvent(int event, int x, int y, int flags, void*)
//{

//    if (event == CV_EVENT_LBUTTONDOWN)
//    {
//        startPoint1 = Point(x, y);
//    }
//    else if (event==CV_EVENT_MOUSEMOVE && (flags&CV_EVENT_FLAG_LBUTTON))
//    {
//        endPoint1=Point(x,y);
//        Mat tempImage = temp.clone();
//        //rectangle(src,原点,终点,linecolor,linewidth,linetype,0);
//        //rectangle(src,Rect(原点,width,height),linecolor,linewidth,linetype,0);
//        rectangle(tempImage,startPoint1,endPoint1,Scalar(250,0,100),2,8,0);
//        //rectangle(matStatic, startPoint, endPoint, Scalar(255, 0, 0), 2, 8, 0);
//        imshow("OriginalImage",tempImage);
//    }
//    roirect1.width = abs(endPoint1.x - startPoint1.x);
//    roirect1.height = abs(endPoint1.y - startPoint1.y);
//    if (roirect1.width > 0 && roirect1.height > 0)
//    {
//        roirect1.x = min(startPoint1.x, endPoint1.x);
//        roirect1.y = min(startPoint1.y, endPoint1.y);
//        Mat roiMat = temp(Rect(roirect1.x, roirect1.y, roirect1.width, roirect1.height));
//        if(event==CV_EVENT_RBUTTONDOWN){
//             matStatic2=roiMat;
//        }
//    }
//}

//Mat processer::getinfo(){
//    //setMouseCallback("OriginalImage",MouseEvent,0);
//}

