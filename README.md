# ImageProcess
实现图像美化系统，系统要求有友好的操作界面，并具有以下功能：
1. 获取图像，可以通过摄像头采集，或直接读取现有的图片，或同时实现两种功能。
2. 可以读取、显示和存储的文件类型包括BMP 文件、JPG 文件和png文件。
3. 具有对图像进行预处理的功能，考虑加噪、去噪、几何变换（至少包括旋转、平移）和裁剪等功能。至少包含2种噪声（如高斯噪声、椒盐噪声、均匀随机噪声等）。
4. 具有图片增强功能，包括亮度、对比度、饱和度的调整
5. 具有对图像进行边缘锐化和描边（分割）的功能。
6. 具有对图像中指定目标进行分割，并且分割出来的目标可以进行交互编辑。
7. 具有图片特殊美化处理功能，例如加框、拼图、雾化、浮雕等，可自行设置2-3个功能。
8. 具有对指定区域进行特征描述和测量功能，至少包含矩形度、圆形性、球状度、重心、周长等。

这次大作业是根据课程所学，制作一款数字图像处理系统。该系统基于QT与OpenCv。
