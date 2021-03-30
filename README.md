# license-plate-detect-recoginition-opencv
使用opencv部署深度学习车牌检测与识别
python版本的主程序是detect_rec_img.py，c++版本的主程序是main.cpp

由于opencv无法在图片上写汉字，因此在C++版本的程序里，把车牌识别结果在终端打印

注意，我在opencv4.5.1的环境里编写的c++程序能编译运行，
但是换一台电脑，opencv环境是4.5.0的，程序运行出错了，因此opencv版本最好是安装目前最新的4.5.1
