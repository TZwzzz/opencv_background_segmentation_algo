#include <opencv2/opencv.hpp>
#include <opencv2/bgsegm.hpp>
using namespace cv;

int main()
{
    VideoCapture cap("../cup.mp4");
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    Ptr<bgsegm::BackgroundSubtractorGMG> fgbg = bgsegm::createBackgroundSubtractorGMG();
    
    while (true)
    {
        Mat frame, fgmask;
        bool ret = cap.read(frame);
        if (!ret) break;

        // 应用背景建模算法
        fgbg->apply(frame, fgmask);

        // 开运算处理
        morphologyEx(fgmask, fgmask, MORPH_OPEN, kernel);

        // 显示结果
        imshow("frame", fgmask);

        // 等待键盘输入
        int k = waitKey(30) & 0xff;
        if (k == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
