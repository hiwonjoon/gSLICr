#include "gSLICr_Lib/gSLICr.h"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void change_image_format(const Mat& inimg, gSLICr::UChar4Image* outimg)
{
	gSLICr::Vector4u* outimg_ptr = outimg->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < outimg->noDims.y;y++)
		for (int x = 0; x < outimg->noDims.x; x++)
		{
			int idx = x + y * outimg->noDims.x;
			outimg_ptr[idx].b = inimg.at<Vec3b>(y, x)[0];
			outimg_ptr[idx].g = inimg.at<Vec3b>(y, x)[1];
			outimg_ptr[idx].r = inimg.at<Vec3b>(y, x)[2];
		}
}

void change_image_format(const gSLICr::UChar4Image* inimg, Mat& outimg)
{
	const gSLICr::Vector4u* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < inimg->noDims.y; y++)
		for (int x = 0; x < inimg->noDims.x; x++)
		{
			int idx = x + y * inimg->noDims.x;
			outimg.at<Vec3b>(y, x)[0] = inimg_ptr[idx].b;
			outimg.at<Vec3b>(y, x)[1] = inimg_ptr[idx].g;
			outimg.at<Vec3b>(y, x)[2] = inimg_ptr[idx].r;
		}
}

void change_image_format(const gSLICr::IntImage* inimg, Mat& outimg)
{
	const int* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < inimg->noDims.y; y++)
		for (int x = 0; x < inimg->noDims.x; x++)
		{
			int idx = x + y * inimg->noDims.x;
			outimg.at<unsigned short>(y, x) = inimg_ptr[idx];
		}
}

#include <boost/python.hpp>
using namespace boost::python;

class gSLIC_py_module {
public :
    gSLIC_py_module() {
        my_settings.img_size.x = 960;
        my_settings.img_size.y = 540;
        my_settings.no_segs = 2000;
        //my_settings.spixel_size = 16;
        my_settings.coh_weight = 0.6f;
        my_settings.no_iters = 5;
        my_settings.color_space = gSLICr::RGB; // gSLICr::CIELAB for Lab, or gSLICr::RGB for RGB or gSLICr::XYZ
        my_settings.seg_method = gSLICr::GIVEN_NUM; // or gSLICr::GIVEN_NUM for given number or gSLICr::GIVEN_NUM
        my_settings.do_enforce_connectivity = true; // wheter or not run the enforce connectivity step
        gSLICr_engine = new gSLICr::engines::core_engine(my_settings);
    }
    ~gSLIC_py_module() {
        delete gSLICr_engine;
    }

    Mat depth_seg(Mat depth_im, double maximum_depth) {
        // gSLICr takes gSLICr::UChar4Image as input and out put
        gSLICr::UChar4Image* in_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);

        Size s(my_settings.img_size.x, my_settings.img_size.y);
        Mat seg_result;
        seg_result.create(s, CV_16U);

        //Colorize Depth
        Mat fit_to_8u, colorized_depth_im;
        double min, max;
        minMaxIdx(depth_im, &min, &max);
        depth_im.convertTo(fit_to_8u, CV_8UC1, 255 / (maximum_depth-min), -min);
        applyColorMap(fit_to_8u, colorized_depth_im, cv::COLORMAP_AUTUMN);

        //Process
        change_image_format(colorized_depth_im, in_img);

        gSLICr_engine->Process_Frame(in_img);
        const gSLICr::IntImage * result = gSLICr_engine->Get_Seg_Res();

        change_image_format(result, seg_result);

        delete in_img;

        return seg_result;
    }

    Mat rgb_seg(Mat rgb_im) {
        // gSLICr takes gSLICr::UChar4Image as input and out put
        gSLICr::UChar4Image* in_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);

        Size s(my_settings.img_size.x, my_settings.img_size.y);
        Mat seg_result;
        seg_result.create(s, CV_16U);

        //Process
        change_image_format(rgb_im, in_img);

        gSLICr_engine->Process_Frame(in_img);
        const gSLICr::IntImage * result = gSLICr_engine->Get_Seg_Res();

        change_image_format(result, seg_result);

        delete in_img;

        return seg_result;
    }


private :
	// instantiate a core_engine
	gSLICr::engines::core_engine* gSLICr_engine;
    gSLICr::objects::settings my_settings;
};

BOOST_PYTHON_MODULE(gSLIC)
{
    class_<gSLIC_py_module>("gSLIC")
        .def("depth_seg",&gSLIC_py_module::depth_seg)
        .def("rgb_seg",&gSLIC_py_module::rgb_seg)
    ;
}

