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
    gSLIC_py_module(int x, int y, int no_segs) {
        my_settings.img_size.x = y;
        my_settings.img_size.y = x;
        my_settings.no_segs = no_segs;
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

    Mat depth_colorize(Mat depth_im, unsigned short  max_depth, unsigned short min_depth) {
        Size s = depth_im.size();
        Mat fit_to_8u, colorized_depth_im;
        for (int y = 0; y < s.height; y++) {
          for (int x = 0; x < s.width; x++) {
            if( depth_im.at<unsigned int>(y,x) < min_depth )
              depth_im.at<unsigned int>(y,x) = min_depth;
          }
        }
        depth_im.convertTo(fit_to_8u, CV_8UC1, 255.0 / (max_depth-min_depth), 255.0 / (max_depth-min_depth) * (-min_depth));
        applyColorMap(fit_to_8u, colorized_depth_im, cv::COLORMAP_HSV);

        return colorized_depth_im;
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

    cv::Point3f projectPoint(int x, int y, int z) {
        float z_3d = static_cast<float>(z);
        float x_3d = (static_cast<float>(x) - 479.75f) * z_3d / 540.686f;
        float y_3d = (static_cast<float>(y) - 269.75f) * z_3d / 540.686f;
        cv::Point3f point(x_3d, y_3d, z_3d);
        return point;
    }

    std::vector<Mat> adjacency_info(const Mat& seg_result, const Mat& depth ) {
        double max;
        cv::minMaxLoc(seg_result, NULL, &max);

        Mat edges( (int)(max)+1,(int)(max)+1, CV_8U, Scalar(0) );

        Mat nodes( (int)(max)+1, 1+3+3, CV_32F, Scalar(0.0) );

        Size s = seg_result.size();
        for (int y = 0; y < s.height; y++) {
          for (int x = 0; x < s.width; x++) {
            cv::Point3f point = projectPoint( x,y, depth.at<unsigned int>(y,x) );

            nodes.at<float>(seg_result.at<unsigned short>(y,x),0) += 1;
            nodes.at<float>(seg_result.at<unsigned short>(y,x),1) += point.x;
						nodes.at<float>(seg_result.at<unsigned short>(y,x),2) += point.y;
						nodes.at<float>(seg_result.at<unsigned short>(y,x),3) += point.z;

            if( y < s.height-1 && 
                seg_result.at<unsigned short>(y,x) != seg_result.at<unsigned short>(y+1,x) )
            {
                edges.at<unsigned char>( seg_result.at<unsigned short>(y,x),seg_result.at<unsigned short>(y+1,x) ) = 1;
                edges.at<unsigned char>( seg_result.at<unsigned short>(y+1,x),seg_result.at<unsigned short>(y,x) ) = 1;
            }
            if( x < s.width-1 && 
                seg_result.at<unsigned short>(y,x) != seg_result.at<unsigned short>(y,x+1) )
            {
                edges.at<unsigned char>( seg_result.at<unsigned short>(y,x),seg_result.at<unsigned short>(y,x+1) ) = 1;
                edges.at<unsigned char>( seg_result.at<unsigned short>(y,x+1),seg_result.at<unsigned short>(y,x) ) = 1;
            }

          }
        }

        for( int i = 0; i < (int)(max)+1; ++i) {
            if( nodes.at<float>(i,0) != 0 ) 
            {
              nodes.at<float>(i,4) = nodes.at<float>(i,1) / nodes.at<float>(i,0);
              nodes.at<float>(i,5) = nodes.at<float>(i,2) / nodes.at<float>(i,0);
              nodes.at<float>(i,6) = nodes.at<float>(i,3) / nodes.at<float>(i,0);
            }
        }

        std::vector<Mat> ret;
        ret.push_back(edges);
        ret.push_back(nodes);
        return ret;
    }

private :
	// instantiate a core_engine
	gSLICr::engines::core_engine* gSLICr_engine;
    gSLICr::objects::settings my_settings;
};

BOOST_PYTHON_MODULE(gSLIC)
{
    class_<gSLIC_py_module>("gSLIC", init<int,int,int>() )
        .def("rgb_seg",&gSLIC_py_module::rgb_seg)
        .def("depth_colorize",&gSLIC_py_module::depth_colorize)
        .def("adjacency_info",&gSLIC_py_module::adjacency_info)
    ;
}

