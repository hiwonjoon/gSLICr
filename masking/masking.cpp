#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "masking.h"

namespace masking{
    using namespace masking;

    void removePlane(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out,
                     double threshold) {
        // Estimate normal
        pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);

        ne.setInputCloud(cloud);
        ne.setSearchMethod(tree);
        ne.setRadiusSearch(0.03);
        ne.compute(*cloud_normals);

        // Segment plane
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::SACSegmentationFromNormals<pcl::PointXYZRGB, pcl::Normal> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(threshold);

        seg.setInputCloud(cloud);
        seg.setInputNormals(cloud_normals);
        seg.segment(*inliers, *coefficients);

        // Remove plane
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::ExtractIndices<pcl::PointXYZRGB> extract(true);
        extract.setNegative(true);
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.filter(*cloud_filtered);

        *cloud_out = *cloud_filtered;
    }

    pcl::PointCloud<pcl::PointXYZRGB> cloudFromRGBD(cv::Mat depth, cv::Mat rgb,
                       float cx, float cy,
                       float fx, float fy,
                       float max_depth) {
        int width = rgb.cols;
        int height = rgb.rows;

        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        cloud.height = height;
        cloud.width = width;
        cloud.resize(height * width);

        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                float z = static_cast<float>(depth.at<uint16_t>(h,w)) / 1000.0f;
                if (z > 0.0f && z < max_depth) {
                    float x = (static_cast<float>(w) - cx) * z / fx;
                    float y = (static_cast<float>(h) - cy) * z / fy;
                    cloud.points[h*height + w].x = x;
                    cloud.points[h*height + w].y = y;
                    cloud.points[h*height + w].z = z;
                    cloud.points[h*height + w].b = rgb.at<cv::Vec3b>(h,w)[0];
                    cloud.points[h*height + w].g = rgb.at<cv::Vec3b>(h,w)[1];
                    cloud.points[h*height + w].r = rgb.at<cv::Vec3b>(h,w)[2];
                }
            }
        }

        return cloud;
    }

    void rgbdFromCloud(
            pcl::PointCloud<pcl::PointXYZRGB> cloud,
            float cx, float cy,
            float fx, float fy,
            cv::Mat &depth) {
        cv::Mat depth_temp = cv::Mat::zeros(540, 960, CV_32FC1);

        for (int i = 0; i < cloud.size(); ++i) {
            float ptx = cloud.points[i].x;
            float pty = cloud.points[i].y;
            float ptz = cloud.points[i].z;

            if (isnan(ptx))
                continue;

            int x = static_cast<int>(ptx * fx / ptz + cx + 0.5f);
            int y = static_cast<int>(pty * fy / ptz + cy + 0.5f);
            float z = ptz * 1000.0f;

            if (x >= 0 && y >= 0 && x <= 540 && y <= 960)
                depth_temp.at<float>(y,x) = z;
        }

        depth = depth_temp;
    }

    cv::Mat maskGenerator(cv::Mat depth) {
        cv::Mat mask = cv::Mat::zeros(540,960, CV_32S);
        cv::Mat kernel = cv::Mat::ones(7, 7, CV_32F);

        cv::filter2D(depth, mask, -1, kernel);
        return mask;
    }

    cv::Mat removeTable(cv::Mat rgb, cv::Mat depth) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        *cloud = cloudFromRGBD(depth, rgb, 479.75f, 269.75f, 540.686f, 540.686f, 2.5f);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_sampled(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        sor.setInputCloud(cloud);
        sor.setLeafSize(0.007f, 0.007f, 0.007f); // pick one pixel in 7mm X 7mm X 7mm voxel
        sor.filter(*cloud_sampled);

        // Remove plane
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filter1(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filter2(new pcl::PointCloud<pcl::PointXYZRGB>);
        removePlane(cloud_sampled, filter1, 0.1);
        removePlane(filter1, filter2, 0.05);

        cv::Mat depth_filtered;
        rgbdFromCloud(*filter2, 479.25f, 269.75f, 540.686f, 540.686f, depth_filtered);

        cv::Mat mask = maskGenerator(depth_filtered);

        return mask;
    }
}
