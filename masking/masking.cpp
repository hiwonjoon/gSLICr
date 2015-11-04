#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "./masking.h"

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

        std::cout << "Input points: " << cloud->points.size() << std::endl;
        std::cout << "Filtered points: " << inliers->indices.size() << std::endl;

        *cloud_out = *cloud_filtered;
    }

    pcl::PointCloud<pcl::PointXYZRGB> cloudFromRGBD(cv::Mat depth, cv::Mat rgb,
                       float cx, float cy,
                       float fx, float fy,
                       float max_depth) {
        int width = depth.cols;
        int height = depth.rows;

        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        cloud.height = height;
        cloud.width = width;
        cloud.reserve(height * width);

        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                float z = static_cast<float>(depth.at<int>(h,w)) / 1000.0f;
                if (z >= 0.2f && z < max_depth) {
                    float x = (static_cast<float>(w) - cx) * z / fx;
                    float y = (static_cast<float>(h) - cy) * z / fy;

                    pcl::PointXYZRGB point;

                    point.x = x;
                    point.y = y;
                    point.z = z;
                    point.r = rgb.at<cv::Vec3b>(h,w)[0];
                    point.g = rgb.at<cv::Vec3b>(h,w)[1];
                    point.b = rgb.at<cv::Vec3b>(h,w)[2];

                    cloud.push_back(point);
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

        for (size_t i = 0; i < cloud.size(); ++i) {
            float ptx = cloud.points[i].x;
            float pty = cloud.points[i].y;
            float ptz = cloud.points[i].z;

            if (isnan(ptx))
                continue;

            int x = static_cast<int>(ptx * fx / ptz + cx + 0.5f);
            int y = static_cast<int>(pty * fy / ptz + cy + 0.5f);
            float z = ptz * 1000.0f;

            if (x >= 0 && y >= 0 && x <= 960 && y <= 540)
                depth_temp.at<float>(y,x) = z;
        }

        depth = depth_temp;
    }

    cv::Mat maskGenerator(cv::Mat depth, int filter_size) {
        cv::Mat mask = cv::Mat::zeros(540, 960, CV_32F);
        cv::Mat kernel = cv::Mat::ones(filter_size, filter_size, CV_32F);
        cv::Mat binary;

        cv::filter2D(depth, mask, -1, kernel);
        cv::threshold(mask, binary, 0.1, 1.0, CV_THRESH_BINARY);
        return binary;
    }

    void superVoxelClustering(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                              pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_out) {
        float voxel_resolution = 0.008f;
        float seed_resolution = 0.1f;
        float color_importance = 0.2f;
        float spatial_importance = 0.4f;
        float normal_importance = 1.0f;
        bool use_single_cam_transform = false;

        pcl::SupervoxelClustering<pcl::PointXYZRGB> super(voxel_resolution, seed_resolution, use_single_cam_transform);
        super.setInputCloud(cloud);
        super.setColorImportance(color_importance);
        super.setSpatialImportance(spatial_importance);
        super.setNormalImportance(normal_importance);

        std::map<uint32_t, pcl::Supervoxel<pcl::PointXYZRGB>::Ptr> supervoxel_clusters;
        super.extract(supervoxel_clusters);

        std::cout << "# of supervoxels : " << supervoxel_clusters.size() << std::endl;

        pcl::PointCloud<pcl::PointXYZL>::Ptr sv_labeled_cloud = super.getLabeledCloud();
        *cloud_out = *sv_labeled_cloud;
    }

    cv::Mat superVoxelToImage(pcl::PointCloud<pcl::PointXYZL>::Ptr cloud) {
        cv::Mat image = cv::Mat::zeros(540, 960, CV_32S);

        float cx = 479.75f;
        float cy = 269.75f;
        float fx = 540.686f;
        float fy = 540.686f;

        for (int i = 0; i < cloud->points.size(); ++i) {
            float ptx = cloud->points[i].x;
            float pty = cloud->points[i].y;
            float ptz = cloud->points[i].z;
            int x = static_cast<int>(ptx * fx / ptz + cx + 0.5f);
            int y = static_cast<int>(pty * fy / ptz + cy + 0.5f);
            int label = cloud->points[i].label;
            image.at<int>(y,x) = label;
        }

        return image;
    }
}
