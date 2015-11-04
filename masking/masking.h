#ifndef _MASKING_H_
#define _MASKING_H_

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <opencv2/core/core.hpp>

namespace masking{
    void removePlane(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out,
                     double threshold);
    pcl::PointCloud<pcl::PointXYZRGB> cloudFromRGBD(cv::Mat depth, cv::Mat rgb,
                       float cx, float cy,
                       float fx, float fy,
                       float max_depth);
    void rgbdFromCloud(pcl::PointCloud<pcl::PointXYZRGB> cloud,
                       float cx, float cy,
                       float fx, float fy,
                       cv::Mat &depth);
    cv::Mat maskGenerator(cv::Mat depth, int filter_size);
    void superVoxelClustering(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                              pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_out);
    cv::Mat superVoxelToImage(pcl::PointCloud<pcl::PointXYZL>::Ptr cloud);
}
#endif
