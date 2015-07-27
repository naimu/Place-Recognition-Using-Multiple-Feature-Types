/**
 * File: Frame.h
 * Date: November 2011
 * Author: Wei Mou
 * Description: Frame class
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_T_FRAME_H__
#define __D_T_FRAME_H__

#include <opencv2/core/core.hpp>
#include <vector>
#include <string>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace DBoW2 {

    enum DETECTOR_TYPE {
        LINE,
        ORB,
        LINEORB
    };
    class Frame
    {
        public:
            Frame();
            Frame(std::string imgFileName, DETECTOR_TYPE detectorType = LINEORB);
            Frame(std::string imgFileName, int id);
            void detectLineFeatures();
            void detectOrbFeatures();
            void loadFromFile(std::string filename);
            void writeToFile(std::string filename);
            void setGroundTruth(const std::vector<double> &pose);
            std::vector<double> getGroundTruth() {return m_pose;};
            unsigned long getOrbFeatureSize(){return m_orbFeatureDescs.size();};
            unsigned long getLineFeatureSize(){return m_lineFeatureDescs.size();};
            std::vector<cv::Mat> getOrbFeatureDescs() {return m_orbFeatureDescs;};
            std::vector<cv::Mat> getLineFeatureDescs() {return m_lineFeatureDescs;};
            int getID() {return m_id;};
        private:
            std::vector<cv::Mat> m_orbFeatureDescs;
            std::vector<cv::Mat> m_lineFeatureDescs;
            std::vector<cv::KeyPoint> m_keyPts;
            std::vector<cv::line_descriptor::KeyLine> m_keyLines;
            std::string m_imageName;
            std::vector<uchar> m_imgDescOrb;
            std::vector<uchar> m_imgDescLine;
            cv::Mat m_img;
            std::vector<double> m_pose;
            int m_id;
            int m_width;
            int m_height;
            DETECTOR_TYPE m_detectorType;
    };
} // namespace DBoW2
#endif

