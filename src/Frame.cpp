#include <Frame.h>

using namespace cv;
using namespace cv::line_descriptor;
using namespace DBoW2;
Frame::Frame() {
}
Frame::Frame(std::string imgFileName, int id) {
    m_imageName = imgFileName;
    m_id = id;
    m_img = imread(imgFileName, 0);
    if(m_img.empty()) {
        printf("load image %s fail\n", m_imageName.c_str());
    }
    m_width = m_img.cols;
    m_height = m_img.rows;
}

void Frame::detectLineFeatures() {
    if(m_img.empty()) {
        printf("image %d empty, skip line detection\n", m_id);
        return;
    }
    Ptr<BinaryDescriptor> lbd = BinaryDescriptor::createBinaryDescriptor();
    lbd->detect(m_img, m_keyLines/*, mask0*/);
    cv::Mat lineDescriptors;
    lbd->compute(m_img, m_keyLines, lineDescriptors);
    if(lineDescriptors.rows < 50) {
        printf("image %d: number of line features low: %d features\n", m_id, lineDescriptors.rows);
    }
    m_lineFeatureDescs.resize(lineDescriptors.rows);
    for(size_t j = 0; j < m_lineFeatureDescs.size(); ++j) {
        m_lineFeatureDescs[j] = lineDescriptors.row(j);
    }
}

void Frame::detectOrbFeatures() {
    if(m_img.empty()) {
        printf("image %d empty, skip orb detection\n", m_id);
        return;
    }
    //Ptr<BinaryDescriptor> lbd = BinaryDescriptor::createBinaryDescriptor();
    cv::Ptr<cv::ORB> orb = cv::ORB::create(800, 1.2, 4);
    cv::Mat orbDescriptors;
    orb->detectAndCompute(m_img, cv::Mat(), m_keyPts, orbDescriptors, false);
    if(orbDescriptors.rows < 100) {
        printf("image %d: number of orb features low: %d features\n", m_id, orbDescriptors.rows);
    }
    m_orbFeatureDescs.resize(orbDescriptors.rows);
    for(size_t j = 0; j < m_orbFeatureDescs.size(); ++j) {
        m_orbFeatureDescs[j] = orbDescriptors.row(j);
    }
}

void Frame::setGroundTruth(const std::vector<double> &pose) {
    m_pose = pose;
}

void Frame::writeToFile(std::string outFileName) {
    //Write serialization for this class
    FileStorage fs(outFileName, FileStorage::WRITE);
    fs << "f_id" << m_id;
    fs << "f_width" << m_width;
    fs << "f_height" << m_height;
    fs << "f_imageName" << m_imageName;
    fs << "f_pose" << m_pose;
    //fs << "f_keyPts" << m_keyPts;
    //fs << "f_keyLines" << m_keyLines; //no matching call for this
    fs << "f_orbFeatureDescs" << m_orbFeatureDescs;
    fs << "f_lineFeatureDescs" << m_lineFeatureDescs;
    fs.release();
}

void Frame::loadFromFile(std::string inFileName) {
    FileStorage fs(inFileName, FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "Failed to open " << inFileName << std::endl;
        return;
    }
    m_id = (int) fs["f_id"];
    m_width = (int)fs["f_width"];
    m_height = (int)fs["f_height"];
    m_imageName = (std::string)fs["f_imageName"];
    FileNode fileNode = fs["f_pose"];
    cv::read(fileNode, m_pose);
    fileNode = fs["f_orbFeatureDescs"];
    cv::read(fileNode, m_orbFeatureDescs);
    fileNode = fs["f_lineFeatureDescs"];
    cv::read(fileNode, m_lineFeatureDescs);
    fs.release();
}
