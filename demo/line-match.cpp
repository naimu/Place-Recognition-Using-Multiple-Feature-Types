#include <opencv2/opencv.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
// DBoW2

#include "DBoW2.h" // defines Surf64Vocabulary and Surf64Database

//#include "DUtils.h"
//#include "DUtilsCV.h" // defines macros CVXX
#include "DVision.h"

using namespace cv;
using namespace cv::line_descriptor;
using namespace std;


// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<vector<float> > > &features, string dir)
{
  features.clear();
  features.reserve(NIMAGES);

  cv::SURF surf(400, 4, 2, EXTENDED_SURF);

  char fileName[256];
  cout << "Extracting SURF features..." << endl;
  for(int i = 0; i < NIMAGES; ++i)
  {
//    stringstream ss;
//    ss << "images/image" << i << ".png";
    sprintf(fileName, "%06d.png", i);
    string imgName = dir + "image_0/" + fileName;

    cv::Mat image = cv::imread(imgName, 0);
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    vector<float> descriptors;

    surf(image, mask, keypoints, descriptors);

    features.push_back(vector<vector<float> >());
    changeStructure(descriptors, features.back(), surf.descriptorSize());
  }
}
int main(int *argc, char ** argv){

    string dir = argv[1];
    char fileName0[256];
    sprintf(fileName0, "%s%06d.png", dir.c_str(), 0);

    char fileName1[256];
    sprintf(fileName1, "%s%06d.png", dir.c_str(), 2);
    Mat img0 = imread(fileName0, 1);
    if(img0.empty()) {
        printf("failed to load image %s\n", fileName0);
        return -1;
    }
    Mat img1 = imread(fileName1, 1);
    if(img1.empty()) {
        printf("failed to load image %s\n", fileName1);
        return -1;
    }

    vector<KeyLine> lines0;
    Mat linesDescriptor0;

    vector<KeyLine> lines1;
    Mat linesDescriptor1;


    Ptr<LSDDetector>        lsd = LSDDetector::createLSDDetector();
    Ptr<BinaryDescriptor>   lbd = BinaryDescriptor::createBinaryDescriptor();
//    Ptr<ORB>                orb = ORB::create(nFeatures,scaleFactor,nLevels);
    Ptr<BinaryDescriptorMatcher>    bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    vector<vector<DMatch> >  lmatches;
    vector<DMatch > lmatchesDraw;
//    std::vector<char> mask( lmatches.size(), 1 );
    /* create binary masks */
     cv::Mat mask0 = Mat::ones( img0.size(), CV_8UC1 );
     cv::Mat mask1 = Mat::ones( img1.size(), CV_8UC1 );

    lsd->detect(img0, lines0, 1, 1/*, mask0*/);
    lbd->compute(img0,lines0,linesDescriptor0);
    printf("lines0 descriptor size %d\n", linesDescriptor0.rows);

    lsd->detect(img1, lines1, /*mask1*/1, 1);
    lbd->compute(img1,lines1,linesDescriptor1);
    printf("lines1 descriptor size %d\n", linesDescriptor1.rows);
//    orb->detectAndCompute(imgRight, cv::Mat(),pointsFourth, pdescFourth,false);

    bdm->knnMatch(linesDescriptor0,linesDescriptor1,lmatches, 2);
//    cout << "linesDescripotor: " << linesDescriptor.rows << " x " <<linesDescriptor.cols  <<endl << linesDescriptor.row(0) << endl;
//    cvtColor(img0, outimg, CV_GRAY2BGR);
//    drawKeylines(outimg, lines, outimg, Scalar(255, 0, 0));
//    printf("lmatches size %lu\n", lmatches.size());
    //lmatchesDraw.resize(lmatches.size());
    for(int i = 0; i < lmatches.size(); i++){
        lmatchesDraw.push_back(lmatches[i][0]);
        //lmatchesDraw[i] = lmatches[i][0];
        printf("match[%d]: qid: %d, tid %d, dist %f\n", i, lmatchesDraw[i].queryIdx, lmatchesDraw[i].trainIdx, lmatchesDraw[i].distance);
    }
    printf("lmatchesDraw size %lu\n", lmatchesDraw.size());

    Mat matchImg;

    Mat outimg0, outimg1;
    drawKeylines(img0, lines0, outimg0, Scalar(255, 0, 0));
    drawKeylines(img1, lines1, outimg1, Scalar(255, 0, 0));
    imshow("lines0", outimg0);
    imshow("lines1", outimg1);
    waitKey();
    /*
    cvtColor(img0, img0, CV_GRAY2BGR);
    cvtColor(img1, img1, CV_GRAY2BGR);*/
    std::vector<char> mask( lmatchesDraw.size(), 1 );
    drawLineMatches( img0, lines0, img1, lines1, lmatchesDraw, matchImg, Scalar::all( -1 ), Scalar::all( -1 ), mask,
                     DrawLinesMatchesFlags::DEFAULT );
    printf("drawLineMatches done\n");
    imshow("matchImg", matchImg);
    waitKey();

    return 0;

}
