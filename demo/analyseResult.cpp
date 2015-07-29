#include <opencv2/opencv.hpp>
// DBoW2

#include "DBoW2.h" // defines Surf64Vocabulary and Surf64Database
#include <ostream>
#include <fstream>
//#include "DUtils.h"
//#include "DUtilsCV.h" // defines macros CVXX
//#include "DVision.h"

using namespace cv;
using namespace std;


// ----------------------------------------------------------------------------
struct QueryFrame {
    int frameID;
    string frameName;
    vector<double> framePose;
    QueryFrame(int id, string name, const vector<double> &pose) {
        frameID = id;
        frameName = name;
        framePose = pose;
    };
    void print() {
        cout<< "queryFrame " << frameID << " :" << std::endl;
        cout<< "queryFrameName: " << frameName << endl;
        cout << "queryFramePose: ";
        for(size_t i = 0; i < framePose.size(); ++i) {
            cout << framePose[i] << " ";
        }
        cout << endl;
        cout << "End queryFrame " << frameID<<endl;
    };
};

struct QueryResult {
    int frameID;
    double resultScore;
    vector<double> resultPose;
    QueryResult(int id, double score, const vector<double> &pose) {
        frameID = id;
        resultScore = score;
        resultPose = pose;
    }
    void print() {
        cout<< "resultFrame " << frameID << " :" << std::endl;
        cout<< "resultFrameScore: " << resultScore << endl;
        cout << "resultFramePose: ";
        for(size_t i = 0; i < resultPose.size(); ++i) {
            cout << resultPose[i] << " ";
        }
        cout << endl;
        cout << "End resultFrame " << frameID<<endl;
    };
};

bool loadResult(string resultFileName, string nodeName = "result") {
    cv::FileStorage resultFS(resultFileName.c_str(), cv::FileStorage::READ);
    if(!resultFS.isOpened()) throw string("Could not open file ") + resultFileName;
    cv::FileNode fquery = resultFS[nodeName];
    printf("query size %lu\n", fquery.size());
    for(size_t i = 0; i < fquery.size() ;++i) {
        cv::FileNode fn = fquery[i]["queryImage"];
        int qId = fn["frameID"];
        printf("qid %d\n", qId);
        string qName = fn["frameName"];
        cv::FileNode ff = fn["framePose"];
        vector<double> pPose;
        cv::read(ff, pPose);
        QueryFrame qframe(qId, qName, pPose);
        qframe.print();
        cv::FileNode fresults = fquery[i]["queryResults"];
        for(size_t j = 0; j < fresults.size(); ++j) {
            int rId = fresults[j]["frameID"];
            double rScore = fresults[j]["score"];
            cv::FileNode fff = fresults[j]["pose"];
            vector<double> rPose;
            cv::read(fff, rPose);
            QueryResult qresult(rId, rScore, rPose);
            qresult.print();
        }
    }
}
int main(int argc, char** argv) {

    if(argc < 2) {
        return -1;
    }
    string resultFileName = argv[1];
    loadResult(resultFileName);
    return 0;

}
