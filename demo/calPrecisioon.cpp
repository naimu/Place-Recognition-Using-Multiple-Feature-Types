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

//double distThresh = std::numeric_limits<double>::max();
//double distThresh = 5;
//double scoreThresh = 0.02;

//#define SHOW_CONFUSI4NMTIX
bool calFP = false;

// ----------------------------------------------------------------------------

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
struct QueryFrame {
    int frameID;
    string frameName;
    vector<double> framePose;
    vector<QueryResult> queryResults;
    QueryFrame(int id, string name, const vector<double> &pose) {
        frameID = id;
        frameName = name;
        framePose = pose;
    };
    void addResult(const QueryResult &result) {
        queryResults.push_back(result);
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

bool loadResult(string resultFileName, vector<QueryFrame> &qFVec, string nodeName = "result") {
    cv::FileStorage resultFS(resultFileName.c_str(), cv::FileStorage::READ);
    if(!resultFS.isOpened()) throw string("Could not open file ") + resultFileName;
    cv::FileNode fquery = resultFS[nodeName];
    printf("query size %lu\n", fquery.size());
    for(size_t i = 0; i < fquery.size() ;++i) {
        cv::FileNode fn = fquery[i]["queryImage"];
        int qId = fn["frameID"];
        if(i % 100 == 0)
            printf("qid %d\n", qId);
        string qName = fn["frameName"];
        cv::FileNode ff = fn["framePose"];
        vector<double> pPose;
        cv::read(ff, pPose);
        QueryFrame qframe(qId, qName, pPose);
//        qframe.print();

        cv::FileNode fresults = fquery[i]["queryResults"];
        if(fresults.size() == 0){
            printf("No results for queryImage %s\n", qName.c_str());

        }
        for(size_t j = 0; j < fresults.size(); ++j) {
            int rId = fresults[j]["frameID"];
            double rScore = fresults[j]["score"];
            cv::FileNode fff = fresults[j]["pose"];
            vector<double> rPose;
            cv::read(fff, rPose);
            QueryResult qresult(rId, rScore, rPose);
            qframe.addResult(qresult);
//            qresult.print();
        }
        qFVec.push_back(qframe);
    }
}

double dist(vector<double> &grundTruth, vector<double> &resultPose){
    double distxz = sqrt(pow(grundTruth[3] - resultPose[3], 2) + pow(grundTruth[11] - resultPose[11], 2));
    return distxz;
}
int main(int argc, char** argv) {

    if(argc < 3) {
        return -1;
    }
    string resultFileName = argv[1];
    double distThresh = atof(argv[2]);

    vector<QueryFrame> queryFrameVec;// all results in this vector
    loadResult(resultFileName, queryFrameVec);
    printf("queryFrameVec size: %lu\n", queryFrameVec.size());

    cout << "======================Analyzing Result========================" << endl << endl;

        int goodRetr = 0, badRetr = 0;
        double avgDist = 0.0;
        int validResultNum = 0;
        for(int i = 0; i < queryFrameVec.size(); i+=1){
            if(queryFrameVec[i].queryResults.empty()){
                printf("No results for queryImage\n");
                continue;
            }

            vector<double> grundTruth = queryFrameVec[i].framePose;
            vector<double> resultPose = queryFrameVec[i].queryResults[0].resultPose;
            printf("grundtruth extraction done \n");
            if(!grundTruth.empty() && !resultPose.empty()){
                validResultNum++;
                double distxz = dist(grundTruth, resultPose);
                if(distxz < distThresh){
                    goodRetr++;
                }
                else
                {
                    cout << "query frame index: " << i << ", " << "result frame index: " << queryFrameVec[i].queryResults[0].frameID << endl;
                    cout << "distxz: " << distxz << endl;
                    badRetr++;
                }

                avgDist+= distxz;
            }

        }

        avgDist/=validResultNum;

        cout << "number for good retrevial: " << goodRetr << "number of bad retrevial: " << endl << "average distance: " << avgDist << endl;


    if(waitKey(0) == 27){
        cout << "esc key is pressed by user" << endl;
        return 1;
    }

    return 0;

}
