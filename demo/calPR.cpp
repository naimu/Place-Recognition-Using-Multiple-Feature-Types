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

double distThresh = std::numeric_limits<double>::max();
//double scoreThresh = 0.02;

//#define SHOW_CONFUSI4NMTIX
bool calFP = true;

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

    string posresultFileName = argv[1];
    vector<QueryFrame> posqueryFrameVec;       // all results in this vector
    loadResult(posresultFileName, posqueryFrameVec);
    printf("queryFrameVec size: %lu\n", posqueryFrameVec.size());

    string negresultFileName = argv[2];
    vector<QueryFrame> negqueryFrameVec;       // all results in this vector
    loadResult(negresultFileName, negqueryFrameVec);
    printf("queryFrameVec size: %lu\n", negqueryFrameVec.size());

    cout << "======================Analyzing Result========================" << endl << endl;

    FILE *filefp = fopen("./precision-recall.txt", "w");
    for(float scoreThresh = 0; scoreThresh <=1.0; scoreThresh += 0.001){
        int tp = 0, fp = 0, fn = 0, tn = 0;
        int validPosResultNum = 0;
        for(int i = 0; i < posqueryFrameVec.size(); i+=1){
            if(posqueryFrameVec[i].queryResults.empty()){
                printf("No results for queryImage\n");
                continue;
            }
            validPosResultNum++;
            bool breaked = false;
            for(int j = 0; j < posqueryFrameVec[i].queryResults.size(); j++){
                if(posqueryFrameVec[i].queryResults[0].resultScore < scoreThresh){
                    fn++;
                    breaked = true;
                    break;
                }

                if(posqueryFrameVec[i].queryResults[j].resultScore < scoreThresh){
                    breaked = true;
                    break;
                }
                vector<double> grundTruth = posqueryFrameVec[i].framePose;
                vector<double> resultPose = posqueryFrameVec[i].queryResults[j].resultPose;
                //printf("grundtruth extraction done \n");
                if(!grundTruth.empty() && !resultPose.empty()){
                    double distxz = dist(grundTruth, resultPose);
                    //cout << "distxz: " << distxz << endl;
                    if(distxz < distThresh){
                        tp++;
                        breaked = true;
                        break;
                    }
                    else
                    {
                        //cout << "query frame index: " << i << ", " << "result frame index: " << posqueryFrameVec[i].queryResults[0].frameID << endl;
                        //cout << "distxz: " << distxz << endl;
                    }
                }

            }
            if(!breaked) {
                fp++;
            }
        }
        assert(tp + fn == validPosResultNum);

        int validNegResultNum = 0;
        for(int i = 0; i < negqueryFrameVec.size(); i+=1){
            //printf("processing frame: %d\n", i);
            if(negqueryFrameVec[i].queryResults.empty()){
                printf("No results for queryImage\n");
                continue;
            }
            validNegResultNum++;
            bool breaked = false;
            for(int j = 0; j < negqueryFrameVec[i].queryResults.size(); j++){
                if(negqueryFrameVec[i].queryResults[j].resultScore > scoreThresh){
                    breaked = true;
                    break;
                }

            }
            if(breaked) {
                fp++;
            }
            else
                tn++;

        }
        assert(tn + fp == validNegResultNum);
        if(tp + fp == 0 || tp + fn == 0)
            continue;
        double precision = (double)tp / (double)(tp + fp);
        double recall = (double)tp / (double)(tp + fn);
        fprintf(filefp, "%lf %lf %lf\n", precision, recall, scoreThresh);
    //    cout << tp << "," << fp << endl;
        //cout << "Precision: " << endl << (double) tp / (double)(tp+fp) << endl;
        //cout << "tp: "  << tp  << ", fp: " << fp << ", fn:" << fn << endl;
        //out << "Recall: " << endl << (double) tp / (double)(tp+fn) << endl;
    }
    fclose(filefp);

    if(waitKey(0) == 27){
        cout << "esc key is pressed by user" << endl;
        return 1;
    }

    return 0;

}
