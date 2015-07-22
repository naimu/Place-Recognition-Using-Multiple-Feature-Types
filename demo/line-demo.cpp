#include <opencv2/opencv.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
// DBoW2

#include "DBoW2.h" // defines Surf64Vocabulary and Surf64Database

//#include "DUtils.h"
//#include "DUtilsCV.h" // defines macros CVXX
//#include "DVision.h"

using namespace cv;
using namespace cv::line_descriptor;
using namespace std;
using namespace DBoW2;


// ----------------------------------------------------------------------------

void printHelp(){
    printf("usage: ./line-demo [--mode=CREATE|QUERY] [<images path>] [-k <branch number>] [-l <level number>] [-d <database file name>] [-s <start index>] [-e <end index>] [-v [<vocabulary file name>]]\n");
}

void loadFeatures(vector<vector<Mat> > &features, string dir, int startNumber, int imgNumber)
{
  //features.clear();
  assert(imgNumber > 0);
  //features.reserve(imgNumber);

  char fileName[256];
  cout << "Extracting Line features from: "<< dir << endl;
  for(int i = startNumber; i < imgNumber; ++i)
  {
//    stringstream ss;
//    ss << "images/image" << i << ".png";
    sprintf(fileName, "%06d.png", i);
    string imgName = dir + fileName;

    //double startTime = (double)getTickCount();
    cv::Mat image = cv::imread(imgName, 0);
    if(image.empty()) {
        printf("cannot load image %s\n", imgName.c_str());
        continue;
    }
    //double loadEndTime = (double)getTickCount();
    cv::Mat mask;
    vector<KeyLine> lines;
    Mat linesDescriptors;
    //Ptr<LSDDetector>        lsd = LSDDetector::createLSDDetector();
    Ptr<BinaryDescriptor>   lbd = BinaryDescriptor::createBinaryDescriptor();
    //double createDetectorEndTime = (double)getTickCount();
    //vector<cv::KeyPoint> keypoints;
    //vector<float> descriptors;

    //lsd->detect(image, lines, 1, 1/*, mask0*/);
    lbd->detect(image, lines/*, mask0*/);
    //double detectEndTime = (double)getTickCount();
    lbd->compute(image,lines,linesDescriptors);
    //double computeDescEndTime = (double)getTickCount();
    //surf(image, mask, keypoints, descriptors);
    vector<Mat> imgDescVec;
    imgDescVec.resize(linesDescriptors.rows);
    //printf("desc rows %d, cols %d\n", linesDescriptors.rows, linesDescriptors.cols);
    for(size_t j = 0; j < imgDescVec.size(); ++j) {
        //imgDescVec2d[j].resize(FLBD::L);
        imgDescVec[j] = linesDescriptors.row(j);
        //if(i == 0 && j == 0)
            //cout << linesDescriptors.row(j) << endl;
        //assert(desRow.isContinuous() && desRow.cols == FLBD::L);
        //desRow.copyTo(imgDescVec2d[j]);
    }
    //double changeStructureEndTime = (double)getTickCount();
    features.push_back(imgDescVec);
    if(i % 100 == 0)
        printf("loaded features for image %s\n", imgName.c_str());
    //printf("========== frame %d ===========\n", i);
    //printf("load image time: %lf\n", (loadEndTime - startTime) / (double)getTickFrequency());
    //printf("create detector time: %lf\n", (createDetectorEndTime - loadEndTime) / (double)getTickFrequency());
    //printf("detect features time: %lf\n", (detectEndTime - createDetectorEndTime) / (double)getTickFrequency());
    //printf("compute features descriptors time: %lf\n", (computeDescEndTime - detectEndTime) / (double)getTickFrequency());
    //printf("change structure time: %lf\n", (changeStructureEndTime - computeDescEndTime) / (double)getTickFrequency());
    //printf("===============================\n\n");

    //features.push_back(vector<vector<unsigned char> >());
    //changeStructure(descriptors, features.back(), surf.descriptorSize());
  }
}

void createVoc(const vector<vector<Mat> > &features, int imageNumber, int vocBranchNumber, int vocLevelNumber)
{
    // branching factor and depth levels 
    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;

    //Surf64Vocabulary voc(k, L, weight, score);
    LBDVocabulary voc(vocBranchNumber, vocLevelNumber, weight, score);

    cout << "Creating a small " << vocBranchNumber << "^" << vocLevelNumber << " vocabulary..." << endl;
    voc.create(features);
    cout << "... done!" << endl;

    cout << "Vocabulary information: " << endl
        << voc << endl << endl;

    // lets do something with this vocabulary
    /*cout << "Matching images against themselves (0 low, 1 high): " << endl;
    BowVector v1, v2;
    for(int i = 0; i < imageNumber; i++)
    {
        voc.transform(features[i], v1);
        for(int j = 0; j < imageNumber; j++)
        {
            voc.transform(features[j], v2);

            double score = voc.score(v1, v2);
            cout << "Image " << i << " vs Image " << j << ": " << score << endl;
        }
    }*/

    // save the vocabulary to disk
    cout << endl << "Saving vocabulary..." << endl;
    // voc.save("small_voc.yml.gz");
    char vocFileName[256];
    sprintf(vocFileName, "KITTI_line_branch%d_level_%d_voc.yml.gz", vocBranchNumber, vocLevelNumber);
    voc.save(vocFileName);
    cout << "Done" << endl;
}

void createDatabase(const vector<vector<Mat> > &features, string vocFileName, string dbFileName)
{
  cout << "Creating database..." << endl;

  // load the vocabulary from disk

  LBDVocabulary voc(vocFileName);
  
  LBDDatabase db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that 
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(size_t i = 0; i < features.size(); i++)
  {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;


  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database..." << endl;
  db.save(dbFileName);
  cout << "... done!" << endl;
  
  // once saved, we can load it again  
  /*cout << "Retrieving database once again..." << endl;
  LBDDatabase db2("KITTI00_line_K9_L3_db.yml.gz");
  cout << "... done! This is: " << endl << db2 << endl;*/
}

void createDatabase(const vector<vector<Mat> > &features, int imageNumber, int vocBranchNumber, int vocLevelNumber)
{
  cout << "Creating database..." << endl;

  // load the vocabulary from disk
  char vocFileName[256];
  sprintf(vocFileName, "KITTI_line_branch%d_level_%d_voc.yml.gz", vocBranchNumber, vocLevelNumber);

  LBDVocabulary voc(vocFileName);
  
  LBDDatabase db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that 
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(int i = 0; i < imageNumber; i++)
  {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;


  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database..." << endl;
  char dbFileName[256];
  sprintf(dbFileName, "KITTI_line_branch%d_level_%d_db.yml.gz", vocBranchNumber, vocLevelNumber);
  db.save(dbFileName);
  cout << "... done!" << endl;
  
  // once saved, we can load it again  
  /*cout << "Retrieving database once again..." << endl;
  LBDDatabase db2("KITTI00_line_K9_L3_db.yml.gz");
  cout << "... done! This is: " << endl << db2 << endl;*/
}

void addImagesToDatabase(const vector<vector<Mat> > &features, LBDDatabase &db, string dbFileName)
{
  cout << "Adding images to database..." << endl;

  for(size_t i = 0; i < features.size(); i++)
  {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;


  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database..." << endl;
  db.save(dbFileName);
  cout << "... done!" << endl;
  
  // once saved, we can load it again  
  /*cout << "Retrieving database once again..." << endl;
  LBDDatabase db2("KITTI00_line_K9_L3_db.yml.gz");
  cout << "... done! This is: " << endl << db2 << endl;*/
}

QueryResults queryImageInDataBase(const LBDDatabase &db, const Mat &image) {
    QueryResults results;
    if(image.empty()) {
        printf("query image is empty!\n");
        return results;
    }
    vector<KeyLine> lines;
    Mat linesDescriptors;
    //Ptr<LSDDetector>        lsd = LSDDetector::createLSDDetector();
    Ptr<BinaryDescriptor>   lbd = BinaryDescriptor::createBinaryDescriptor();

    //lsd->detect(image, lines, 1, 1/*, mask0*/);
    lbd->detect(image, lines/*, mask0*/);
    lbd->compute(image,lines,linesDescriptors);
    vector<Mat> imgDescVec;
    imgDescVec.resize(linesDescriptors.rows);
    for(size_t j = 0; j < imgDescVec.size(); ++j) {
        imgDescVec[j] = linesDescriptors.row(j);
    }
    db.query(imgDescVec, results, 4);
    cout << "query result is: " << results << endl;
    return results;
}

void queryImages(const vector<vector<Mat> > &features, const LBDDatabase &db) {
    double truePositive = 0;
    double falsePositive = 0;
    double falseNegative = 0;
    for(size_t i = 0; i < features.size(); ++i) {
        QueryResults results;
        db.query(features[i], results, 4);
        bool correct = false;
        for(int j = 0; j < results.size(); ++j) {
            if(abs((int)results[j].Id - (int)i) <= 5) {
                truePositive++;
                correct = true;
                break;
            }
        }
        if(!correct){
            printf("false detection: query image %lu\n", i);
            cout << results << endl;
            sleep(1);
            falseNegative++;
        }
        //printf("query image %d done with %d\n", i, correct);
    }
    printf("recall rate: %lf\n", truePositive / (truePositive + falseNegative));
}

int main(int argc, char** argv) {

    if(argc < 6) {
        printHelp();
        return -1;
    }
    char* mode = 0;
    enum ModeOption {UNKNOWN = -1, CREATEALL = 0, CREATEVOC, CREATEDB, QUERY, ADDDATA};
    ModeOption modeOpt = UNKNOWN;
    const char *mode_option = "--mode=";
    string dbFileName, vocFileName;
    int vocBranchNumber = -1;
    int vocLevelNumber = -1;
    int startIndex = 0;
    int endIndex = 0;
    string dirPath;
    for(int i = 1; i < argc; ++i) {
        if(strncmp(mode_option, argv[i], strlen(mode_option)) == 0) {
            mode = argv[i] + strlen(mode_option);
            printf("mode is %s\n", mode);
            if(strncmp(mode, "CREATEALL", strlen(mode)) == 0) {
                modeOpt = CREATEALL;
                printf("set mode to CREATEALL\n");
            }
            else if(strncmp(mode, "CREATEVOC", strlen(mode)) == 0) {
                modeOpt = CREATEVOC;
                printf("set mode to CREATEVOC\n");
            }
            else if(strncmp(mode, "CREATEDB", strlen(mode)) == 0) {
                modeOpt = CREATEDB;
                printf("set mode to CREATEDB\n");
            }
            else if(strncmp(mode, "QUERY", strlen(mode)) == 0) {
                modeOpt = QUERY;
                printf("set mode to QUERY\n");
            }
            else if(strncmp(mode, "ADDDATA", strlen(mode)) == 0) {
                modeOpt = ADDDATA;
                printf("set mode to ADDDATA\n");
            }
            else
                modeOpt = UNKNOWN;
        }
        else if(strncmp(argv[i], "-k", 2) == 0) {
            vocBranchNumber = atoi(argv[++i]);
            printf("voc branch number %d\n", vocBranchNumber);
        }
        else if(strncmp(argv[i], "-l", 2) == 0) {
            vocLevelNumber = atoi(argv[++i]);
            printf("voc level number %d\n", vocLevelNumber);
        }
        else if(strncmp(argv[i], "-d", 2) == 0) {
            dbFileName = argv[++i];
        } 
        else if(strncmp(argv[i], "-v", 2) == 0) {
            vocFileName = argv[++i];
        }
        else if(strncmp(argv[i], "-s", 2) == 0) {
            startIndex = atoi(argv[++i]);
            printf("start index %d\n", startIndex);
        }
        else if(strncmp(argv[i], "-e", 2) == 0) {
            endIndex = atoi(argv[++i]);
            printf("end index %d\n", endIndex);
        }
        else if(argv[i][0] != '-') {
            dirPath = argv[i];
        }
    }
    if(dirPath.empty()) {
        printf("please indicate images path\n");
        printHelp();
        return -1;
    }
    if(startIndex >= endIndex) {
        printf("start index not small than end index: start %d, end %d\n", startIndex, endIndex);
        printHelp();
        return -1;
    }
    if((modeOpt == CREATEALL || modeOpt == CREATEVOC) && vocBranchNumber < 0) {
        printf("please indicate branch number using -k\n");
        printHelp();
        return -1;
    }
    if((modeOpt == CREATEALL || modeOpt == CREATEVOC) && vocLevelNumber < 0) {
        printf("please indicate level number using -l\n");
        printHelp();
        return -1;
    }
    if((modeOpt == CREATEDB) && vocFileName.empty()) {
        printf("please indicate vocabulary file name using -v\n");
        printHelp();
        return -1;
    }
    if(modeOpt == QUERY || modeOpt == ADDDATA) {
        if (dbFileName.find("yml") == std::string::npos) {
            printf("incorrect database file: %s\n", dbFileName.c_str());
            printHelp();
            return -1;
        }
    }
    //string dir = argv[1];
    switch(modeOpt) {
        case CREATEALL:
            {
                printf("creating vocabulary and database\n");
                //string dir0 = "/media/ys/Storage/Dataset/KITTI/10/image_0/";
                //int imageNumber0 = 1200;
                //string dir1 = "/media/ys/Storage/Dataset/KITTI/10/image_0/";
                //int imageNumber1 = 10;

                vector<vector<Mat> > features;
                long startTime = getTickCount();
                loadFeatures(features, dirPath, startIndex, endIndex);
                //printf("feature size %lu\n", features.size());
                //loadFeatures(features, dir1, imageNumber1);
                int totalImageNumber = features.size();
                printf("image number: %lu\n", features.size());
                long endFeatureTime = getTickCount();
                printf("elapsed time [load features]: %lf sec\n",double(endFeatureTime-startTime) / (double)getTickFrequency());
                createVoc(features, totalImageNumber, vocBranchNumber, vocLevelNumber);
                long endCreateVocTime = getTickCount();
                printf("elapsed time [create Vocabulary]: %lf sec\n",double(endCreateVocTime-endFeatureTime) / (double)getTickFrequency());
                createDatabase(features, totalImageNumber, vocBranchNumber, vocLevelNumber);
                long endCreateDatabaseTime = getTickCount();
                printf("elapsed time [create Database]: %lf sec\n",double(endCreateDatabaseTime-endCreateVocTime) / (double)getTickFrequency());
                printf("total elapsed time: %lf sec\n", double(getTickCount()-startTime) / (double)getTickFrequency());
                break;
            }
        case CREATEVOC:
            {
                printf("creating vocabulary\n");
                //string dir0 = "/media/ys/Storage/Dataset/KITTI/10/image_0/";
                //int imageNumber0 = 1200;
                //string dir1 = "/media/ys/Storage/Dataset/KITTI/10/image_0/";
                //int imageNumber1 = 10;

                vector<vector<Mat> > features;
                long startTime = getTickCount();
                loadFeatures(features, dirPath, startIndex, endIndex);
                //printf("feature size %lu\n", features.size());
                //loadFeatures(features, dir1, imageNumber1);
                int totalImageNumber = features.size();
                printf("image number: %lu\n", features.size());
                long endFeatureTime = getTickCount();
                printf("elapsed time [load features]: %lf sec\n",double(endFeatureTime-startTime) / (double)getTickFrequency());
                createVoc(features, totalImageNumber, vocBranchNumber, vocLevelNumber);
                long endCreateVocTime = getTickCount();
                printf("elapsed time [create Vocabulary]: %lf sec\n",double(endCreateVocTime-endFeatureTime) / (double)getTickFrequency());
                break;
            }
        case CREATEDB:
            {
                printf("creating Database\n");
                //string dir0 = "/media/ys/Storage/Dataset/KITTI/10/image_0/";
                //int imageNumber0 = 1200;
                //string dir1 = "/media/ys/Storage/Dataset/KITTI/10/image_0/";
                //int imageNumber1 = 10;

                vector<vector<Mat> > features;
                long startTime = getTickCount();
                loadFeatures(features, dirPath, startIndex, endIndex);
                //printf("feature size %lu\n", features.size());
                //loadFeatures(features, dir1, imageNumber1);
                int totalImageNumber = features.size();
                printf("image number: %lu\n", features.size());
                long endFeatureTime = getTickCount();
                printf("elapsed time [load features]: %lf sec\n",double(endFeatureTime-startTime) / (double)getTickFrequency());
                if(dbFileName.empty())
                    dbFileName = "database.yml";
                createDatabase(features, vocFileName, dbFileName);
                long endCreateVocTime = getTickCount();
                printf("elapsed time [create Vocabulary]: %lf sec\n",double(endCreateVocTime-endFeatureTime) / (double)getTickFrequency());
                break;
            }
        case QUERY:
            {
                printf("query image\n");
                cout << "Retrieving database: " << dbFileName <<endl;
                LBDDatabase db2(dbFileName);
                cout << "... done! This is: " << endl << db2 << endl;
                //int imageNumber0 = 1200;
                //string dir0 = "/media/ys/Storage/Dataset/KITTI/10/image_1/";
                vector<vector<Mat> > features;
                loadFeatures(features, dirPath, startIndex, endIndex);
                queryImages(features, db2);
                //Mat qImg = imread(queryImageName, 0);
                //if(qImg.empty()) {
                //    printf("cannot load image %s\n", queryImageName.c_str());
                //}
                //queryImageInDataBase(db2, qImg);
                break;
            }
        case ADDDATA:
            {
                cout << "Retrieving database: " << dbFileName <<endl;
                LBDDatabase db2(dbFileName);
                cout << "... done! This is: " << endl << db2 << endl;
                //int imageNumber0 = 800;
                //string dir0 = "/media/ys/Storage/Dataset/KITTI/06/image_0/";
                vector<vector<Mat> > features;
                loadFeatures(features, dirPath, startIndex, endIndex);
                addImagesToDatabase(features, db2, dbFileName);
                //Mat qImg = imread(queryImageName, 0);
                //if(qImg.empty()) {
                //    printf("cannot load image %s\n", queryImageName.c_str());
                //}
                //queryImageInDataBase(db2, qImg);
                break;
            }
        case UNKNOWN:
            printf("unknown mode option\n");
            break;
    }
    return 0;

}
