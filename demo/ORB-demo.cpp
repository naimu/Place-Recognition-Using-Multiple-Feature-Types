/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>

// DBoW2
#include "DBoW2.h" // defines ORBVocabulary and ORBDatabase

#include "DUtils.h"
#include "DUtilsCV.h" // defines macros CVXX
#include "DVision.h"

// OpenCV
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/features2d.hpp>
#if CV24
//#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d.hpp>

#endif


using namespace DBoW2;
using namespace DUtils;
using namespace std;
using namespace cv;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void loadFeatures(vector<vector<Mat>> &features, string dir);
void changeStructure(const vector<float> &plain, vector<vector<float> > &out,
  int L);
void testVocCreation(const vector<vector<vector<float> > > &features);
void testDatabase(const vector<vector<vector<float> > > &features);


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
const int NIMAGES = 4541;

// extended surf gives 128-dimensional vectors
const bool EXTENDED_SURF = false;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv)
{
  string dir = argv[1];
  vector<vector<Mat>> features;
  loadFeatures(features, dir);


  testVocCreation(features);

  wait();

  testDatabase(features);

  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<Mat > > &features, string dir)
{
  features.clear();
  features.reserve(NIMAGES);

  Ptr<ORB> orb = ORB::create(500, 1.2, 1);


  char fileName[256];
  cout << "Extracting ORB features..." << endl;
  for(int i = 0; i < NIMAGES; ++i)
  {
//    stringstream ss;
//    ss << "images/image" << i << ".png";
    sprintf(fileName, "%06d.png", i);
    string imgName = dir + "image_0/" + fileName;

    cv::Mat image = cv::imread(imgName, 0);
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    Mat descriptors;

    orb->detectAndCompute(image, cv::Mat(), keypoints, descriptors, false);
    features.push_back(vector<vector<float> >());
    changeStructure(descriptors, features.back(), surf.descriptorSize());
  }
}

// ----------------------------------------------------------------------------

void changeStructure(const vector<float> &plain, vector<vector<float> > &out,
  int L)
{
  out.resize(plain.size() / L);

  unsigned int j = 0;
  for(unsigned int i = 0; i < plain.size(); i += L, ++j)
  {
    out[j].resize(L);
    std::copy(plain.begin() + i, plain.begin() + i + L, out[j].begin());
  }
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<vector<vector<float> > > &features)
{
  // branching factor and depth levels 
 // const int k = 9;
 // const int L = 3;
  const int k = 10;
  const int L = 6;
  const WeightingType weight = TF_IDF;
  const ScoringType score = L1_NORM;

  Surf64Vocabulary voc(k, L, weight, score);

  cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl
  << voc << endl << endl;

  // lets do something with this vocabulary
  cout << "Matching images against themselves (0 low, 1 high): " << endl;
  BowVector v1, v2;
  for(int i = 0; i < NIMAGES; i++)
  {
    voc.transform(features[i], v1);
    for(int j = 0; j < NIMAGES; j++)
    {
      voc.transform(features[j], v2);
      
      double score = voc.score(v1, v2);
      cout << "Image " << i << " vs Image " << j << ": " << score << endl;
    }
  }

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  // voc.save("small_voc.yml.gz");
  voc.save("KITTI00_surf64_k10_L6_voc.yml.gz");
  cout << "Done" << endl;
}

// ----------------------------------------------------------------------------

void testDatabase(const vector<vector<vector<float> > > &features)
{
  cout << "Creating a small database..." << endl;

  // load the vocabulary from disk
  Surf64Vocabulary voc("small_voc.yml.gz");
  
  Surf64Database db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that 
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(int i = 0; i < NIMAGES; i++)
  {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;
  for(int i = 0; i < NIMAGES; i++)
  {
    db.query(features[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the 
    // database. ret[1] is the second best match.

    cout << "Searching for Image " << i << ". " << ret << endl;
  }

  cout << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database..." << endl;
  db.save("small_db.yml.gz");
  cout << "... done!" << endl;
  
  // once saved, we can load it again  
  cout << "Retrieving database once again..." << endl;
  Surf64Database db2("small_db.yml.gz");
  cout << "... done! This is: " << endl << db2 << endl;
}

// ----------------------------------------------------------------------------


