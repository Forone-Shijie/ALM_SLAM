//
// Created by sjzhang on 24-8-16.
//

#include "cvdrawingutils.h"
#include "aruco.h"
#include "timers.h"
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <stdexcept>

using namespace cv;
using namespace ::aruco;
using namespace std;

string TheMarkerMapConfigFile;
float TheMarkerSize = -1;
VideoCapture TheVideoCapturer;
Mat TheInputImage, TheInputImageCopy;
CameraParameters TheCameraParameters;
MarkerMap TheMarkerMapConfig;
MarkerDetector TheMarkerDetector;
MarkerMapPoseTracker TheMSPoseTracker;
int waitTime = 0;
int ref_id = 0;

char camPositionStr[100];
char camDirectionStr[100];

class CmdLineParser
{
    int argc;char** argv;public:
    CmdLineParser(int _argc, char** _argv): argc(_argc), argv(_argv){}
    //is the param?
    bool operator[](string param)
    {int idx = -1;   for (int i = 0; i < argc && idx == -1; i++)if (string(argv[i]) == param) idx = i;return (idx != -1); }
    //return the value of a param using a default value if it is not present
    string operator()(string param, string defvalue = "-1"){int idx = -1;for (int i = 0; i < argc && idx == -1; i++)if (string(argv[i]) == param) idx = i;if (idx == -1) return defvalue;else return (argv[idx + 1]);}
};

/************************************
 ************************************/
int main(int argc, char** argv)
{
    try
    {
        CmdLineParser cml(argc, argv);
        if (argc < 4 || cml["-h"])
        {
            cerr << "Invalid number of arguments" << endl;
            cerr << "Usage: (in.avi|live[:camera_index(e.g 0 or 1)])) marksetconfig.yml camera_intrinsics.yml  "
                    "[-s marker_size] [-ref_id reference_marker_id] [-e use EnclosedMarker or not]\n\t" << endl;
            return false;
        }
        TheMarkerMapConfigFile = argv[2];
        TheMarkerMapConfig.readFromFile(TheMarkerMapConfigFile);
        TheMarkerSize = stof(cml("-s", "1"));
        ref_id = std::stoi(cml("-ref_id"));
        if(-1 == ref_id){cout<<"You need indicate a referene_marker by use the parameter [-ref_id].\n"<<endl;return false;}

        // read from camera or from  file
        string TheInputVideo=string(argv[1]);
        if ( TheInputVideo.find( "live")!=std::string::npos)
        {
            int vIdx = 0;
            // check if the :idx is here
            char cad[100];
            if (TheInputVideo.find(":") != string::npos)
            {
                std::replace(TheInputVideo.begin(), TheInputVideo.end(), ':', ' ');
                sscanf(TheInputVideo.c_str(), "%s %d", cad, &vIdx);
            }
            cout << "Opening camera index " << vIdx << endl;
            TheVideoCapturer.open(vIdx);
            waitTime = 10;
        }
        else TheVideoCapturer.open(argv[1]);        // check video is open
        if (!TheVideoCapturer.isOpened())throw std::runtime_error("Could not open video");

        // read first image to get the dimensions
        TheVideoCapturer >> TheInputImage;

        // read camera parameters if passed
        TheCameraParameters.readFromXMLFile(argv[3]);
        TheCameraParameters.resize(TheInputImage.size());

		// prepare the detector
        TheMarkerDetector.setDictionary( TheMarkerMapConfig.getDictionary());
        TheMarkerDetector.getParameters().detectEnclosedMarkers(std::stoi(cml("-e", "0"))); //If use enclosed marker, set -e 1(true), or set -e 0(false). Default value is 0.

        if (cml["-config"])TheMarkerDetector.loadParamsFromFile(cml("-config"));
        // prepare the pose tracker if possible
        // if the camera parameers are avaiable, and the markerset can be expressed in meters, then go

        if (TheMarkerMapConfig.isExpressedInPixels() && TheMarkerSize > 0)
            TheMarkerMapConfig = TheMarkerMapConfig.convertToMeters(TheMarkerSize);

        cout << "TheCameraParameters.isValid()=" << TheCameraParameters.isValid() << " "<< TheMarkerMapConfig.isExpressedInMeters() << endl;

        if (TheCameraParameters.isValid() && TheMarkerMapConfig.isExpressedInMeters()){
            TheMSPoseTracker.setParams(TheCameraParameters, TheMarkerMapConfig);
            TheMarkerSize=cv::norm(TheMarkerMapConfig[0][0]- TheMarkerMapConfig[0][1]);
        }

		char key = 0;
        int index = 0;
        // capture until press ESC or until the end of the video
        cout << "Press 's' to start/stop video" << endl;
        Timer avrgTimer;
        do
        {
            TheVideoCapturer.retrieve(TheInputImage);
            TheInputImage.copyTo(TheInputImageCopy);
            index++;  // number of images captured
            if (index>1) avrgTimer.start();//do not consider first frame that requires initialization
            // Detection of the markers
            vector<Marker> detected_markers = TheMarkerDetector.detect(TheInputImage,TheCameraParameters,TheMarkerSize);
            // estimate 3d camera pose if possible
            if (TheMSPoseTracker.isValid())
                if (TheMSPoseTracker.estimatePose(detected_markers)) {
				Mat rMatrix,camPosMatrix,camVecMatrix;//定义旋转矩阵，平移矩阵，相机位置矩阵和姿态矩阵
				Rodrigues(TheMSPoseTracker.getRvec(),rMatrix);//获得相机坐标系与世界坐标系之间的旋转向量并转化为旋转矩阵
				camPosMatrix=rMatrix.inv()*(-TheMSPoseTracker.getTvec().t());//计算出相机在世界坐标系下的三维坐标
				Mat vect=(Mat_<float>(3,1)<<0,0,1);//定义相机坐标系下的单位方向向量，将其转化到世界坐标系下便可求出相机在世界坐标系中的姿态
				camVecMatrix=rMatrix.inv()*vect;//计算出相机在世界坐标系下的姿态

				    sprintf(camPositionStr,"Camera Position: px=%f, py=%f, pz=%f", camPosMatrix.at<float>(0,0),
                        camPosMatrix.at<float>(1,0), camPosMatrix.at<float>(2,0));
                    sprintf(camDirectionStr,"Camera Direction: dx=%f, dy=%f, dz=%f", camVecMatrix.at<float>(0,0),
                        camVecMatrix.at<float>(1,0), camVecMatrix.at<float>(2,0));
				}

            if (index>1) avrgTimer.end();
            cout<<"Average computing time "<<avrgTimer.getAverage()<<" ms"<<endl;

            // print the markers detected that belongs to the markerset
            for (auto idx : TheMarkerMapConfig.getIndices(detected_markers))
			{
                detected_markers[idx].draw(TheInputImageCopy, Scalar(0, 0, 255));
				if(ref_id == detected_markers[idx].id) CvDrawingUtils::draw3dAxis(TheInputImageCopy,detected_markers[idx],TheCameraParameters);
		    }
		    putText(TheInputImageCopy,camPositionStr,Point(10,13),cv::FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,255,255));
		    putText(TheInputImageCopy,camDirectionStr,Point(10,30),cv::FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,255,255));
imshow("out",TheInputImageCopy);

key = waitKey(waitTime);
if (key=='s') waitTime = waitTime?0:10;
        } while (key != 27 && TheVideoCapturer.grab());
    }
    catch (std::exception& ex) { cout << "Exception :" << ex.what() << endl;  }
}
