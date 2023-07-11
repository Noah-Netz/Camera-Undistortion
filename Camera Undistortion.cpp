#include <iostream>
#include <sstream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

//const float calibrationSquareDimension = 0.02482f; //meters
//const float calibrationSquareDimension = 0.03158f; //meters
const float calibrationSquareDimension = 0.02350f; //meters
const Size chessboardDimensions = Size(6, 8); //Count the INSIDE corners of the outer squares, not the number of squares
bool goodImage = false;

void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& corners)
{
    for (int i = 0; i < boardSize.height; i++)
    {

        for (int j = 0; j < boardSize.width; j++)
        {
            corners.push_back(Point3f(j * squareEdgeLength, i * squareEdgeLength, 0.0f));
        }

    }
}


void getChessboardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults = true)
{
    bool enableCheckerboardDetection = true;
    //int n = 0;
    for (vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++)
    {
        //if (n % 10 != 0) { continue; } // Skip frames that are not a multiple of 5
        vector<Point2f> pointBuf;
        bool found = findChessboardCorners(*iter, Size(chessboardDimensions), pointBuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

        if (found)
        {
            allFoundCorners.push_back(pointBuf);
        }

        if (showResults)
        {
            drawChessboardCorners(*iter, Size(chessboardDimensions), pointBuf, found);
            imshow("Looking for corners", *iter);
            waitKey(0);
        }
    }
}

Mat distanceCoefficients;

void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCore)
{
    //Mat distanceCoefficients;
    vector<vector<Point2f>> checkerboardImageSpacePoints;
    getChessboardCorners(calibrationImages, checkerboardImageSpacePoints, false);

    vector<vector<Point3f>> worldSpaceCornerPoints(1);

    createKnownBoardPosition(boardSize, squareEdgeLength, worldSpaceCornerPoints[0]);
    worldSpaceCornerPoints.resize(checkerboardImageSpacePoints.size(), worldSpaceCornerPoints[0]);

    vector<Mat> rVectors, tVectors;
    distanceCoefficients = Mat::zeros(8, 1, CV_64F);

    //fisheye::calibrate(worldSpaceCornerPoints, checkerboardImageSpacePoints, boardSize, cameraMatrix, distanceCoefficients, rVectors, tVectors);
    calibrateCamera(worldSpaceCornerPoints, checkerboardImageSpacePoints, boardSize, cameraMatrix, distanceCoefficients, rVectors, tVectors);

    cout << "Camera Matrix: " << cameraMatrix << endl;
    cout << "Distortion Coefficients: " << distanceCoefficients << endl;
}

bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients)
{
    ofstream outStream(name);
    if (outStream)
    {
        uint16_t rows = cameraMatrix.rows;
        uint16_t columns = cameraMatrix.cols;

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++)
            {
                double value = cameraMatrix.at<double>(r, c);
                outStream << value << endl;
            }
        }

        rows = distanceCoefficients.rows;
        columns = distanceCoefficients.cols;

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++)
            {
                double value = distanceCoefficients.at<double>(r, c);
                outStream << value << endl;
            }
        }

        outStream.close();
        return true;

    }

    return false;
}

bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distanceCoefficients)
{
    ifstream inStream(name);
    if (inStream)
    {
        // Read camera matrix
        cameraMatrix = Mat::zeros(3, 3, CV_64F);
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 3; c++)
            {
                double value;
                inStream >> value;
                cameraMatrix.at<double>(r, c) = value;
            }
        }

        // Read distance coefficients
        distanceCoefficients = Mat::zeros(1, 5, CV_64F);
        for (int i = 0; i < 5; i++)
        {
            double value;
            inStream >> value;
            distanceCoefficients.at<double>(0, i) = value;
        }

        inStream.close();
        return true;
    }


    return false;
}


void undistortVideo(VideoCapture& cap, Mat& cameraMatrix, Mat& distanceCoefficients, VideoWriter& writer)
{
    // Check that the camera matrix and distortion coefficients are valid
    if (cameraMatrix.empty() || distanceCoefficients.empty()) {
        cerr << "Error: Camera matrix or distortion coefficients are empty" << endl;
        return;
    }

    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);

    // Get the optimal new camera matrix for undistortion
    Mat optimalCameraMatrix = getOptimalNewCameraMatrix(cameraMatrix, distanceCoefficients, Size(width, height), 0);
    //Mat optimalCameraMatrix;
    //fisheye::estimateNewCameraMatrixForUndistortRectify(cameraMatrix, distanceCoefficients, Size(width, height), Matx33d::eye(), optimalCameraMatrix, 1);

    // Create an undistortion map from the camera calibration data
    Mat map1, map2;
    initUndistortRectifyMap(cameraMatrix, distanceCoefficients, Mat(), optimalCameraMatrix, Size(width, height), CV_32FC1, map1, map2);
    //fisheye::initUndistortRectifyMap(cameraMatrix, distanceCoefficients, Matx33d::eye(), optimalCameraMatrix, Size(width, height), CV_32FC1, map1, map2);

    // Create a new window to display the undistorted frames
    namedWindow("Undistorted", WINDOW_AUTOSIZE);

    // Process each frame in the input video
    Mat frame, undistorted;
    while (true) {
        if (!cap.read(frame)) {
            // If no more frames, exit the loop
            break;
        }

        //Rect crop_rec(0, 0, width / 2, height);
        //frame = frame(crop_rec);

        // Undistort the frame using the map
        remap(frame, undistorted, map1, map2, INTER_CUBIC);

        // Display the undistorted frame
        imshow("Undistorted", undistorted);

        // Exit the loop if the user presses the "q" key
        if (waitKey(1) == 'q') {
            break;
        }
    }

    // Destroy the window
    destroyWindow("Undistorted");

}

int main()
{
    bool leftCam = false;
    bool rightCam = false;
    Mat frame;
    Mat drawToFrame;

    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);

    //hMat distanceCoefficients;

    vector<Mat> savedImages;

    vector<vector<Point2f>> markedCorners, rejectedCandidates;

    //VideoCapture cap("rtsp://192.168.1.101:8554/payload");

    VideoCapture cap(2);

    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);


    bool enableCheckerboardDetection = false;

    bool cameraCalibrated = false;



    if (!cap.isOpened())
    {
        cout << "Unable to load video";
        return 0;
    }

    int framesPerSecond = 30;

    namedWindow("Stream", WINDOW_AUTOSIZE);



    while (true)
    {

        //counter++;
        if (!cap.read(frame))
            break;

        
        if (leftCam)
        {
            Rect crop_rec(0, 0, width / 2, height);
            frame = frame(crop_rec);
        }
        else if (rightCam)
        {
            Rect crop_rec(width / 2, 0, width / 2, height);
            frame = frame(crop_rec);
        }
        else
        {
            Rect crop_rec(0, 0, width, height);
            frame = frame(crop_rec);
        }
        
        //Rect crop_rec(0, 0, width / 2, height);
        //frame = frame(crop_rec);

        //resize(frame, frame, Size(width, height / 1.3));


        if (enableCheckerboardDetection)
        {
            goodImage = false;
            static int counter = 0;
            vector<Vec2f> foundPoints;
            bool found = false;

            frame.copyTo(drawToFrame);

            if (frame.empty()) {
                cerr << "Error: frame is empty" << endl;
                return 0;
            }

            if (frame.type() != CV_8UC1 && frame.type() != CV_8UC3) {
                cerr << "Error: frame type is not supported" << endl;
                return 0;
            }

            //cout << "Chessboard dimensions: " << chessboardDimensions << endl;

            if (counter % 10 == 0) // Find the chessboard corners every 3 frames
            {
                goodImage = true;
                found = findChessboardCorners(frame, chessboardDimensions, foundPoints);
                if (found)
                {
                    drawChessboardCorners(drawToFrame, chessboardDimensions, foundPoints, found);
                    Mat temp;
                    frame.copyTo(temp);
                    savedImages.push_back(temp);
                    cout << "Image saved! Total saved images: " << savedImages.size() << endl;
                }
                else
                    cout << "Chessboard corners not found" << endl;
            }

            imshow("Stream", drawToFrame);
            counter++;
        }
        else
        {
            imshow("Stream", frame);
        }

        char character = waitKey(1000 / framesPerSecond);

        switch (character)
        {
        case ' ':
            //Saving image
            if (enableCheckerboardDetection && goodImage)
            {
                Mat temp;
                frame.copyTo(temp);
                savedImages.push_back(temp);
                cout << "Image saved! Total saved images: " << savedImages.size() << endl;
            }
            break;
        case 13:
            //Start calibration
            if (savedImages.size() > 15)
            {
                cameraCalibration(savedImages, chessboardDimensions, calibrationSquareDimension, cameraMatrix, distanceCoefficients);
                saveCameraCalibration("CameraCalibration", cameraMatrix, distanceCoefficients);
                cameraCalibrated = true;

            }

            break;
        case 27:
            //Exit program
            return 0;
            break;
        case 'c':
            //Toggle checkerboard detection
            cout << "Detecting Checkerboard";
            enableCheckerboardDetection = !enableCheckerboardDetection;
            break;
        case 'u':
            //undistort
            if (cameraCalibrated == true) {
                VideoWriter writer("undistorted_video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), cap.get(CAP_PROP_FPS), cv::Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));

                undistortVideo(cap, cameraMatrix, distanceCoefficients, writer);

                writer.release();
            }
        case 's':
            //Save camera calibration
            if (cameraCalibrated == true) {
                saveCameraCalibration("Calibration.txt", cameraMatrix, distanceCoefficients);
            }
        case 'l':
            //Load camera calibration
            loadCameraCalibration("calibration.txt", cameraMatrix, distanceCoefficients);
            cameraCalibrated = true;
            break;
        case 'z':
            //display only left cam
            leftCam = !leftCam;
            break;
        case 'x':
            //display only right cam
            rightCam = !rightCam;
            break;
        }

    }

    return 0;
}