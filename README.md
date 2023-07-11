# Rectify Map Video Undistortion

This code uses the OpenCV library to create a rectify map and undistort a live video stream by detecting a checkerboard pattern. The program provides controls for capturing images, calibrating the camera, saving the calibration data, loading previous calibration, and toggling checkerboard detection.

## Prerequisites

Before using this code, ensure that you have the following:

- OpenCV library installed.
- A camera connected to your system.
- A printed checkerboard pattern of the appropriate size.

## Instructions

To use this code:

1. Print out a checkerboard pattern of the appropriate size. Make sure it matches the dimensions and square size specified in the code.

2. Modify the `chessboardDimensions` constant in the code to match the number of **inside corners** of the checkerboard pattern. Do **not** count the number of squares.

3. Adjust the `calibrationSquareDimension` constant in the code to the proper size of the squares on your printed checkerboard. The unit can be in meters or any other desired unit.

4. Run the code and adjust the camera to ensure the checkerboard pattern is visible in the video stream.

5. Press the 'c' key to enable checkerboard detection. Enabling checkerboard detection will cause the program to find the checkerboard in the image once every ten frames. It also captures the image when the checkerboard is detected.

6. (Outdated) Press the spacebar to capture an image when the checkerboard pattern is detected. The saved images will be used for camera calibration. (This was used before the checkerboard detection also captured the image. It has been kept in incase someone wants to go back to manually capturing images)

7. Press Enter (Return) to start the camera calibration process. Ensure you have captured enough images (at least 15) before starting the calibration. The calibration process will estimate the camera matrix and distortion coefficients.
   
8. (Optional) Press the 's' key to save the camera calibration data to a file named "Calibration.txt". Note that when the camera is calibrated by pressing Enter (step 8), the calibration data is automatically saved. However, this behavior may change in future versions.

9. After calibration, the camera matrix and distortion coefficients will be displayed in the console output.

10. Press 'u' to undistort the live video stream using the calibration data. The undistorted video will be displayed in a separate window.

11. Press 'l' to load previously saved camera calibration data from a file named "calibration.txt". This can be useful if you want to use previously calibrated data without performing the calibration process again.

12. Press 'z' to display only the left camera (if available).

13. Press 'x' to display only the right camera (if available).

14. Press Esc or 'q' to exit the program.

Note: Make sure to adjust the code according to your specific camera setup, including the camera index, video source, and any necessary modifications for different camera configurations.

**Important:** Make sure to adjust the code according to your specific camera setup, including the camera index, video source, and any necessary modifications for different camera configurations.

## Code Explanation

The code consists of the following key functions:

1. `createKnownBoardPosition`: Creates known 3D coordinates of the corners on the checkerboard.
2. `getChessboardCorners`: Detects chessboard corners in the captured images and stores them for calibration.
3. `cameraCalibration`: Performs camera calibration using the captured chessboard corner coordinates.
4. `saveCameraCalibration`: Saves the camera calibration data (camera matrix and distortion coefficients) to a file.
5. `loadCameraCalibration`: Loads camera calibration data from a file.
6. `undistortVideo`: Undistorts the live video stream using the camera calibration data.

The main loop of the code captures frames from the video stream and processes user inputs. It provides controls for capturing images, starting calibration, toggling checkerboard detection, undistorting the video, and saving/loading camera calibration data.

Make sure to read the code comments for more detailed explanations of each section.

Note: The code includes options for working with left and right cameras. Modify the code accordingly if you have a different camera configuration.

Please adjust the code as needed to suit your specific requirements and camera setup.
