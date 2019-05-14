#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

const int sliderMaxValue = 255;
Mat frameGray;
Mat frameThresholded;
int threshold_slider;

#define THRESHOLD 67

void on_trackbar(int, void*)
{
	threshold(frameGray, frameThresholded, threshold_slider, 255, ADAPTIVE_THRESH_MEAN_C);
}

int main(int argc, const char * argv[]) {
	Mat frame;

	VideoCapture cap = VideoCapture("C:\\Users\\Max\\Dropbox\\Uni\\8. Semester\\AR\\MarkerMovie.mp4"); 
	//VideoCapture cap = VideoCapture(0);

	while (cap.read(frame)) {

		// mirror effect
		flip(frame, frame, 1);

		cvtColor(frame, frameGray, CV_BGR2GRAY);

		namedWindow("ThresholdTrackbar", cv::WINDOW_AUTOSIZE);
		createTrackbar("ThresholdTrackbar", "ThresholdTrackbar", &threshold_slider, sliderMaxValue, on_trackbar);

		if (threshold_slider == 0) {
			adaptiveThreshold(frameGray, frameThresholded, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 33, 5);
		}
		else {
			on_trackbar(threshold_slider, 0);
		}

		vector<vector<Point> > contours;
		findContours(frameThresholded, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

		vector<Point> approx;

		// Go over all contours
		for (int k = 0; k < contours.size(); k++) {
			// Find polygons for every contour, that has a maximum 2% difference in arc length to the original contour
			approxPolyDP(Mat(contours[k]), approx, arcLength(Mat(contours[k]), true) * 0.02, true);
			
			// Is approximated polygon a square?
			if (approx.size() == 4) {
				Rect boundingRect = cv::boundingRect(approx);

				// Discard small and large contour areas
				double contourSize = contourArea(approx, false);
				if (contourSize > 700 && contourSize < 12000) {
					// Draw polygon around contour with 4 points
					cv::polylines(frame, approx, true, Scalar(0, 0, 255), 4);

					// Divide polygon lines into 7 parts with 6 points and endpoints
					for (int j = 0; j < approx.size(); j++)
					{
						for (int a = 1; a < 7; a++)
						{
							// divide at 1/7, 2/7, ...
							double alpha = a / 7.0;

							// position on line with: a * p1 + (1-a) * p2
							double x = alpha * approx[j].x + (1.0 - alpha) * approx[((j + 1) % 4)].x;
							double y = alpha * approx[j].y + (1.0 - alpha) * approx[((j + 1) % 4)].y;

							// draw dividing point
							circle(frame, Point(x, y), 1, Scalar(255, 0, 0), 2, 8);
						}

						// draw endpoint
						circle(frame, Point(approx[j].x, approx[j].y), 1, Scalar(0, 255, 0), 2, 8);
					}
				}
			}
							
		}

		cv::imshow("Main Frame", frame);
		//imshow("Thresholded Frame", frameThresholded);

		char key = (char)cv::waitKey(30);   // explicit cast
		if (key == 27) break;                // break if `esc' key was pressed. 
		//if (key == ' ') do_something();
	}

	return 0;
}
