#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

const int sliderMaxValue = 255;
Mat frameGray;
Mat frameThresholded;
int threshold_slider;

void on_trackbar(int, void*)
{
	threshold(frameGray, frameThresholded, threshold_slider, 255, ADAPTIVE_THRESH_MEAN_C);
}


int main(int argc, const char * argv[]) {
	Mat frame;

	//VideoCapture cap = VideoCapture("MarkerMovie.MP4");
	VideoCapture cap = VideoCapture(0);

	while (cap.read(frame)) {

		// mirror effect
		flip(frame, frame, 1);

		cvtColor(frame, frameGray, CV_BGR2GRAY);

		namedWindow("MyTrackbar", cv::WINDOW_AUTOSIZE);
		createTrackbar("MyTrackbar", "MyTrackbar", &threshold_slider, sliderMaxValue, on_trackbar);

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
			approxPolyDP(Mat(contours[k]), approx, arcLength(Mat(contours[k]), true) * 0.05, true);
			
			// Is approximated polygon a square?
			if (approx.size() == 4) {
				Rect boundingRect = cv::boundingRect(approx);
				// Discard small rectangles
				if (boundingRect.area() > 300) {
					cv::polylines(frame, approx, true, Scalar(0, 0, 255), 4);

					for (int j = 0; j < approx.size(); j++)
					{
						for (int a = 1; a < 7; a++)
						{
							double alpha = a / 6.0;

							double x = alpha * approx[j].x + (1.0 - alpha) * approx[((j + 1) % 4)].x;
							double y = alpha * approx[j].y + (1.0 - alpha) * approx[((j + 1) % 4)].y;

							circle(frame, Point(x, y), 1, Scalar(100, 100, 100), 1, 8);
						}
					}
				}
			}
							
		}

		imshow("Display window", frame);

		if (waitKey(30) >= 0) break;
	}

	return 0;
}
