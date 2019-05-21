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

bool useTrackbar = false;

// Change threshold_slider with the trackbar - this is the callback function if the trackbar is being moved
void on_trackbar(int, void*)
{
	if (threshold_slider == 0) 
		adaptiveThreshold(frameGray, frameThresholded, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 33, 5);
	else
		threshold(frameGray, frameThresholded, threshold_slider, 255, ADAPTIVE_THRESH_MEAN_C);
}

// Return value in grayscale of subpixel point p in frame - uses pointer of Mat (so address of Mat[0])
int getSubpixelValue(const Mat &frame, const Point2f p)
{
	// position of the top left pixel of subpixel square
	int x = int(floorf(p.x));
	int y = int(floorf(p.y));

	// out of bounds
	if (x < 0 || x >= frame.cols - 1 ||
		y < 0 || y >= frame.rows - 1)
		return 127;

	// delta in both directions into the subpixel square
	float dx = p.x - x;
	float dy = p.y - y;

	// pointer to grayscale color (one byte per pixel) of top left pixel
	unsigned char* i = (unsigned char*)((frame.data + y * frame.step) + x);
	float middleTop = (1 - dx) * i[0] + dx * i[1];
	// go one line lower (y+1)
	i += frame.step; 
	float middleBot = (1 - dx) * i[0] + dx * i[1];
	float middle = (1 - dy) * middleTop + dy * middleBot;

	return (int)middle;
}

int main(int argc, const char * argv[]) {
	Mat frame;

	VideoCapture cap = VideoCapture("C:\\Users\\Max\\Dropbox\\Uni\\8. Semester\\AR\\MarkerMovie.mp4"); 
	//VideoCapture cap = VideoCapture(0);
	
	if (useTrackbar == true)
	{
		namedWindow("ThresholdTrackbar", cv::WINDOW_AUTOSIZE);
		createTrackbar("ThresholdTrackbar", "ThresholdTrackbar", &threshold_slider, sliderMaxValue, on_trackbar);
	}


	while (cap.read(frame)) {

		// mirror effect
		flip(frame, frame, 1);

		// to grayscale
		cvtColor(frame, frameGray, CV_BGR2GRAY);

		if (useTrackbar == true)
			on_trackbar(threshold_slider, 0);
		else
			threshold(frameGray, frameThresholded, THRESHOLD, 255, ADAPTIVE_THRESH_MEAN_C);


		vector<vector<Point> > contours;
		findContours(frameThresholded, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

		vector<Point> approx;

		// Go over all contours
		for (int i = 0; i < contours.size(); i++) {

			// Find polygons for every contour, that has a maximum 2% difference in arc length to the original contour
			approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true) * 0.02, true);
			
			// Is approximated polygon a square?
			if (approx.size() == 4) {

				// Discard small and large contour areas
				double contourSize = contourArea(approx, false);

				if (contourSize > 600 && contourSize < 12000) {

					// Discard obvious non-squares
					std:vector<double> edges;

					edges.push_back(cv::norm(approx[0] - approx[1]));
					edges.push_back(cv::norm(approx[1] - approx[2]));
					edges.push_back(cv::norm(approx[2] - approx[3]));
					edges.push_back(cv::norm(approx[3] - approx[0]));

					double maxEdge = *max_element(edges.begin(), edges.end());
					double minEdge = *min_element(edges.begin(), edges.end());

					if (maxEdge < (2.5 * minEdge))
					{
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

								Point2f subPoint = Point2f(x, y);

								// draw dividing point
								circle(frame, subPoint, 1, Scalar(255, 0, 0), 2, 8);

								int mine = getSubpixelValue(frameGray, subPoint);
							}

							// draw endpoint
							circle(frame, Point(approx[j].x, approx[j].y), 1, Scalar(0, 255, 0), 2, 8);
						}
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
