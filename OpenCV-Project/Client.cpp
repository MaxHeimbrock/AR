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

// Struct holding all infos about each strip, e.g. length
struct MyStrip {
	// discrete length - stripes width is const 3 in for loop in processing
	int stripeLength;
	// Indices like -2 to +2
	int nStop;
	int nStart;
	// vector of directions to orient the stripe
	Point2f stripeVecX;
	Point2f stripeVecY;
};

// returns an empty Mat for the stripes pixels in the correct size
// sets the stripe struct
Mat calculate_Stripe(double dx, double dy, MyStrip& st) {
	// Norm (euclidean distance) from the direction vector is the length (derived from the Pythagoras Theorem)
	double diffLength = sqrt(dx * dx + dy * dy);

	// Length proportional to the marker size
	st.stripeLength = (int)(0.8 * diffLength);

	if (st.stripeLength < 5)
		st.stripeLength = 5;

	// Make stripeLength odd (because of the shift in nStop), Example 6: both sides of the strip must have the same length XXXOXXX
	//st.stripeLength |= 1;
	if (st.stripeLength % 2 == 0)
		st.stripeLength++;

	// E.g. stripeLength = 5 --> from -2 to 2: Shift -> half top, the other half bottom
	//st.nStop = st.stripeLength >> 1;
	st.nStop = st.stripeLength / 2;
	st.nStart = -st.nStop;

	Size stripeSize;

	// Sample a strip of width 3 pixels
	stripeSize.width = 3;
	stripeSize.height = st.stripeLength;

	// Normalized direction vector
	st.stripeVecX.x = dx / diffLength;
	st.stripeVecX.y = dy / diffLength;

	// Normalized perpendicular vector
	st.stripeVecY.x = st.stripeVecX.y;
	st.stripeVecY.y = -st.stripeVecX.x;

	// 8 bit unsigned char with 1 channel, gray
	return Mat(stripeSize, CV_8UC1);
}

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
		// For every Square
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
						//cv::polylines(frame, approx, true, Scalar(0, 0, 255), 4);

						vector<vector<double>> fittedLines(4);
				
						// Divide polygon lines into 7 parts with 6 points and endpoints
						// For every line of square
						for (int j = 0; j < approx.size(); j++)
						{
							// Euclidic distance, 7 -> parts, both directions dx and dy
							double dx = ((double)approx[(j + 1) % 4].x - (double)approx[j].x) / 7.0;
							double dy = ((double)approx[(j + 1) % 4].y - (double)approx[j].y) / 7.0;

							// makes a placeholder matrix in the correct size for the strip of pixels
							MyStrip strip;
							Mat stripPixels = calculate_Stripe(dx, dy, strip);

							// Vector to keep track of intermediate points per square
							vector<Point2f> intermediatePoints(6);

							// For every intermediate point
							for (int a = 1; a < 7; a++)
							{
								// Position calculation
								double px = (double)approx[j].x + (double)a * dx;
								double py = (double)approx[j].y + (double)a * dy;

								Point2f p = Point2f(px, py);

								// draw dividing point
								//circle(frame, p, 1, Scalar(255, 0, 0), 2, 8);

								// test my subpixel function
								// int mine = getSubpixelValue(frameGray, subPoint);

								// find subpixel intensity of all pixels in strip and safe in Mat stripPixels

								// in x axis, always 3 pixel wide
								for (int m = -1; m <= +1; m++)
								{
									// in y axis
									for (int n = strip.nStart; n <= strip.nStop; n++)
									{
										Point2f subPixel;

										// navigate to every pixel in strip
										subPixel.x = px + ((double)m * strip.stripeVecX.x) + ((double)n*strip.stripeVecY.x);
										subPixel.y = py + ((double)m * strip.stripeVecX.y) + ((double)n*strip.stripeVecY.y);

										// Combined Intensity of the subpixel
										int pixelIntensity = getSubpixelValue(frameGray, subPixel);

										// Converte from index to pixel coordinate
										// m (Column, real) -> -1,0,1 but we need to map to 0,1,2 -> add 1 to 0..2
										int w = m + 1;

										// n (Row, real) -> add stripeLenght >> 1 to shift to 0..stripeLength
										// n=0 -> -length/2, n=length/2 -> 0 ........ + length/2
										int h = n + (strip.stripeLength >> 1);

										// Set pointer to correct position and safe subpixel intensity
										stripPixels.at<uchar>(h, w) = (uchar)pixelIntensity;
									}
								}

								// here the strip is filled

								// Discard outer values, as sobel filter can not be applied
								vector<double> peakValues(strip.stripeLength-2);								

								// Find maximum sobel value in for loop
								int maxIndex = -1;
								double maxValue = 0;

								// If length = 5, k is 1, 2, 3 - discard 0 and 4
								for (int k = 1; k < strip.stripeLength - 1; k++)
								{
									// This is the sobel filter applied with absolute value
									double sobelValue = abs(-(stripPixels.at<uchar>(k - 1, 0) + 2 * stripPixels.at<uchar>(k - 1, 1) + stripPixels.at<uchar>(k - 1, 2)) +
										(stripPixels.at<uchar>(k + 1, 0) + 2 * stripPixels.at<uchar>(k + 1, 1) + stripPixels.at<uchar>(k + 1, 2)));

									// Is value new maximum?
									if (sobelValue >= maxValue)
									{
										maxValue = sobelValue;
										maxIndex = k - 1;
									}
									peakValues[k - 1] = sobelValue;									
								}

								double y0, y1, y2;

								y0 = (maxIndex - 1 < 0) ? 0 : peakValues[maxIndex - 1];
								y1 = peakValues[maxIndex];
								y2 = (maxIndex + 1 >= peakValues.size()) ? 0 : peakValues[maxIndex + 1];

								double pos = (y2 - y0) / (4 * y1 - 2 * y0 - 2 * y2);

								if (isnan(pos)) {
									continue;
								}

								// Exact point with subpixel accuracy
								Point2d edgeCenter;

								// Back to Index positioning, Where is the edge (max gradient) in the picture?
								int maxIndexShift = maxIndex - (strip.stripeLength >> 1);

								// Shift the original edgepoint accordingly -> Is the pixel point at the top or bottom?
								edgeCenter.x = (double)p.x + (((double)maxIndexShift + pos) * strip.stripeVecY.x);
								edgeCenter.y = (double)p.y + (((double)maxIndexShift + pos) * strip.stripeVecY.y);

								// Fill the vector for each square intermediate points
								intermediatePoints[a-1] = edgeCenter;

								// Draw / Highlight the subpixel with blue color
								circle(frame, edgeCenter, 2, CV_RGB(0, 0, 255), -1);
							}

							// Fit the line and add it to the other fitted lines of the square
							vector<double> fittedLine;
							fitLine(intermediatePoints, fittedLine, CV_DIST_L2, 0, 0.01, 0.01);
							fittedLines[j] = fittedLine;

							// draw endpoint
							//circle(frame, Point(approx[j].x, approx[j].y), 1, Scalar(0, 255, 0), 2, 8);
						}	

						// Here I have all lines fitted in fittedLines
						//------------------------------

						Point2f corners[4];

						// Calculate the intersection points of both lines
						// For every line of square
						for (int j = 0; j < 4; ++j) 
						{
							// Index of second line 
							int jj = (j + 1) % 4;

							// We look for:
							// g0 = (x0,y0) + a*(u0,v0) == g1 = (x1,y1) + b*(u1,v1)

							double x0, x1, y0, y1, u0, u1, v0, v1;

							// From first line
							u0 = fittedLines[j][0];
							v0 = fittedLines[j][1];
							x0 = fittedLines[j][2];
							y0 = fittedLines[j][3];

							// From second line
							u1 = fittedLines[jj][0];
							v1 = fittedLines[jj][1];
							x1 = fittedLines[jj][2];
							y1 = fittedLines[jj][3];

							// (x|y) = p + s * vec
							// s = Ds / D (see Cramer's rule)
							// (x|y) = p + (Ds / D) * vec
							// (x|y) = (p * D / D) + (Ds * vec / D)
							// (x|y) = (p * D + Ds * vec) / D
							// (x|y) = a / c;

							// Cramer's rule
							// 2 unknown a,b -> Equation system
							double a = x1 * u0 * v1 - y1 * u0 * u1 - x0 * u1 * v0 + y0 * u0 * u1;
							double b = -x0 * v0 * v1 + y0 * u0 * v1 + x1 * v0 * v1 - y1 * v0 * u1;

							// Calculate the cross product to check if both direction vectors are parallel -> = 0
							// c -> Determinant = 0 -> linear dependent -> the direction vectors are parallel -> No division with 0
							double c = v1 * u0 - v0 * u1;
							if (fabs(c) < 0.001) {
								std::cout << "lines parallel" << std::endl;
								continue;
							}

							// We have checked for parallelism of the direction vectors
							// -> Cramer's rule, now divide through the main determinant
							a /= c;
							b /= c;

							// Exact corner
							corners[j].x = a;
							corners[j].y = b;

							// Added in sheet 4 Ex9 (b)- End *******************************************************************

							Point p;
							p.x = (int)corners[j].x;
							p.y = (int)corners[j].y;

							// Added in sheet 4 Ex9 (c)- Start *****************************************************************

							circle(frame, p, 5, CV_RGB(255, 255, 0), -1);
						}

						// We now have the corners in corners[]
						//-------------------------------------

						// Calculate homeography to target corners
						Point2f targetCorners[4];
						targetCorners[0].x = -0.5; targetCorners[0].y = -0.5;
						targetCorners[1].x = 5.5; targetCorners[1].y = -0.5;
						targetCorners[2].x = 5.5; targetCorners[2].y = 5.5;
						targetCorners[3].x = -0.5; targetCorners[3].y = 5.5;

						Mat H = getPerspectiveTransform(corners, targetCorners);

						// Create image for the marker
						Mat imageMarker(Size(6, 6), CV_8UC1);

						warpPerspective(frameGray, imageMarker, H, Size(6, 6));
						threshold(imageMarker, imageMarker, 90, 255, ADAPTIVE_THRESH_MEAN_C);
						
						/*
						Mat enlarged;

						cv::resize(imageMarker, enlarged, cv::Size(imageMarker.cols * 10, imageMarker.rows * 10), cv::INTER_NEAREST);

						cv::imshow("Mai Frame", enlarged);
						*/

						int code = 0;
						for (int j = 0; j < 6; ++j) {
							// Check if border is black
							int pixel1 = imageMarker.at<uchar>(0, j); //top
							int pixel2 = imageMarker.at<uchar>(5, j); //bottom
							int pixel3 = imageMarker.at<uchar>(j, 0); //left
							int pixel4 = imageMarker.at<uchar>(j, 5); //right

							// 0 -> black
							if ((pixel1 > 0) || (pixel2 > 0) || (pixel3 > 0) || (pixel4 > 0)) {
								code = -1;
								break;
							}
						}

						if (code < 0) {
							continue;
						}

						// Copy the BW values into cP -> codePixel on the marker 4x4 (inner part of the marker, no black border)
						int cP[4][4];
						for (int j = 0; j < 4; ++j) {
							for (int k = 0; k < 4; ++k) {
								// +1 -> no borders!
								cP[j][k] = imageMarker.at<uchar>(j + 1, k + 1);
								// If black then 1 else 0
								cP[j][k] = (cP[j][k] == 0) ? 1 : 0;
							}
						}

						// Save the ID of the marker, for each side
						int codes[4];
						codes[0] = codes[1] = codes[2] = codes[3] = 0;

						// Calculate the code from all sides at once
						for (int j = 0; j < 16; j++) {
							// /4 to go through the rows
							int row = j >> 2;
							int col = j % 4;

							// Multiplied by 2 to check for black values -> 0*2 = 0
							codes[0] <<= 1;
							codes[0] |= cP[row][col]; // 0°

							// 4x4 structure -> Each column represents one side 
							codes[1] <<= 1;
							codes[1] |= cP[3 - col][row]; // 90°

							codes[2] <<= 1;
							codes[2] |= cP[3 - row][3 - col]; // 180°

							codes[3] <<= 1;
							codes[3] |= cP[col][3 - row]; // 270°
						}

						// Account for symmetry -> One side complete white or black
						if ((codes[0] == 0) || (codes[0] == 0xffff)) {
							continue;
						}

						// Search for the smallest marker ID
						code = codes[0];
						for (int j = 1; j < 4; ++j) {
							if (codes[j] < code) {
								code = codes[j];
							}
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
