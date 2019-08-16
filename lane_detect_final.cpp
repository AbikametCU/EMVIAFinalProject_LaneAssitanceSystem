/***********************************************************************************
*Copyright (C) Vignesh Iyer
*	       Abikametnathan Anbunathan
*
*This code shall not be used or distributed for purposes other than academic or
*creative.The authors or the University of Colorado Boulder is not responsible
*for misuse of this code.
*
************************************************************************************/

/*-------------------------------------------------------------
 * @File : lane_detect_test.cpp
 *
 * @Desription : It detects lanes, cars, as well as pedestrains
 *  	         using certain OpenCV functions as well as by
 *  	         usage of machine learning algorithms
 *
 * @Author : Vignesh Iyer, Abikametnathan Anbunathan
 *
 * @Date : 6th August, 2019
 * -------------------------------------------------------------*/

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/videoio.hpp>

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <pthread.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sched.h>
#include <syslog.h>
#include <semaphore.h>

#define NUM_THREADS 4

using namespace cv;
using namespace std;


Mat src,dst,gray,out,canny,hough;
Mat img; /* The main input frame */
Mat pedestrain_image1,pedestrain_image2;


Mat car_image;



int th1=50;
int th2=150;
vector<Vec4i> lines;
vector<Vec4i>left_lane;
vector<Vec4i>right_lane;

vector<Point>left_pts;
vector<Point>right_pts;

Vec4d right_line;
Vec4d left_line;
std::vector<std::vector<cv::Vec4i> > final_output(2);
bool Turn;
double img_center;
int frames=0;
char fileppm[50];
sem_t semS1;
pthread_t lane_threads[NUM_THREADS];
pthread_attr_t orig_sched_attr;


double before_transform,after_transform,transform_time;
double total_transform_time;



typedef struct
{
    int threadIdx;
    unsigned long long sequencePeriods;
} threadParams_t;

double getTimeMsec(void)
{
  struct timespec event_ts = {0, 0};
  clock_gettime(CLOCK_MONOTONIC, &event_ts);
  return ((event_ts.tv_sec)*1000.0) + ((event_ts.tv_nsec)/1000000.0);
}

void detectAndDisplay(Mat& frame, CascadeClassifier& cascade);


/*--------------------------------------
 * @Function: canny_edge
 * @Input : This function takes a Mat obj
 * 	    as input and calulcates edges
 * 	    using the Gaussian Blur and
 * 	    Canny APIs of OpenCV
 * @Return Value : This function returns
 *                 a Mat obj
 *---------------------------------------*/

Mat canny_edge(Mat frame)
{
	/*------------------------------------
	 * Using Gaussian Blur, Canny edge for
	 * edge detection and being able to
	 * find the lanes
	 * -----------------------------------*/
	cvtColor(frame,gray,COLOR_RGB2GRAY);
	GaussianBlur(gray,out,Size(5,5),0,0);
	Canny(out,canny,th1,th2);
	return canny;
}


/*--------------------------------------------
 * @Function: mask
 * @Input : This function takes a Mat obj
	    as input and an empty image is
	    created for creating a mask with
	    the help of polygon function and
	    then a bitwise AND on the orginal
	    frame
 * @Return Value : This function returns Mat
 * 		   obj.
 *-------------------------------------------*/

Mat mask(Mat frame)
{
	Mat mask_polygon,masking;
	int height=frame.rows;
	Point polygon_mask[3]={Point(0,height),Point(1200,height),Point(700,290)};
	Point pts[4] = {
      		cv::Point(210, 900),
      		cv::Point(400, 450),
      		cv::Point(717, 450),
      		cv::Point(1280, 900)
  		};
	mask_polygon=Mat::zeros(frame.size(),frame.type());

	fillConvexPoly(mask_polygon, pts,4, cv::Scalar(255, 0, 0));
	bitwise_and(frame,mask_polygon,masking);
	return masking;
}


/*---------------------------------------------------
 * @Function : line_detect
 * @Input : It takes Mat obj and a vec4i line as input
 *          which then caluclates the slope of left as
 *          well as right lines of the current lane
 *          based on a threshold and reference point
 *          as the center of the masked frame output
 * @Return Value : The function return double vec4i
 *      	   output to be used for further
 *      	   calucaltions
 * --------------------------------------------------*/

std::vector<std::vector<cv::Vec4i> > line_detect(Mat frame, vector<Vec4i>lines)
{
	Point low_coord, high_coord;
	vector<vector<Vec4i> > output(2);
	double slope_thresh=0.4;
	size_t j = 0;
	vector<double>slopes;
	vector<Vec4i>Use_lanes;
	for(auto i : lines)
	{
		low_coord=Point(i[0],i[1]);
		high_coord=Point(i[2],i[3]);

		double slope=((static_cast<double>(high_coord.y)-static_cast<double>(low_coord.y))/(static_cast<double>(high_coord.x)-static_cast<double>(low_coord.x)));
		if (abs(slope) > slope_thresh)
		{
      			slopes.push_back(slope);
      			Use_lanes.push_back(i);
    		}
  	}
	img_center = static_cast<double>((frame.cols / 2));
  	while (j < Use_lanes.size())
	{
    		low_coord = Point(Use_lanes[j][0], Use_lanes[j][1]);
    		high_coord = Point(Use_lanes[j][2], Use_lanes[j][3]);

    		/* Condition to classify line as left side or right side */
    		if (slopes[j] > 0 && high_coord.x > img_center && low_coord.x > img_center)
		{
      			right_lane.push_back(Use_lanes[j]);
			line( frame, low_coord, high_coord, Scalar(0,0,255), 1, LINE_AA);
			Turn=true;
		}
		else if (slopes[j] < 0 && high_coord.x < img_center && low_coord.x < img_center)
		{
        		left_lane.push_back(Use_lanes[j]);
   			line( frame, low_coord, high_coord, Scalar(255,0,0), 1, LINE_AA);
			Turn=false;
		}
    		j++;
  	}
	output[0]=right_lane;
	output[1]=left_lane;
	return output;

}


/*----------------------------------------------------
 * @Function : Left_Right(Mat , vector<vector<Vec4i>>)
 * @Input : It takes Mat as well s double vector as
 *  	    input and calulates if the lane is moving
 *  	    in left or right direction. The previous
 *  	    function return value is the input and it
 *  	    caluculates the turn based on intersection
 *  	    based on the intersection of left and right
 *  	    lines and edits in the original frame with
 *  	    left, right or center.
 * @Return Value : This function returns 0 on success
 * ---------------------------------------------------*/

int Left_Right(Mat frame,vector<vector<Vec4i> > output)
{
	std::string output_val;
        Point coord1,coord2;
	double vanish_x;
        double thr_vp = 10;
	double right_m,left_m;
	Point left_b,right_b;
	if (Turn == true)
	{
    		for (auto i : output[0])
		{
      			coord1 = cv::Point(i[0], i[1]);
      			coord2 = cv::Point(i[2], i[3]);

      			right_pts.push_back(coord1);
      			right_pts.push_back(coord2);
    		}

		if (right_pts.size() > 0)
		{
      			/* The right line is formed here */
      			cv::fitLine(right_pts, right_line, CV_DIST_L2, 0, 0.01, 0.01);
      			right_m = right_line[1] / right_line[0];
      			right_b = cv::Point(right_line[2], right_line[3]);
    		}
  	}

  	/* If left lines are being detected, fit a line using all the init and final points of the lines */
  	if (Turn == false)
	{
    		for (auto j : output[1])
		{
      			coord1 = cv::Point(j[0], j[1]);
      			coord2 = cv::Point(j[2], j[3]);

      			left_pts.push_back(coord1);
      			left_pts.push_back(coord2);
    		}

    		if (left_pts.size() > 0)
		{
      			/* The left line is formed here */

      			cv::fitLine(left_pts, left_line, CV_DIST_L2, 0, 0.01, 0.01);
      			left_m = left_line[1] / left_line[0];
      			left_b = cv::Point(left_line[2], left_line[3]);
    		}
  	}
	/* The vanishing point is the point where both lane boundary lines intersect */
	
        vanish_x = static_cast<double>(((right_m*right_b.x) - (left_m*left_b.x) - right_b.y + left_b.y) / (right_m - left_m));
        
	/* The vanishing points location determines where is the road turning */
        if (vanish_x < (img_center - thr_vp))
	{
		output_val = "Left Turn";
	}
	else if (vanish_x > (img_center + thr_vp))
	{
		output_val = "Right Turn";
	}
	else if (vanish_x >= (img_center - thr_vp) && vanish_x <= (img_center + thr_vp))
	{
		output_val = "Straight";
	}
	cv::putText(frame, output_val, cv::Point(50, 90), cv::FONT_HERSHEY_COMPLEX_SMALL, 3, cvScalar(0, 255, 0), 1, CV_AA);
        return 0;

}


/*------------------------------------------------------------
 * @Class : Detector
 * @Function : This uses HOG and SVM for pedestrain detection.
 * 	       Currently the default API provided by OpenCV is
 * 	       used for the same.
 * -----------------------------------------------------------*/

class Detector
{

    HOGDescriptor hog_people;
public:
    Detector() : hog_people()
    {
    	hog_people.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    }

    	/*-----------------------------------------
	 * @Function : detect (InputArray)
	 * @Input : InputArray is taken as input and
	 *  	    detectMultiScale is used to by
	 *  	    the trained algorithm to detect
	 *  	    people.
	 * @Return Value : Returns vector<Rect>
	 *-----------------------------------------*/
    	vector<Rect> detect(InputArray img)
    	{

		/* ---------------------------------------------------------------------
		 * Run the detector with default parameters. to get a higher hit-rate
		 * (and more false alarms, respectively), decrease the hitThreshold
		 * and groupThreshold (set groupThreshold to 0 to turn off the grouping
		 * completely).
		 * ---------------------------------------------------------------------*/

        	vector<Rect> found;
        	hog_people.detectMultiScale(img, found, 0.5, Size(8,8), Size(64,64), 1.1, 2, false);
        	return found;
    	}


	/*---------------------------------------------
	 * @Function : adjustRect(Rect & r) const
	 * @Input : This function creates a box around
	 * 	    the detected people with a green
	 * 	    colored box
	 * @Return Value : void
	 * --------------------------------------------*/
    	void adjustRect(Rect & r) const
    	{

		/*-------------------------------------------------------------
		 * The HOG detector returns slightly larger rectangles than the
		 * real objects, so we slightly shrink the rectangles to get a
		 * nicer output.
		 * ------------------------------------------------------------*/

		r.x += cvRound(r.width*0.1);
        	r.width = cvRound(r.width*0.8);
        	r.y += cvRound(r.height*0.07);
        	r.height = cvRound(r.height*0.8);

	}
};


/*-------------------------------------------------------
 * @Thread Init : void *lane_ops
 * @Input : (void *)
 * @Function : This thread intialization is for detection
 * 	       of the right and left lines of the present 
 * 	       lane in which the car is present. This 
 * 	       gives output only if the slope of the line
 * 	       is above a certain threshold.
 *-------------------------------------------------------*/

void *lane_ops(void *)
{
	Mat canny_out,mask_out;
	canny_out=canny_edge(img);
	mask_out=mask(canny_out);
	HoughLinesP(mask_out, lines, 2, CV_PI/180, 100, 100, 50);
	final_output=line_detect(img,lines);

	return 0;
}


/*-----------------------------------------------------
 * @Thread Init : void *Car_detect
 * @Input : (void *)
 * @Function : This is a thread used for car detection
 * 	       algorithm which runs for a ROI as given 
 * 	       which is a rough estimation of the road 
 * 	       as given in the video.  
 *----------------------------------------------------*/

void *Car_detect(void *)
{

	Rect ROI_car=Rect(450,350,300,300);
	car_image=img(ROI_car);
	const String cascade_name("cars_1.xml");
	CascadeClassifier cascade;

	if (!cascade.load(cascade_name))
	{
		perror("Error in loading cascade");
	}

	detectAndDisplay(img, cascade);
	return 0;
}


/*--------------------------------------------------
 * @Thread Init :void  *People_detect_firstROI
 * @Input : (void *)
 * @Function : This thread is for people/pedestrian
 * 	       detection which has a given ROI for
 * 	       the left half of the input frame and
 * 	       runs every 10 frames to speed up the 
 * 	       output frame rate.
 *--------------------------------------------------*/

void *People_detect_firstROI(void *)
{
	if(frames%10==0)
	{
		Rect ROI_1=Rect(0,0,img.cols/2,img.rows);
		pedestrain_image1=img(ROI_1);


		Detector detector_1;
		vector<Rect> found_1 = detector_1.detect(pedestrain_image1);

		for (vector<Rect>::iterator i = found_1.begin(); i != found_1.end(); ++i)
		{
			Rect &r = *i;
			detector_1.adjustRect(r);
			rectangle(img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
		}
	}
	return 0;
}


/*------------------------------------------------------
 * @Thread Init : void *People_detect_secondROI
 * @Input : (void *)
 * @Function : This thread initalization also works for
 * 	       every 10 frames but takes the whole ROI
 * 	       due to imperfect pedestrian detection.
 *------------------------------------------------------*/

void *People_detect_secondROI(void *)
{

	if(frames%10==0)
	{
		Rect ROI_2=Rect(0,0,img.cols,img.rows);
		pedestrain_image2=img(ROI_2);

		Detector detector_2;
		vector<Rect> found_2 = detector_2.detect(pedestrain_image2);

		for (vector<Rect>::iterator j = found_2.begin(); j != found_2.end(); ++j)
		{
			Rect &r = *j;
			detector_2.adjustRect(r);
			rectangle(img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
		}
  	}
	return 0;
}



int main(int argc, char** argv)
{

	/*---------------------------------------
	 * Intialization of the thread paramters
	 * as well as the starting for storing in 
	 * syslog.
	 *---------------------------------------*/
	
	threadParams_t threadParams[NUM_THREADS];
  	pthread_attr_t rt_sched_attr[NUM_THREADS];
  	int rt_max_prio, rt_min_prio;
  	struct sched_param rt_param[NUM_THREADS];
  	struct sched_param main_param;
  	pthread_attr_t main_attr;
  	pid_t mainpid;
	int rc;

	int log_options = LOG_PID;
	openlog("EMVIA PROJECT:",log_options,LOG_INFO);

	VideoCapture cap(argv[1]);

	while(1)
  	{
    		before_transform = getTimeMsec();
    		cap >>img;


    		if(img.empty())
    		{
      			cout << "End of video" << endl;
      			break;
    		}

		
		/*-------------------------------------------------------------------
		 * Creation of thread with keeping default paramaters for the threads
		 * by using NULL since all tasks are indpendent of each other and 
		 * does not require priority and given way to the core assignment 
		 * according to the available core as well as joining the threads.
		 *-------------------------------------------------------------------*/
    		
		rc= pthread_create(&lane_threads[0],NULL,People_detect_firstROI,NULL);
	  	if(rc<0)
	  	{
	    		printf( "problem while creating thread[0]\n");
	    		return -1;
		}
		
		rc= pthread_create(&lane_threads[1],NULL,lane_ops,NULL);
	  	if(rc<0)
	  	{
	     		printf( "problem while creating thread[0]\n");
	     		return -1;
		}
		
		rc= pthread_create(&lane_threads[2],NULL,Car_detect,NULL);
	 	if(rc<0)
	  	{
	     		printf( "problem while creating thread[1]\n");
	     		return -1;
		}
		
		rc= pthread_create(&lane_threads[3],NULL,People_detect_secondROI,NULL);
	  	if(rc<0)
	  	{
	     		printf( "problem while creating thread[0]\n");
	     		return -1;
		}


		for(int i=0;i<4;i++)
	  	{
		    pthread_join(lane_threads[i], NULL);
		}


		/*--------------------------------------------
	 	 * Converting the frames to images and saving
	 	 * them in a .ppm format to be later converted
	 	 * to a video stream
	 	 * -------------------------------------------*/

		after_transform = getTimeMsec();
		transform_time = after_transform - before_transform;
		syslog(LOG_INFO,"\nTime taken to  transform frame %d = %lf milliseconds",frames,transform_time);
		total_transform_time+=transform_time;
		imshow("frame",img);
		sprintf(fileppm,"images_1/image%04d.jpg",frames);
		imwrite(fileppm,img);
		frames++;

		int key;
		key = waitKey( 1 );

		if( (char)key == 27 )
		{
			break;
		}

	}
	destroyAllWindows();
	cap.release();


	/*frame processing stats*/

	syslog(LOG_INFO, "average time per frame = %lf milliseconds",(double)total_transform_time/frames);
	syslog(LOG_INFO, "total frames = %d",frames);
	printf("total frames = %d\n",frames);
	printf("average time per frame = %lf milliseconds\n",(double)total_transform_time/frames);



	return 0;
}


/*-----------------------------------------------------------------
 * @Function : detectAndDisplay(Mat&, CascadeClassifer&)
 * @Input : This function detects the vehicles using HAAR cascades
 * 	    to detect the cars using a car.xml file available as
 * 	    given.Then it creates a box around the same on the
 * 	    input frame
 * @Return Value : void
 * ----------------------------------------------------------------*/

void detectAndDisplay(Mat& frame, CascadeClassifier& cascade)
{
	Mat frame_gray;
    	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    	equalizeHist(frame_gray, frame_gray);

    	std::vector<Rect> faces;
    	cascade.detectMultiScale(frame_gray, faces);
    	for (auto& face : faces)
        	rectangle(frame, face, Scalar(255, 255, 0), 2);

}
