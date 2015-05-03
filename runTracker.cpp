#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include "STCTracker.h"
/*****************************************************************************/
//在命令行
//1)输入 -b init.txt -v david.mpg则选择david视频并且初始位置固定
//2)输入 -v david.mpg则选择david视频并且初始位置需要手动设置
//3)不输入任何命令则打开摄像头，手动设定初始位置
/*****************************************************************************/
//-v david.mpg -b david_init.txt
#define STC 1

// Global variables
Rect box;
Rect STCbox;
bool drawing_box = false;
bool gotBB = false;
Rect boxRegion;//STC algorithm display context region 
bool fromfile =false;
string video;

static double TestTime=0.f;
int frameCount=1;

void readBB(char* file)	// get tracking box from file
{
	ifstream tb_file (file);
	string line;
	getline(tb_file, line);
	istringstream linestream(line);
	string x1, y1, w1, h1;
	getline(linestream, x1, ',');
	getline(linestream, y1, ',');
	getline(linestream, w1, ',');
	getline(linestream, h1, ',');
	int x = atoi(x1.c_str());
	int y = atoi(y1.c_str());
	int w = atoi(w1.c_str());
	int h = atoi(h1.c_str());
	box = Rect(x, y, w, h);
}

void print_help(void)
{
	printf("-v    source video\n-b        tracking box file\n");
}

void read_options(int argc, char** argv, VideoCapture& capture)
{
	for (int i=0; i<argc; i++)
	{
		if (strcmp(argv[i], "-b") == 0)	// read tracking box from file
		{
			//printf("-b%d\n",i);
			if (argc>i)
			{
				readBB(argv[i+1]);
				gotBB = true;
			}
			else
			{
				print_help();
			}
		}
		if (strcmp(argv[i], "-v") == 0)	// read video from file
		{
			//printf("-v%d\n",i);
			if (argc>i)
			{
				video = string(argv[i+1]);
				capture.open(video);				
				fromfile = true;
			}
			else
			{
				print_help();
			}
		}
	}
}

// bounding box mouse callback
void mouseHandler(int event, int x, int y, int flags, void *param){
  switch( event ){
  case CV_EVENT_MOUSEMOVE:
    if (drawing_box){
        box.width = x-box.x;
        box.height = y-box.y;
    }
    break;
  case CV_EVENT_LBUTTONDOWN:
    drawing_box = true;
    box = Rect( x, y, 0, 0 );
    break;
  case CV_EVENT_LBUTTONUP:
    drawing_box = false;
    if( box.width < 0 ){
        box.x += box.width;
        box.width *= -1;
    }
    if( box.height < 0 ){
        box.y += box.height;
        box.height *= -1;
    }
    gotBB = true;
    break;
  }
}

int main(int argc, char * argv[])
{
	VideoCapture capture;

	// Read options
	read_options(argc, argv,capture);
	Mat frame;
	Mat first;
	if (argc==1)
	{
		capture.open(0);
	}
	// Init camera
	if (!capture.isOpened())
	{
		cout<<"capture device failed to open!"<<endl;
		return -1;
	}

	//display video frame rate
	float rate=capture.get(CV_CAP_PROP_FPS);
	cout<<"rate="<<rate<<endl;

	//
	cout<<"totalframenumber="<<capture.get(CV_CAP_PROP_FRAME_COUNT)<<endl;

	//Register mouse callback to draw the bounding box
	cvNamedWindow("Tracker", CV_WINDOW_AUTOSIZE);
	cvSetMouseCallback("Tracker", mouseHandler, NULL );
	//保存图像的路径
	string imgFormat=".\\imgs\\img%05d.png";
	char image_name[256];

	if (fromfile)
	{
		capture.set(CV_CAP_PROP_POS_FRAMES,0*rate);
		capture >> frame;
		frameCount++;
		frame.copyTo(first);
		//显示帧数
		stringstream buf;
		buf << frameCount;
		string num = buf.str();
		putText(frame, num, Point(15, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(100, 0, 255), 3);
		//显示
		imshow("Tracker", frame);
		//保存
		sprintf(image_name, imgFormat.c_str(), frameCount);
		imwrite(image_name,frame);
	}
	else
	{
		capture.set(CV_CAP_PROP_FRAME_WIDTH, 320);
		capture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	}

	// Initialization
	while(!gotBB)
	{	
		if (!fromfile)
		{
			capture >> frame;
		}
		else
		{
			first.copyTo(frame);
		}
		rectangle(frame, box, Scalar(0,0,255));
		imshow("Tracker", frame);
		if (cvWaitKey(33) == 'q') 
		{	return 0; }
	}

	//Remove callback
	cvSetMouseCallback("Tracker", NULL, NULL ); 
	
	//display pos of init box 
	cout<<"init x="<<box.x<<" int y="<<box.y<<endl;


#ifdef STC
	STCbox=box;
	//STC initialization
	STCTracker stcTracker;
	stcTracker.init(frame, STCbox,boxRegion);
#endif
	Mat last_gray;
	cvtColor(frame, last_gray, CV_RGB2GRAY);

	// Run-time
	Mat current_gray;
	
	while (1)
	{
		capture >> frame;
		if (frame.empty())
			break;
		
#ifdef STC
		double t = (double)cvGetTickCount();
		// tracking
		stcTracker.tracking(frame, STCbox,boxRegion,frameCount);	
		t = (double)cvGetTickCount() - t;
		cout << "cost time: " << t / ((double)cvGetTickFrequency()*1000000.) <<" s,";
		cout<<cvRound(((double)cvGetTickFrequency()*1000000.)/t)<<"FPS"<<endl;
		cout<<"object size:="<<STCbox.width<<"*"<<STCbox.height<<endl;
#endif
		frameCount++;

		// show the result
		
		//显示当前帧
		stringstream buf;
		buf << frameCount;
		string num = buf.str();
		putText(frame, num, Point(15, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(100, 0, 255), 3);

#ifdef STC
		//putText(frame, "      STC", Point(80, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 3);
		rectangle(frame, STCbox, Scalar(0, 0, 255), 3);
		//rectangle(frame, boxRegion, Scalar(255, 200, 255), 3);
#endif

		imshow("Tracker", frame);
		sprintf(image_name, imgFormat.c_str(), frameCount);
		imwrite(image_name,frame);
		if ( cvWaitKey(1) == 27 )
			break;
	}

	//等待输入
	//cin.get();
	return 0;
}