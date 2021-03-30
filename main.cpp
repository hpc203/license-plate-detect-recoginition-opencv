#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

typedef struct Landmarks_8 {
	Point2f left_top;
	Point2f right_top;
	Point2f left_bottom;
	Point2f right_bottom;
} Landmarks_8;

class lprnet
{
public:
	lprnet()
	{
		this->net = readNet("Final_LPRNet_model.onnx");
	}
	string rec(Mat img);
private:
	Net net;
	const Size plate_size = Size(94, 24);
};

class detect_plate_recognition
{
	public:
		detect_plate_recognition(float confThreshold, float nmsThreshold) :LPR()
		{
			this->confidence_threshold = confThreshold;
			this->nms_threshold = nmsThreshold;
			this->net = readNet("mnet_plate.onnx");
			this->generate_priors();
		}
		void detect_rec(Mat& srcimg);
	private:
		const int im_height = 640;
		const int im_width = 640;
		Net net;
		float confidence_threshold;
		float nms_threshold;
		const int top_k = 1000;
		const int keep_top_k = 500;
		const float vis_thres = 0.6;
		const int min_sizes[3][2] = { {24, 48},{96, 192},{384, 768} };
		const int steps[3] = { 8, 16, 32 };
		const float variance[2] = { 0.1, 0.2 };
		const bool clip = false;
		const int num_prior = 16800;
		float* prior_data;
		const Point2f points_ref[4] = { Point2f{0.0, 0.0}, Point2f{94.0, 0.0} ,Point2f{0.0, 24.0}, Point2f{94.0, 24.0} };

		Mat resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left);
		void generate_priors();
		void decode(const Mat loc, const Mat conf, const Mat landms, vector<Rect>& boxes, vector<float>& confidences, vector<int>& classIds, vector<Landmarks_8>& four_pair_points);
		Mat crop_plate(const Point2f* points_src, const Mat srcimg, const int xmin, const int ymin, const int xmax, const int ymax);
		const Size plate_size = Size(94, 24);
		lprnet LPR;
};

extern const string name = "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新0123456789ABCDEFGHJKLMNPQRSTUVWXYZIO-";
string lprnet::rec(Mat img)
{
	Mat blob = blobFromImage(img, 1 / 128.0, this->plate_size, Scalar(127.5));
	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs);
	int height = outs[0].rows;
	int width = outs[0].cols;
	int i = 0;
	int* preb_label = new int[width];
	for (i = 0; i < width; i++)
	{
		Mat scores = outs[0].col(i).rowRange(0, height);
		Point classIdPoint;
		double score;
		minMaxLoc(scores, 0, &score, 0, &classIdPoint);
		preb_label[i] = classIdPoint.y;
	}

	vector<int> no_repeat_blank_label;
	int pre_c = preb_label[0];
	int last = height - 1;
	if (pre_c != last)
	{
		no_repeat_blank_label.push_back(pre_c);
	}
	int c = 0;
	for (i = 0; i < width; i++)
	{
		c = preb_label[i];
		if ((pre_c == c) || (c == last))
		{
			if (c == last)
			{
				pre_c = c;
			}
			continue;
		}
		no_repeat_blank_label.push_back(c);
		pre_c = c;
	}
	delete [] preb_label;
	int len_s = no_repeat_blank_label.size();
	string result;
	for (i = 0; i < len_s; i++)
	{
		//cout << name[no_repeat_blank_label[i]] << endl;
		result.push_back(name[no_repeat_blank_label[i]]);
	}
	return result;
}

Mat detect_plate_recognition::resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->im_height;
	*neww = this->im_width;
	Mat dstimg;
	if (srch != srcw)
	{
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1)
		{
			*newh = this->im_height;
			*neww = int(this->im_width / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((this->im_width - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->im_width - *neww - *left, BORDER_CONSTANT, 0);
		}
		else
		{
			*newh = (int)this->im_height * hw_scale;
			*neww = this->im_width;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*top = (int)(this->im_height - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, this->im_height- *newh - *top, 0, 0, BORDER_CONSTANT, 0);
		}
	}
	else
	{
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}

void detect_plate_recognition::generate_priors()
{
	this->prior_data = new float[this->num_prior *4];
	float* pdata = prior_data;
	int i = 0, j = 0, h = 0, w = 0;
	for (i = 0; i < 3; i++)
	{
		const int feature_map_height = ceil(this->im_height / this->steps[i]);
		const int feature_map_width = ceil(this->im_width / this->steps[i]);
		for (h = 0; h < feature_map_height; h++)
		{
			for (w = 0; w < feature_map_width; w++)
			{
				for (j = 0; j < 2; j++)
				{
					pdata[0] = (w + 0.5)*this->steps[i] / this->im_width;       ///cx
					pdata[1] = (h + 0.5)*this->steps[i] / this->im_height;      ////cy
					pdata[2] = (float)this->min_sizes[i][j] / this->im_width;  ///width
					pdata[3] = (float)this->min_sizes[i][j] / this->im_height; ///height
					pdata += 4;
				}
			}
		}
	}
}

void detect_plate_recognition::decode(const Mat loc, const Mat conf, const Mat landms, vector<Rect>& boxes, vector<float>& confidences, vector<int>& classIds, vector<Landmarks_8>& four_pair_points)
{
	int i = 0;
	float cx = 0, cy = 0, width = 0, height = 0, x = 0, y = 0;
	int xmin = 0, ymin = 0;
	float* ploc_data = (float*)loc.data;
	float* pconf_data = (float*)conf.data;
	float* plandms_data = (float*)landms.data;
	float* pprior_data = this->prior_data;
	Landmarks_8 four_pair_point;
	for (i = 0; i < this->num_prior; i++)
	{
		if (pconf_data[1] > this->confidence_threshold)
		{
			confidences.push_back(pconf_data[1]);
			cx = pprior_data[0] + ploc_data[0] * this->variance[0] * pprior_data[2];
			cy = pprior_data[1] + ploc_data[1] * this->variance[0] * pprior_data[3];
			width = pprior_data[2] * exp(ploc_data[2] * this->variance[1]);
			height = pprior_data[3] * exp(ploc_data[3] * this->variance[1]);
			xmin = (int)((cx - 0.5 * width)*this->im_width);
			ymin = (int)((cy - 0.5 * height)*this->im_height);
			boxes.push_back(Rect(xmin, ymin, (int)(width*this->im_width), (int)(height*this->im_height)));

			x = pprior_data[0] + plandms_data[4] * this->variance[0] * pprior_data[2];   ///left_top;
			y = pprior_data[1] + plandms_data[5] * this->variance[0] * pprior_data[3];
			four_pair_point.left_top.x = x * this->im_width;
			four_pair_point.left_top.y = y * this->im_height;

			x = pprior_data[0] + plandms_data[6] * this->variance[0] * pprior_data[2];  ///right_top
			y = pprior_data[1] + plandms_data[7] * this->variance[0] * pprior_data[3];
			four_pair_point.right_top.x = x * this->im_width;
			four_pair_point.right_top.y = y * this->im_height;

			x = pprior_data[0] + plandms_data[2] * this->variance[0] * pprior_data[2];   ///left_bottom;
			y = pprior_data[1] + plandms_data[3] * this->variance[0] * pprior_data[3];
			four_pair_point.left_bottom.x = x * this->im_width;
			four_pair_point.left_bottom.y = y * this->im_height;

			x = pprior_data[0] + plandms_data[0] * this->variance[0] * pprior_data[2];  ///right_bottom
			y = pprior_data[1] + plandms_data[1] * this->variance[0] * pprior_data[3];
			four_pair_point.right_bottom.x = x * this->im_width;
			four_pair_point.right_bottom.y = y * this->im_height;
			four_pair_points.push_back(four_pair_point);
		}
		ploc_data += 4;
		pconf_data += 2;
		plandms_data += 8;
		pprior_data += 4;
	}
}

Mat detect_plate_recognition::crop_plate(const Point2f* points_src, const Mat srcimg, const int xmin, const int ymin, const int xmax, const int ymax)
{
	Rect rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1); 
	Mat plate_src = srcimg(rect);
	Mat plate_dst;
	Point2f srcPts[4];
	for (int i = 0; i < 4; i++)
	{
		srcPts[i] = Point2f(points_src[i].x - xmin, points_src[i].y - ymin);
	}
	Mat perspectiveMat = getPerspectiveTransform(srcPts, this->points_ref);
	warpPerspective(plate_src, plate_dst, perspectiveMat, this->plate_size);
	return plate_dst;
}
void detect_plate_recognition::detect_rec(Mat& srcimg)
{
	int newh = 0, neww = 0, top = 0, left = 0;
	Mat img = this->resize_image(srcimg, &newh, &neww, &top, &left);
	Mat blob = blobFromImage(img, 1.0, Size(), Scalar(104, 117, 123));

	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
	
	////post process
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;
	vector<Landmarks_8> four_pair_points;
	this->decode(outs[0], outs[1], outs[2], boxes, confidences, classIds, four_pair_points);
	vector<int> indices;
	NMSBoxes(boxes, confidences, this->confidence_threshold, this->nms_threshold, indices);
	float ratioh = (float)srcimg.rows / newh;
	float ratiow = (float)srcimg.cols / neww;
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		int xmin = (int)max((box.x - left)*ratiow, 0.f);
		int ymin = (int)max((box.y - top)*ratioh, 0.f);
		int xmax = (int)min((box.x - left + box.width)*ratiow, (float)srcimg.cols);
		int ymax = (int)min((box.y - top + box.height)*ratioh, (float)srcimg.rows);
		Landmarks_8 points = four_pair_points[idx];
		Point2f points_src[4];
		points_src[0] = Point2f((points.left_top.x - left)*ratiow, (points.left_top.y - top)*ratioh);
		points_src[1] = Point2f((points.right_top.x - left)*ratiow, (points.right_top.y - top)*ratioh);
		points_src[2] = Point2f((points.left_bottom.x - left)*ratiow, (points.left_bottom.y - top)*ratioh);
		points_src[3] = Point2f((points.right_bottom.x - left)*ratiow, (points.right_bottom.y - top)*ratioh);
		Mat plate_roi = this->crop_plate(points_src, srcimg, xmin, ymin, xmax, ymax);
		string plate_number = LPR.rec(plate_roi);
		//cout << plate_number << endl;

		rectangle(srcimg, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 3);
		/*string label = format("%.2f", confidences[idx]);
		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		ymin = max(ymin, labelSize.height);*/
		//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
		//putText(srcimg, label, Point(xmin, ymin), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);

		for (int j = 0; j < 4; j++)
		{
			circle(srcimg, Point((int)points_src[j].x, (int)points_src[j].y), 2, Scalar(255, 0, 0), -1);
		}
		//imshow("plate_roi", plate_roi);
	}
}

int main()
{
	detect_plate_recognition license_plate(0.02, 0.4);
	string imgpath = "0.jpg";
	Mat srcimg = imread(imgpath);
	license_plate.detect_rec(srcimg);
	
	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}