#include <vector>
#include <opencv2\core\core.hpp>

using namespace cv;

/*
convolution block
--------------------------------------------------------------------
	               R = 2			   R = 2			   R = 3
		  conv 8   step = 2  conv 16   step = 2  conv 32   step = 2
input    conv3x3   max pool  conv3x3   max pool  conv 3x3  max pool
28x28x1->28x28x8->14x14x8->  14x14x16->  7x7x16->  7x7x32->  3x3x32->

			R = 3
conv 128	step = 1
conv 3x3	max pool
3x3x128  -> 1x1x128


MLP block
--------------------------------------------------------------------
  1x1x128     1x1x32       1x1x10 (4448 ����������)
  
  1x1x288     1x1x36       1x1x10
->full layer->hidden layer->output
288
|convolution block| = (3x3+1)x(8+2+4+8)=10*22 = 220
|MLP block| = 4448
|net| = 4448+220=4668

*/

/*

    9*(22)
8x8x1 -> 8x8x4 -> 8x8x8 -> 4x4x10 
Test CNN
1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1

3x3
1 1 1 
1 1 1
1 1 1


*/

void Conv(const Mat& src, const Mat& kernel, Mat& dst);
void Conv(const Mat& src, const std::vector<double>& kernel, Mat& dst, int ker_size, int ker_pos = 0, bool useBias = false);

void MaxPooling(const Mat& src, Mat& dst, int R = 2, int stride = 2);
void MaxPooling(Mat& m, int R = 2, int stride = 2);
void MaxPooling(const std::vector<Mat>& src, std::vector<Mat>& dst, int R = 2, int stride = 2);
void MaxPooling(std::vector<Mat>& lay, int R = 2, int stride = 2);

void ReLU(Mat& m);

struct cnn_layer{
	std::vector<Mat> conv_mat; // ��������������� �������. depth ����
	std::vector<Mat> pooling_mat; // ��������������� ������� ����� �������. depth ����
	int conv_size; // ������ ���� ������� ker_size x ker_size
	int depth; // ������� ����
	int pooling_size; // ������ ���� ������� pooling_size x pooling_size
	int pooling_stride; // ��� �������
	bool useBias; // �������� ����������� �������� � ���������� �������
	bool useReLU; // ������������ ������� ��������� ReLU ��� ������� ����
	bool usePooling; // �������� subsampling ����� �������
	Size mat_size; // ������ ������ � conv_mat �� �������

	cnn_layer() : usePooling(false), pooling_size(2), pooling_stride(2), useBias(true), useReLU(true), conv_size(3), depth(1){
	}

	cnn_layer(const cnn_layer& L);

	void ReLU(); // ��������� max(0,x) ��� ���� conv_mat
	void MaxPooling(); // ��������� ������� ������ � conv_mat
	Size poolingMatSize(); // ���������� ������ ������� ����� �������
};

class CNN {
	public:
	std::vector<double> w;
	std::vector<cnn_layer> layers;
	
	int convolutionNumber(); // ��������� ����� ������� ��� ������������ layers
	int weightsNumber(); // ��������� ����� ���������� ����
	
	CNN(){}
	CNN(std::vector<cnn_layer>& Layers); // �������� ������ �� layers=Layers. �������������� ��������� w ������

	// ��������� ��������� ������ ���� ��� ����� input (CV_64FC1).
	// ��������� ��������� � layers.back()
	// offset - �������� ������������ w. ������ ������� ����������� ���� � ������� w
	void calc(const Mat& input, int offset = 0);

	// ��������� ��������� ������ ���� ��� ����� input (CV_64FC1).
	// ��������� ��������� � layers.back()
	// W - ��������� ���� (���������� ����)
	// offset - �������� ������������ W. ������ ������� ����������� ���� � ������� w
	void calc(const Mat& input, const std::vector<double>& W, int offset = 0);

	std::vector<Mat>& get(const Mat& input, const std::vector<double>& W, int offset = 0){
		calc(input, W, offset);
		return (layers.back().usePooling) ? layers.back().pooling_mat : layers.back().conv_mat;
	}
};