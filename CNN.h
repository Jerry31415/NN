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
  1x1x128     1x1x32       1x1x10 (4448 параметров)
  
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
	std::vector<Mat> conv_mat; // отфильтрованные матрицы. depth штук
	std::vector<Mat> pooling_mat; // отфильтрованные матрицы после пулинга. depth штук
	int conv_size; // размер ядра свертки ker_size x ker_size
	int depth; // глубина слоя
	int pooling_size; // размер окна пулинга pooling_size x pooling_size
	int pooling_stride; // шаг пулинга
	bool useBias; // включить прибавление смещения к результату свертки
	bool useReLU; // использовать фукнцию активации ReLU для данного слоя
	bool usePooling; // включить subsampling после сверток
	Size mat_size; // размер матриц в conv_mat до пулинга

	cnn_layer() : usePooling(false), pooling_size(2), pooling_stride(2), useBias(true), useReLU(true), conv_size(3), depth(1){
	}

	cnn_layer(const cnn_layer& L);

	void ReLU(); // применяет max(0,x) для всех conv_mat
	void MaxPooling(); // уменьшает размеры матриц в conv_mat
	Size poolingMatSize(); // возвращает размер матрицы после пулинга
};

class CNN {
	public:
	std::vector<double> w;
	std::vector<cnn_layer> layers;
	
	int convolutionNumber(); // возвращет число сверток для определенных layers
	int weightsNumber(); // возвращет число параметров сети
	
	CNN(){}
	CNN(std::vector<cnn_layer>& Layers); // копирует ссылку на layers=Layers. Инициализирует параметры w нулями

	// Вычисляет результат работы сети для входа input (CV_64FC1).
	// Результат храниться в layers.back()
	// offset - смещение относительно w. Индекс первого сверточного веса в массиве w
	void calc(const Mat& input, int offset = 0);

	// Вычисляет результат работы сети для входа input (CV_64FC1).
	// Результат храниться в layers.back()
	// W - параметры сети (сверточные веса)
	// offset - смещение относительно W. Индекс первого сверточного веса в массиве w
	void calc(const Mat& input, const std::vector<double>& W, int offset = 0);

	std::vector<Mat>& get(const Mat& input, const std::vector<double>& W, int offset = 0){
		calc(input, W, offset);
		return (layers.back().usePooling) ? layers.back().pooling_mat : layers.back().conv_mat;
	}
};