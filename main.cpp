#include <iostream>
#include <opencv2\imgcodecs\imgcodecs.hpp>
#include "MinistReader.h"
#include "TrainNet.h"

#include <ctime>
#include <chrono>

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
|convolution block| = (3x3+1)x(8+2+2+4)=10*16 = 220
|MLP block| = 4458
|net| = 4458+220=4678

*/

void initCNN_layers(std::vector<cnn_layer>& Layers){
	bool usePoolingAll = true;
	Layers.resize(4);
	Layers[0].conv_size = 3;
	Layers[0].depth = 8;
	Layers[0].pooling_size = 2;
	Layers[0].pooling_stride = 2;
	Layers[0].useBias = true;
	Layers[0].usePooling = usePoolingAll;
	Layers[0].useReLU = true;
	Layers[0].mat_size = Size(28,28);

	Layers[1].conv_size = 3;
	Layers[1].depth = 16;
	Layers[1].pooling_size = 2;
	Layers[1].pooling_stride = 2;
	Layers[1].useBias = true;
	Layers[1].usePooling = usePoolingAll;
	Layers[1].useReLU = true;
	Layers[1].mat_size = Size(14, 14);

	Layers[2].conv_size = 3;
	Layers[2].depth = 32;
	Layers[2].pooling_size = 3;
	Layers[2].pooling_stride = 2;
	Layers[2].useBias = true;
	Layers[2].usePooling = usePoolingAll;
	Layers[2].useReLU = true;
	Layers[2].mat_size = Size(7, 7);

	Layers[3].conv_size = 3;
	Layers[3].depth = 128;
	Layers[3].pooling_size = 3;
	Layers[3].pooling_stride = 1;
	Layers[3].useBias = true;
	Layers[3].usePooling = usePoolingAll;
	Layers[3].useReLU = true;
	Layers[3].mat_size = Size(3, 3);
}

void Labels2Vec(std::vector<uchar> src, std::vector<std::vector<double>>& dst, int max_elem = 9){
	for (int i = 0; i < src.size(); ++i){
		std::vector<double> tmp(10, 0);
		tmp[src[i]] = 1;
		dst.push_back(tmp);
	}
}
/*
z = 2*x*x-x*y+y*y-7*x+6*y-2
z(8/7;-17/7) = -93/7
z(1.143, -2.4286) -13.286
*/


int main(){
	std::vector<cnn_layer> Layers;
	initCNN_layers(Layers);

	std::vector<double> result;

	TrainNet train(Layers, { 28*28, 10 }, 100);

	// загружаем элементы обучающей выборки
	std::vector<uchar> labels;
	ReadLabels("train-labels.idx1-ubyte", labels);
	Labels2Vec(labels, train.true_vectors);
	ReadImages("train-images.idx3-ubyte", train.images, CV_64FC1);


	std::vector<uchar> test_labels;
	std::vector<Mat> test_images;
	ReadLabels("t10k-labels.idx1-ubyte", test_labels);
	ReadImages("t10k-images.idx3-ubyte", test_images, CV_64FC1);



	std::cout << "init - ok\n";

	for (int i = 0; i < train.W.size(); ++i){
		train.W[i]=(double)(1.)/(1+rand()%10000); 
	}

	while (true){
		train.NextBatch();
		std::chrono::time_point<std::chrono::system_clock> start, end;
		start = std::chrono::system_clock::now();
		//for (int i = 0; i < 2; ++i){
			std::cout << "Accuracy=" << train.Accuracy(test_labels, test_images) << "\n";
			train.solve(1, train.W, true);
		//}
		end = std::chrono::system_clock::now();
		int elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
		std::cout << elapsed_seconds << "s\n";
	}

	std::cin.get();
}
//12