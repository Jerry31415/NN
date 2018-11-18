#include "MLP.h"
#include "CNN.h"
#include "GDM.h"
#include <omp.h>

#define OMP_SUPPORT

void Mat2Vec(const Mat& src, std::vector<double>& dst){
	if (dst.size() != src.rows*src.cols) dst.resize(src.rows*src.cols, 0);
	for (int i = 0; i < src.rows; ++i){
		for (int j = 0; j < src.cols; ++j){
			dst[i*src.cols + j] = src.at<uchar>(i, j);
		}
	}
}

int argmax(const std::vector<double>& v){
	int ind = 0;
	for (int i = 1; i < v.size(); ++i){
		if (v[i] > v[ind]) ind = i;
	}
	return ind;
}

class TrainNet : public GDM{
public:
	CNN cnn;
	MLP mlp;
	int mlp_offset;
	std::vector<double> W;
	std::vector<Mat> images;
	std::vector<std::vector<double>> true_vectors;

	std::vector<int> batch_indexes;
	std::vector<int> batch;
	int batch_size;

	TrainNet(){}

	void CalcBatchIndexes(){
		batch_indexes.resize(images.size(), 0);
		for (int i = 0; i < batch_indexes.size(); ++i){
			batch_indexes[i] = i;
		}
		std::random_shuffle(batch_indexes.begin(), batch_indexes.end());
	}

	void NextBatch(){
		if (batch_indexes.empty()) CalcBatchIndexes();
		static int batch_index = 0;
		if (batch_index + batch_size >= batch_indexes.size()){
			std::random_shuffle(batch_indexes.begin(), batch_indexes.end());
			batch_index = 0;
		}
		else {
			if (batch.size() != batch_size){
				batch.resize(batch_size, 0);
			}
			for (int i(batch_index),j(0); i < batch_index + batch_size; ++i,++j){
				batch[j] = batch_indexes[i];
			}
			batch_index += batch_size;
		}
	}

	TrainNet(std::vector<cnn_layer>& cnnLayers, const std::vector<int>& mlpLayers, int batch_size_ = 300){
		cnn = cnnLayers;
		mlp = mlpLayers;
		W.resize(/*cnn.weightsNumber() +*/ mlp.weightsNumber(), 0);
		mlp_offset = 0;// mlp.weightsNumber();
		init(W.size());
		batch_size = batch_size_;
	}

	void NET(const Mat& src, std::vector<double>& dst){
		/*std::vector<double> v_res_cnn_layer(cnn.layers.back().conv_mat.size(), 0);
		cnn.calc(src, W, mlp_offset);
		if (cnn.layers.back().usePooling){
			for (int i = 0; i < cnn.layers.back().depth; ++i){
				v_res_cnn_layer[i] = cnn.layers.back().pooling_mat[i].at<double>(0);
			}
		}
		else {
			for (int i = 0; i < cnn.layers.back().depth; ++i){
				v_res_cnn_layer[i] = cnn.layers.back().conv_mat[i].at<double>(0);
			}
		}*/
		static std::vector<double> vecSrc;
		Mat2Vec(src, vecSrc);
		mlp.getA_ReLU(vecSrc, dst, W);
		Softmax(dst);
	}

	inline double dist(const std::vector<double>& a, const std::vector<double>& b){
		double res(0), t;
		for (int i = 0; i < a.size(); ++i){
			t = a[i] - b[i];
			res += t*t;
		}
		return res;
	}
	
#ifndef OMP_SUPPORT 
	double S(std::vector<double>& arg){
		std::vector<double> res(mlp.LS.back(), 0);
		double sum(0), d;
		for (int i = 0; i < images.size(); ++i){
			NET(images[i], res);
			d = dist(res, true_vectors[i]);
			sum += d;
			if (i % 1000 == 0) std::cout << 100.*(i + 1000) / images.size() << "% - ready sum=" << sum/(i+1) << "\n";
		}
		return sum / images.size();
	}
#else
	double S(std::vector<double>& arg){
		double sum(0), d;
		int i;
		std::vector<double> dv(batch_size, 0);
		#pragma omp parallel for shared(dv) private(i)
		for (i = 0; i < batch.size(); ++i){
			std::vector<double> res(mlp.LS.back(), 0);
			NET(images[batch[i]], res);
			d = dist(res, true_vectors[batch[i]]);
			dv[i] += d;
		}
		for (auto& e : dv) sum += e;
		return sum / images.size();
	}
#endif

	double Accuracy(std::vector<uchar>& test_labels, std::vector<Mat>& test_images){
		std::vector<double> res(mlp.LS.back(), 0);
		double tp(0),tn;
		for (int i = 0; i < test_images.size(); ++i){
			NET(test_images[i], res);
			if (argmax(res) == test_labels[i]){
				tp++;
			}
		}
		return tp / test_images.size();
	}


};