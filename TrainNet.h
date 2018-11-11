#include "MLP.h"
#include "CNN.h"
#include "GDM.h"
#include <omp.h>

#define OMP_SUPPORT

class TrainNet : public GDM{
public:
	CNN cnn;
	MLP mlp;
	int mlp_offset;
	std::vector<double> W;
	std::vector<Mat> images;
	std::vector<std::vector<double>> true_vectors;

	TrainNet(){}

	TrainNet(std::vector<cnn_layer>& cnnLayers, const std::vector<int>& mlpLayers){
		cnn = cnnLayers;
		mlp = mlpLayers;
		W.resize(cnn.weightsNumber() + mlp.weightsNumber(), 0);
		mlp_offset = mlp.weightsNumber();
	}

	void NET(const Mat& src, std::vector<double>& dst){
		std::vector<double> v_res_cnn_layer(cnn.layers.back().conv_mat.size(), 0);
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
		}
		mlp.getA_ReLU(v_res_cnn_layer, dst, W);
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
		std::vector<double> dv(images.size(), 0);
		#pragma omp parallel for shared(dv) private(i)
		for (i = 0; i < images.size(); ++i){
			std::vector<double> res(mlp.LS.back(), 0);
			NET(images[i], res);
			d = dist(res, true_vectors[i]);
			dv[i] += d;
		}
		for (auto& e : dv) sum += e;
		return sum / images.size();
	}
#endif
};