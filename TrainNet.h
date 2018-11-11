#include "MLP.h"
#include "CNN.h"
#include "GDM.h"

class TrainNet : public GDM{
	std::vector<double> v_res_cnn_layer;
	std::vector<Mat> res_cnn_layer;
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
		v_res_cnn_layer.resize(cnn.layers.back().conv_mat.size(), 0);
	}

	void NET(const Mat& src, std::vector<double>& dst){
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

	double dist(const std::vector<double>& a, const std::vector<double>& b){
		double res(0), t;
		for (int i = 0; i < a.size(); ++i){
			t = a[i] - b[i];
			res += t*t;
		}
		return res;
	}

	double S(std::vector<double>& arg){
		std::vector<double> res(mlp.LS.back(), 0);
		double sum(0), d;
		for (int i = 0; i < images.size(); ++i){
			NET(images[i], res);
			d = dist(res, true_vectors[i]);
			sum += d;
			if (i % 1000 == 0) std::cout << 100.*(i + 1000) / images.size() << "% - ready sum=" << sum << "\n";
		}
		return sum;
	}
};
/*
#include <thread>

class TrainNet2T : public GDM{
	std::vector<double> v_res_cnn_layer;
	std::vector<Mat> res_cnn_layer;


	std::vector<cnn_layer> cnnLayers_first;
	std::vector<cnn_layer> cnnLayers_second;

	bool first_thr, second_thr;
	double sum0, sum1;
public:
	TrainNet tr0;
	TrainNet tr1;
	int mlp_offset;
	std::vector<double> W;
	std::vector<Mat> images;
	std::vector<std::vector<double>> true_vectors;

	TrainNet2T(std::vector<cnn_layer>& cnnLayers, const std::vector<int>& mlpLayers){
		first_thr = second_thr = false;
		sum0 = sum1 = 0;
		for (int i = 0; i < cnnLayers.size(); ++i){
			cnnLayers_first.push_back(cnnLayers[i]);
			cnnLayers_second.push_back(cnnLayers[i]);
		}
		tr0 = TrainNet(cnnLayers_first, mlpLayers);
		tr1 = TrainNet(cnnLayers_second, mlpLayers);
	}

	void init(){
		for (int i = 1; i < images.size(); i += 2){
			tr0.images.push_back(images[i - 1]);
			tr1.images.push_back(images[i]);
			tr0.true_vectors.push_back(true_vectors[i - 1]);
			tr1.true_vectors.push_back(true_vectors[i]);
		}
	}

	double dist(const std::vector<double>& a, const std::vector<double>& b){
		double res(0), t;
		for (int i = 0; i < a.size(); ++i){
			t = a[i] - b[i];
			res += t*t;
		}
		return res;
	}
	
	void th_func0(std::vector<double> arg){
		first_thr = true;
		sum0 = tr0.S(arg);
		first_thr = false;
	}

	void th_func1(std::vector<double> arg){
		second_thr = true;
		sum1 = tr1.S(arg);
		second_thr = false;
	}

	double S(std::vector<double>& arg){
		std::thread thread0(&TrainNet2T::th_func0, arg);
		std::thread thread1(&TrainNet2T::th_func1, arg);
		thread0.detach();
		thread1.detach();
		
		while (true){
			_sleep(10);
			if (!first_thr && !second_thr) break;
		}
		return sum0 + sum1;
	}
};
*/
