//#pragma once
#include <vector>
#include <algorithm>
#include <opencv2\core.hpp>
#include <iostream>

using namespace cv;

// Находит локльный минимум функции S от N аргументов
class GDM{
	Mat G;
	std::vector<double> t;
public:

	virtual ~GDM(){
	}

	void init(int N){
		t.resize(N, 0);
	}

	// Функция S должна быть определена пользователем
	virtual double S(std::vector<double>& arg) = 0;

	// Возвращает численную аппроксимацию первой 
	// частной производной функции S по аргументу с номером k
	double dS_arg_k(std::vector<double>& arg, int k, double eps = 0.001){
		if (k < 0 || k >= t.size()){
			throw std::runtime_error("Error in dS_arg_k function: incorrect k argument value\n");
		}
		std::copy(arg.begin(), arg.end(), t.begin());
		double S0, S1;
		t[k] -= eps;
		S0 = S(t);
		t[k] += 2 * eps;
		S1 = S(t);
		return (S1 - S0) / (2.*eps);
	}

	// Возвращает численную аппроксимацию первой 
	// частной производной функции S по аргументу с номером k
	double dS_arg_kO(std::vector<double>& arg, int k, double SAarg){
		double S0;
		arg[k] -= 1;
		S0 = S(arg);
		arg[k] += 1;
		return (SAarg - S0);
	}

	// Возвращет в dst вектор-градиент функции S(arg)
	void gradient(std::vector<double>& arg, Mat& dst, double eps = 0.001){
		dst = Mat::zeros(arg.size(), 1, CV_64FC1);
		double SArg = S(arg);
		std::cout << "SArg - ok\n";
		for (int k = 0; k < arg.size(); ++k){
			dst.at<double>(k, 0) = dS_arg_kO(arg, k, SArg);
			std::cout << (k + 1) << " / " << arg.size() << "\n";
		}
	}

	// Производит itter иттераций метода Ньютона для минимизации S.
	// Результат возвращает в dst.
	// Если useApproximation = true, то использует значение dst в качестве начального приближения 
	void solve(int itter, std::vector<double>& dst, bool useApproximation = false, double eps = 1){

		auto dist = [](const Mat& v){
			double res = 0;
			for (int i = 0; i < v.rows; ++i){
				res += v.at<double>(i, 0)*v.at<double>(i, 0);
			}
			return sqrt(res);
		};

		if (dst.empty()) {
			if (useApproximation) throw std::runtime_error("Error in solve function: dst is empty\n");
			dst.resize(t.size(), 0);
		}
		int cnt = 0;
		for (int z = 0; z < t.size(); ++z) t[z] = 0;
		double S0, S1;
		bool shake = false;
		do{
			S0 = S(dst);
			gradient(dst, G, eps);
			double GD = dist(G);
			if (GD < 0.00005) {
				if(shake) break;
				else {
					shake = true;
					for (int i = 0; i < G.rows; ++i) G.at <double>(i, 0) += 0.1;
					GD = dist(G);
					//continue;
				}
			}
			for (int z = 0; z < dst.size(); ++z) dst[z] = dst[z] - (1. / GD)*G.at<double>(z, 0);
			S1 = S(dst);
		/*	if (S1>S0){
				for (int z = 0; z < dst.size(); ++z) dst[z] = dst[z] + (1. / GD)*G.at<double>(z, 0);
				return;
			}*/
			++cnt;
		} while (cnt<itter);
	}
};