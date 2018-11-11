#include "MLP.h"

using namespace cv;

template<typename T>
std::string toString(const T& val){
	std::stringstream ss;
	ss << val;
	return ss.str();
}

void Softmax(std::vector<double>& v){
	double den = 0;
	for (auto& e : v) den += exp(e);
	for (int i = 0; i < v.size(); ++i) v[i] = exp(v[i]) / den;
}

MLP::MLP(const std::vector<int>& layers_size){
	LS.resize(layers_size.size());
	std::copy(layers_size.begin(), layers_size.end(), LS.begin());
	w.resize(weightsNumber(), 0);
}

int MLP::weightsNumber(){
	int sum = 0;
	for (size_t i = 1; i < LS.size(); ++i){
		sum += (LS[i - 1] + 1)*LS[i];
	}
	return sum;
}

int MLP::neuronNumber(){
	int sum = 0;
	for (size_t i = 1; i < LS.size()-1; ++i){
		sum += LS[i];
	}
	return sum;
}

void MLP::get(const std::vector<double>& arg, std::vector<double>& dst){
	if (arg.size() != LS[0]) {
		std::cerr << "Error:arg.size!=L0.size. Incorrect arguments number\n";
		return;
	}
	std::vector<double> r;
	r.resize(arg.size(), 0);
	std::copy(arg.begin(), arg.end(), r.begin());
	int wInd(0), rInd(0);
	for (size_t k = 1; k < LS.size(); ++k){
		for (int z = 0; z < LS[k]; ++z){
			double sum = 0;
			for (int t = 0; t < LS[k - 1]; ++t){
				sum += w[wInd + t] * r[t + rInd];
			}
			sum += w[wInd + LS[k - 1]];
			wInd += LS[k - 1] + 1;
			r.push_back(sum);
		}
		rInd += LS[k - 1];
	}
	for (size_t i = r.size() - LS[LS.size() - 1]; i < r.size(); ++i){
		dst.push_back(r[i]);
	}
	if (dst.size() != LS[LS.size() - 1]) throw std::runtime_error("Error: dst.size() != LS[LS.size() - 1]");
}

void MLP::getA(const std::vector<double>& arg, std::vector<double>& dst, const std::function<double(double)>& activation_func){
	if (arg.size() != LS[0]) {
		std::cerr << "Error:arg.size!=L0.size. Incorrect arguments number\n";
		return;
	}
	std::vector<double> r;
	r.resize(arg.size(), 0);
	std::copy(arg.begin(), arg.end(), r.begin());
	int wInd(0), rInd(0);
	for (size_t k = 1; k < LS.size(); ++k){
		for (int z = 0; z < LS[k]; ++z){
			double sum = 0;
			for (int t = 0; t < LS[k - 1]; ++t){
				sum += w[wInd + t] * r[t + rInd];
			}
			sum += w[wInd + LS[k - 1]];
			wInd += LS[k - 1] + 1;
			r.push_back(activation_func(sum));
		}
		rInd += LS[k - 1];
	}
	for (size_t i = r.size() - LS[LS.size() - 1]; i < r.size(); ++i){
		dst.push_back(r[i]);
	}
}

void MLP::getA_ReLU(const std::vector<double>& arg, std::vector<double>& dst){
	getA(arg, dst, [](const double& v){return (v < 0) ? 0 : v; });
}

void MLP::get(const std::vector<double>& arg, std::vector<double>& dst, const std::vector<double>& W){
	if (arg.size() != LS[0]) {
		std::cerr << "Error:arg.size!=L0.size. Incorrect arguments number\n";
		return;
	}
	std::vector<double> r;
	r.resize(arg.size(), 0);
	std::copy(arg.begin(), arg.end(), r.begin());
	int wInd(0), rInd(0);
	for (size_t k = 1; k < LS.size(); ++k){
		for (int z = 0; z < LS[k]; ++z){
			double sum = 0;
			for (int t = 0; t < LS[k - 1]; ++t){
				sum += W[wInd + t] * r[t + rInd];
			}
			sum += W[wInd + LS[k - 1]];
			wInd += LS[k - 1] + 1;
			r.push_back(sum);
		}
		rInd += LS[k - 1];
	}
	for (size_t i = r.size() - LS[LS.size() - 1]; i < r.size(); ++i){
		dst.push_back(r[i]);
	}
	if (dst.size() != LS[LS.size() - 1]) throw std::runtime_error("Error: dst.size() != LS[LS.size() - 1]");
}

void MLP::getA(const std::vector<double>& arg, std::vector<double>& dst, const std::vector<double>& W, const std::function<double(double)>& activation_func){
	if (arg.size() != LS[0]) {
		std::cerr << "Error:arg.size!=L0.size. Incorrect arguments number\n";
		return;
	}
	std::vector<double> r;
	r.resize(arg.size(), 0);
	std::copy(arg.begin(), arg.end(), r.begin());
	int wInd(0), rInd(0);
	for (size_t k = 1; k < LS.size(); ++k){
		for (int z = 0; z < LS[k]; ++z){
			double sum = 0;
			for (int t = 0; t < LS[k - 1]; ++t){
				sum += W[wInd + t] * r[t + rInd];
			}
			sum += W[wInd + LS[k - 1]];
			wInd += LS[k - 1] + 1;
			r.push_back(activation_func(sum));
		}
		rInd += LS[k - 1];
	}
	if (dst.size() != (LS[LS.size() - 1])){
		dst.resize(LS[LS.size() - 1], 0);
	}
	for (size_t i(r.size() - LS[LS.size() - 1]), j(0); i < r.size(); ++i,++j){
		dst[j]=r[i];
	}
}

void MLP::getA_ReLU(const std::vector<double>& arg, std::vector<double>& dst, const std::vector<double>& W){
	getA(arg, dst, W, [](const double& v){return (v < 0) ? 0 : v; });
}

void MLP::SaveImage(const std::string& name, Size size){
	int mxelem = *std::max_element(LS.begin(), LS.end());
	int w(size.width), h(size.height);
	int step_x = (w - 200) / LS.size();
	int step_y = (h - 200) / mxelem;
	Mat im = Mat::zeros(h, w, CV_8UC3);
	Point2i p(100, 100);
	for (size_t k = 1; k < LS.size(); ++k){
		for (int z = 0; z < LS[k]; ++z){
			Point2i sp(step_x*(k + 1), -(h - 200) / (2*LS[k]) + (z + 1) * (h - 200) / LS[k]);
			for (int t = 0; t < LS[k - 1]; ++t){
				Point2i fp(step_x*k, -(h - 200) / (2*LS[k - 1]) + (t + 1) * (h - 200) / LS[k - 1]);
				line(im, p + fp, p + sp, Scalar(255, 128, 128));
			}
		}
	}
	Scalar color;
	Point2i pc;
	for (size_t k = 0; k < LS.size(); ++k){
		for (int z = 0; z < LS[k]; ++z){
			pc = p + Point2i(step_x*(k + 1), -(h - 200) / (2 * LS[k]) + (z + 1) * (h - 200) / LS[k]);
			if (k == 0){
				color = Scalar(128, 128, 255);
				putText(im, "in", Point2i(pc.x - 25, pc.y + 3), FONT_HERSHEY_PLAIN, 1, color);
			}
			else if (k == LS.size() - 1) {
				color = Scalar(128, 255, 128);
				putText(im, "out", Point2i(pc.x + 10, pc.y + 3), FONT_HERSHEY_PLAIN, 1, color);
			}
			else color = Scalar(255, 255, 255);
			circle(im, pc, 3, color, 6);
		}
	}


	putText(im, "Hidden layers: "+toString(LS.size()-2), Point2i(5,20), FONT_HERSHEY_PLAIN, 1, Scalar(128,255,128));
	putText(im, "Input: " + toString(LS[0]), Point2i(5, 35), FONT_HERSHEY_PLAIN, 1, Scalar(128, 255, 128));
	putText(im, "Output: " + toString(LS.back()), Point2i(5, 50), FONT_HERSHEY_PLAIN, 1, Scalar(128, 255, 128));
	putText(im, "Neurons: " + toString(neuronNumber()), Point2i(5, 65), FONT_HERSHEY_PLAIN, 1, Scalar(128, 255, 128));
	putText(im, "Parameters: " + toString(weightsNumber()), Point2i(5, 80), FONT_HERSHEY_PLAIN, 1, Scalar(128, 255, 128));
	
	imwrite(name.c_str(), im);
}