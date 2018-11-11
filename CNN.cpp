#include "CNN.h"

void Conv(const Mat& src, const Mat& kernel, Mat& dst){
	dst = Mat::zeros(src.size(), CV_64FC1);
	int R((kernel.rows - 1) / 2), xx, yy;
	double conv_res;
	for (int i = 0; i < dst.rows; ++i){
		for (int j = 0; j < dst.cols; ++j){
			conv_res = 0;
			for (int y = -R; y <= R; ++y){
				yy = i + y;
				if (yy < 0 || yy >= src.rows) continue;
				for (int x = -R; x <= R; ++x){
					xx = j + x;
					if (xx < 0 || xx >= src.cols) continue;
					conv_res += src.at<double>(i + y, j + x)*kernel.at<double>(y + R, x + R);
				}
			}
			dst.at<double>(i, j) = conv_res;
		}
	}
}

void Conv(const Mat& src, const std::vector<double>& kernel, Mat& dst, int ker_size, int ker_pos, bool useBias){
	#ifdef Debug
	if (dst.size != src.size){
		dst = Mat::zeros(src.size(), CV_64FC1);
	}
	#endif
	int R((ker_size-1)/2), kInd;
	double conv_res;
	if (useBias){
		for (int i = 0; i < dst.rows; ++i){
			for (int j = 0; j < dst.cols; ++j){
				conv_res = 0;
				kInd = ker_pos;
				if ((-R + i) >= 0 && (R + i) < src.rows){
					for (int y(-R + i); y <= (R + i); ++y){
						if ((-R + j) >= 0 && R + j < src.cols){
							for (int x(-R + j); x <= (R + j); ++x){
								conv_res += src.at<double>(y, x)*kernel[kInd++];
							}
						}
						else{
							for (int x(-R + j); x <= (R + j); ++x){
								if (x < 0 || x >= src.cols) continue;
								conv_res += src.at<double>(y, x)*kernel[kInd++];
							}
						}
					}
				}
				else{
					for (int y(-R + i); y <= (R + i); ++y){
						if (y < 0 || y >= src.rows) continue;
						if ((-R + j) >= 0 && R + j < src.cols){
							for (int x(-R + j); x <= (R + j); ++x){
								conv_res += src.at<double>(y, x)*kernel[kInd++];
							}
						}
						else{
							for (int x(-R + j); x <= (R + j); ++x){
								if (x < 0 || x >= src.cols) continue;
								conv_res += src.at<double>(y, x)*kernel[kInd++];
							}
						}
					}
				}
				conv_res += kernel[kInd];
				dst.at<double>(i, j) = conv_res;
			}
		}
	}
	else {
		for (int i = 0; i < dst.rows; ++i){
			for (int j = 0; j < dst.cols; ++j){
				conv_res = 0;
				for (int y(-R + i), kInd(ker_pos); y <= (R + i); ++y){
					if (y < 0 || y >= src.rows) continue;
					for (int x(-R + j); x <= (R + j); ++x){
						if (x < 0 || x >= src.cols) continue;
						conv_res += src.at<double>(y, x)*kernel[kInd++];
					}
				}
				dst.at<double>(i, j) = conv_res;
			}
		}
	}
}

void MaxPooling(const Mat& src, Mat& dst, int R, int stride){
	double max, tmp;
	#ifdef Debug
	Size s((src.rows - R) / stride + 1, ((src.cols - R) / stride + 1));
	if (dst.size() != s){
		dst = Mat::zeros(s, CV_64FC1);
	}
	#endif
	int RM1(R - 1), ii, jj;
	for (int i(0), y(0); i < src.rows; i += stride, ++y){
		if (i + RM1 >= src.rows) break;
		for (int j(0), x(0); j < src.cols; j += stride, ++x){
			if (j + RM1 >= src.cols) break;
			max = -9999999;
			for (ii = i; ii < i + R; ++ii){
				for (jj = j; jj < j + R; ++jj){
					if ((tmp=src.at<double>(ii, jj))>max) max = tmp;
				}
			}
			dst.at<double>(y, x) = max;
		}
	}
}

void MaxPooling(Mat& m, int R, int stride){
	double max, tmp;
	Mat dst = Mat::zeros((m.rows - R) / stride + 1, ((m.cols - R) / stride + 1), CV_64FC1);
	for (int i(0), y(0); i < m.rows; i += stride, ++y){
		if (i + R >= m.rows + 1) break;
		for (int j(0), x(0); j < m.cols; j += stride, ++x){
			if (j + R >= m.cols + 1) break;
			max = -9999999;
			for (int ii = i; ii < i + R; ++ii){
				for (int jj = j; jj < j + R; ++jj){
					if ((tmp=m.at<double>(ii, jj))>max) max = tmp;
				}
			}
			dst.at<double>(y, x) = max;
		}
	}
	dst.copyTo(m);
}

void MaxPooling(const std::vector<Mat>& src, std::vector<Mat>& dst, int R, int stride){
	if (dst.size() != src.size()){
		Size pooling_mat_size;
		pooling_mat_size.height = (src.begin()->rows - R) / stride + 1;
		pooling_mat_size.width  = (src.begin()->cols - R) / stride + 1;		
		for (int j = 0; j < src.size(); ++j){
			dst.push_back(Mat::zeros(pooling_mat_size, CV_64FC1));
		}
	}
	for (int i = 0; i < src.size(); ++i){
		MaxPooling(src[i], dst[i], R, stride);
	}
}

void MaxPooling(std::vector<Mat>& lay, int R, int stride){
	for (int i = 0; i < lay.size(); ++i){
		MaxPooling(lay[i], R, stride);
	}
}

void ReLU(Mat& m){
	for (int i = 0; i < m.rows; ++i){
		for (int j = 0; j < m.cols; ++j){
			if (m.at<double>(i, j) < 0) m.at<double>(i, j) = 0;
		}
	}
}

// cnn_layer struct
//-------------------------------------------------------------------------

cnn_layer::cnn_layer(const cnn_layer& L){
	for (int i = 0; i < L.conv_mat.size(); ++i){
		conv_mat.push_back(L.conv_mat[i]);
		pooling_mat.push_back(L.pooling_mat[i]);
	}
	conv_size = L.conv_size;
	depth = L.depth;
	pooling_size = L.pooling_size;
	pooling_stride = L.pooling_stride;
	useBias = L.useBias;
	useReLU = L.useReLU;
	usePooling = L.usePooling;
	mat_size = L.mat_size;
}

void cnn_layer::ReLU(){
	for (int i = 0; i < conv_mat.size(); ++i){
		::ReLU(conv_mat[i]);
	}
}

void cnn_layer::MaxPooling(){
	::MaxPooling(conv_mat, pooling_mat, pooling_size, pooling_stride);
}

Size cnn_layer::poolingMatSize(){
	Size pooling_mat_size;
	pooling_mat_size.height = (mat_size.height - pooling_size) / pooling_stride + 1;
	pooling_mat_size.width = (mat_size.width - pooling_size) / pooling_stride + 1;
	return pooling_mat_size;
}

// CNN class
//-------------------------------------------------------------------------

int CNN::convolutionNumber(){
	int sum = 0;
	if (!layers.empty()){
		sum = layers[0].depth;
		for (int i = 1; i < layers.size(); ++i){
			sum += layers[i].depth / layers[i - 1].depth;
		}
	}
	return sum;
}

int CNN::weightsNumber(){
	int sum = 0;
	if (!layers.empty()){
		sum = layers[0].depth*(layers[0].conv_size*layers[0].conv_size + layers[0].useBias);
		for (int i = 1; i < layers.size(); ++i){
			sum += (layers[i].depth / layers[i - 1].depth)*(layers[i].conv_size*layers[i].conv_size + layers[i].useBias);
		}
	}
	return sum;
}

CNN::CNN(std::vector<cnn_layer>& Layers){
	layers = Layers;
	w.resize(weightsNumber(), 0);
	Size pooling_mat_size;
	for (int i = 0; i < layers.size(); ++i){
		pooling_mat_size = layers[i].poolingMatSize();
		for (int j = 0; j < layers[i].depth; ++j){
			layers[i].conv_mat.push_back(Mat::zeros(layers[i].mat_size, CV_64FC1));
			layers[i].pooling_mat.push_back(Mat::zeros(pooling_mat_size, CV_64FC1));
		}
	}
}

/*
98 - ready sum=8841.68
99 - ready sum=8932.25
9021.51
23s
*/

void CNN::calc(const Mat& input, int offset){
	int ker_pos = offset;
	// вычисляем нулевой слой
	for (int i = 0; i < layers[0].depth; ++i){
		Conv(input, w, layers[0].conv_mat[i], layers[0].conv_size, ker_pos, layers[0].useBias);
		ker_pos += layers[0].conv_size*layers[0].conv_size + layers[0].useBias;
	}
	if (layers[0].useReLU) layers[0].ReLU();
	if (layers[0].usePooling) layers[0].MaxPooling();
	for (int L = 1; L < layers.size(); ++L){ // цикл по слоям
		int indCR = 0;
		if (layers[L - 1].usePooling){
			// цикл по сверткам
			for (int i = 0; i < layers[L].depth / layers[L - 1].depth; ++i){
				// цикл по (входным) матрицам с предыдущего слоя
				for (int j = 0; j < layers[L - 1].conv_mat.size(); ++j){
					Conv(layers[L - 1].pooling_mat[j], w, layers[L].conv_mat[indCR], layers[L].conv_size, ker_pos, layers[L].useBias);
					++indCR;
					if (indCR > layers[L].depth){
						throw std::runtime_error("Error: out of range indCR");
					}
				}
				ker_pos += layers[L].conv_size*layers[L].conv_size + layers[L].useBias;
			}
		}
		else {
			// цикл по сверткам
			for (int i = 0; i < layers[L].depth / layers[L - 1].depth; ++i){
				// цикл по (входным) матрицам с предыдущего слоя
				for (int j = 0; j < layers[L - 1].conv_mat.size(); ++j){
					Conv(layers[L - 1].conv_mat[j], w, layers[L].conv_mat[indCR], layers[L].conv_size, ker_pos, layers[L].useBias);
					++indCR;
					if (indCR > layers[L].depth){
						throw std::runtime_error("Error: out of range indCR");
					}
				}
				ker_pos += layers[L].conv_size*layers[L].conv_size + layers[L].useBias;
			}
		}
		if (layers[L].useReLU) layers[L].ReLU();
		if (layers[L].usePooling) layers[L].MaxPooling();
	}
}

void CNN::calc(const Mat& input, const std::vector<double>& W, int offset){
	int ker_pos(offset), indCR;
	// вычисляем нулевой слой
	for (int i = 0; i < layers[0].depth; ++i){
		Conv(input, W, layers[0].conv_mat[i], layers[0].conv_size, ker_pos, layers[0].useBias);
		ker_pos += layers[0].conv_size*layers[0].conv_size + layers[0].useBias;
	}
	if (layers[0].useReLU) layers[0].ReLU();
	if (layers[0].usePooling) layers[0].MaxPooling();
	for (int L = 1; L < layers.size(); ++L){ // цикл по слоям
		indCR = 0;
		if (layers[L - 1].usePooling){
			// цикл по сверткам
			for (int i = 0; i < layers[L].depth / layers[L - 1].depth; ++i){
				// цикл по (входным) матрицам с предыдущего слоя
				for (int j = 0; j < layers[L - 1].conv_mat.size(); ++j){
					Conv(layers[L - 1].pooling_mat[j], W, layers[L].conv_mat[indCR], layers[L].conv_size, ker_pos, layers[L].useBias);
					++indCR;
					#ifdef Debug
					if (indCR > layers[L].depth){
						throw std::runtime_error("Error: out of range indCR");
					}
					#endif
				}
				ker_pos += layers[L].conv_size*layers[L].conv_size + layers[L].useBias;
			}
		}
		else {
			// цикл по сверткам
			for (int i = 0; i < layers[L].depth / layers[L - 1].depth; ++i){
				// цикл по (входным) матрицам с предыдущего слоя
				for (int j = 0; j < layers[L - 1].conv_mat.size(); ++j){
					Conv(layers[L - 1].conv_mat[j], W, layers[L].conv_mat[indCR], layers[L].conv_size, ker_pos, layers[L].useBias);
					++indCR;
					#ifdef Debug
					if (indCR > layers[L].depth){
						throw std::runtime_error("Error: out of range indCR");
					}
					#endif
				}
				ker_pos += layers[L].conv_size*layers[L].conv_size + layers[L].useBias;
			}
		}
		if (layers[L].useReLU) layers[L].ReLU();
		if (layers[L].usePooling) layers[L].MaxPooling();
	}
}