#include <algorithm>

class Function{
public:
	virtual ~Function(){}
	// calculate function value for argument 'arg'
	virtual double get(const double& arg) const = 0;
	// return function derivative value of 'arg'
	virtual double d(const double& arg) const = 0;
};

class Identity : public Function{
public:
	Identity(){}

	double get(const double& arg) const {
		return arg;
	}

	double d(const double& arg) const {
		return 1.;
	}
};

class ReLU : public Function{
public:
	ReLU(){}

	double get(const double& arg) const {
		return std::max(0., arg);
	}

	double d(const double& arg) const {
		return (arg > 0);
	}
};

// Range (0,inf)
class SoftPlus : public Function{
public:
	SoftPlus(){}
	
	double get(const double& arg) const {
		return log(1.+exp(arg));
	}

	double d(const double& arg) const {
		return 1. / (1 + exp(-arg));
	}
};

