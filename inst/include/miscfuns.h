#ifndef miscfuns_h
#define miscfuns_h

// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>

extern "C" void pnnls_(double* A, int* MDA, int* M, int* N, double* B, double* X, double* RNORM, double* W, double* ZZ, int* INDEX, int* MODE, int* K);

// The program NNLS
inline Eigen::VectorXd pnnlssum_(const Eigen::MatrixXd &A, 
	const Eigen::VectorXd &b, const double &sum){
	int m = A.rows(), n = A.cols();
	Eigen::MatrixXd AA(m+1, n);
	AA.topRows(m) = (A * sum).colwise() - b;
	AA.bottomRows(1) = Eigen::RowVectorXd::Ones(n);
	Eigen::VectorXd x(n), bb(m + 1), w(n), zz(m + 1);
	double rnorm;
	bb.head(m) = Eigen::VectorXd::Zero(m);
	bb.tail(1) = Eigen::VectorXd::Ones(1);
	Eigen::VectorXi index(n);
	int mode, k = 0;
	m++;
	pnnls_(AA.data(), &m, &m, &n, bb.data(), x.data(), &rnorm, w.data(), zz.data(), index.data(), &mode, &k);
	x = x / x.sum() * sum;
	return x;
}

// The program pnnqp using LLT
inline Eigen::VectorXd pnnqp_(const Eigen::MatrixXd &q, const Eigen::VectorXd &p, const double &sum){
	Eigen::LLT<Eigen::MatrixXd> llt;
	llt.compute(q);
	return pnnlssum_(llt.matrixU(), llt.matrixL().solve(-p), sum);
}

// diff function
inline Eigen::VectorXd diff_(const Eigen::VectorXd & x){
	return x.tail(x.size() - 1) - x.head(x.size() - 1);
}

inline Eigen::VectorXd log1pexp(const Eigen::VectorXd &x){
	return x.unaryExpr([](const double &y){
		return (y <= -37) ? std::exp(y) : ((y <= 18) ?  std::log1p(std::exp(y)) : ((y <= 33.3) ? y + std::exp(-y) : y));
	});
}


// Simple implementaion of density of normal mixture.
struct dnormptr{
	dnormptr(const double &mu_, const double &stdev_) : mu(mu_), stdev(stdev_) {};

	const double operator()(const double & x) const {
		return R::dnorm4(x, mu, stdev, false);
	}

	double mu, stdev;
};

inline Eigen::MatrixXd dnormarrayorigin(const Eigen::VectorXd &x,
	const Eigen::VectorXd &mu0, const double& stdev){
	Eigen::MatrixXd ans(x.size(), mu0.size());
	for (auto i = mu0.size() - 1; i >= 0; i--){
		std::transform(x.data(), x.data() + x.size(), ans.col(i).data(), dnormptr(mu0[i], stdev));
	}
	return ans;
}

inline Eigen::MatrixXd dnormarraylog(const Eigen::VectorXd &x,
	const Eigen::VectorXd &mu0, const double& stdev){
	return (x * Eigen::RowVectorXd::Ones(mu0.size()) - Eigen::VectorXd::Ones(x.size()) * mu0.transpose()).array().square() / (-2 * stdev * stdev) - M_LN_SQRT_2PI - std::log(stdev);
}

inline Eigen::MatrixXd dnormarray(const Eigen::VectorXd &x,
	const Eigen::VectorXd &mu0, const double& stdev, const bool& lg = false){
	if (lg){
		return dnormarraylog(x, mu0, stdev);
	}else{
		return dnormarrayorigin(x, mu0, stdev);
	}
}

inline Eigen::VectorXd dnpnormorigin(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0, const double &stdev = 1){
	if (mu0.size() == 1){
		Eigen::VectorXd ans(x.size());
		std::transform(x.data(), x.data() + x.size(), ans.data(), dnormptr(mu0[0], stdev));
		if (pi0[0] == 1) {return ans;} else {return ans * pi0[0];}
	}else{
		Eigen::MatrixXd ans(x.size(), mu0.size());
		for (auto i = mu0.size() - 1; i >= 0; i--){
			std::transform(x.data(), x.data() + x.size(), ans.col(i).data(), dnormptr(mu0[i], stdev));
		}
		return ans * pi0;	
	}
}

inline Eigen::VectorXd dnpnormlog(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0, const double &stdev = 1){
	Eigen::VectorXd ans(x.size());
	if (mu0.size() == 1){
		ans = (x.array() - mu0[0]).square() / (-2 * stdev * stdev) - M_LN_SQRT_2PI - std::log(stdev);
		if (pi0[0] != 1) ans = ans + Eigen::VectorXd::Constant(x.size(), std::log(pi0[0]));
	}else{
		Eigen::MatrixXd temp = dnormarraylog(x, mu0, stdev);
		ans = temp.col(0) + Eigen::VectorXd::Constant(x.size(), std::log(pi0[0]));
		for (int i = 1; i < mu0.size(); i++){
			ans = (ans + log1pexp(temp.col(i) + Eigen::VectorXd::Constant(x.size(), std::log(pi0[i])) - ans)).eval();
		}
	}
	return ans;
}

inline Eigen::VectorXd dnpnorm_(const Eigen::VectorXd &x,
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0,
	const double& stdev, const bool& lg = false){
	if (lg){
		return dnpnormlog(x, mu0, pi0, stdev);
	}else{
		return dnpnormorigin(x, mu0, pi0, stdev);
	}
}

#endif