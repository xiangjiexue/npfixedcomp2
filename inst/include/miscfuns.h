#ifndef miscfuns_h
#define miscfuns_h
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>
#include <unsupported/Eigen/SpecialFunctions>

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


// Simple implementaion of density of normal mixture.
// if mu0 a scalar
inline Eigen::VectorXd dnormarraylogs(const Eigen::VectorXd &x, const double &mu0, const double &stdev){
	return (x.array() - mu0).square() / (-2 * stdev * stdev) - M_LN_SQRT_2PI - std::log(stdev);
}

inline Eigen::MatrixXd dnormarraylog(const Eigen::VectorXd &x,
	const Eigen::VectorXd &mu0, const double& stdev){
	return (mu0.transpose().colwise().replicate(x.size()).colwise() - x).array().square() / (-2 * stdev * stdev) - M_LN_SQRT_2PI - std::log(stdev);
}

inline Eigen::MatrixXd dnormarray(const Eigen::VectorXd &x,
	const Eigen::VectorXd &mu0, const double& stdev, const bool& lg = false){
	if (lg){
		return dnormarraylog(x, mu0, stdev);
	}else{
		return dnormarraylog(x, mu0, stdev).array().exp();
	}
}

inline Eigen::VectorXd dnpnormorigin(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0, const double &stdev = 1){
	if (mu0.size() == 1){
		Eigen::VectorXd ans = dnormarraylogs(x, mu0[0], stdev).array().exp();
		if (pi0[0] == 1) {return ans;} else {return ans * pi0[0];}
	}else{
		return dnormarray(x, mu0, stdev) * pi0;	
	}
}

inline Eigen::VectorXd dnpnorm_(const Eigen::VectorXd &x,
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0,
	const double& stdev, const bool& lg = false){
	if (lg){
		return dnpnormorigin(x, mu0, pi0, stdev).array().log();
	}else{
		return dnpnormorigin(x, mu0, pi0, stdev);
	}
}

// Simple implementaion of CDF of normal mixture.
struct pnormptr{
	pnormptr(const double &mu_, const double &stdev_, const bool& lt_, const bool& lg_ = false) : mu(mu_), stdev(stdev_), lt(lt_), lg(lg_) {};

	const double operator()(const double & x) const {
		return R::pnorm5(x, mu, stdev, lt, lg);
	}

	double mu, stdev;
	bool lt, lg;
};

inline Eigen::MatrixXd pnormarray(const Eigen::VectorXd &x,
	const Eigen::VectorXd &mu0, const double& stdev, 
	const bool &lt = true, const bool &lg = false){
	Eigen::MatrixXd ans(x.size(), mu0.size());
	for (auto i = mu0.size() - 1; i >= 0; i--){
		std::transform(x.data(), x.data() + x.size(), ans.col(i).data(), pnormptr(mu0[i], stdev, lt, lg));
	}
	return ans;
}

inline Eigen::VectorXd pnpnormorigin(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0, 
	const double &stdev = 1, const bool &lt = true){
	if (mu0.size() == 1){
		Eigen::ArrayXd temp = (x.array() - mu0[0]) / stdev / std::sqrt(2);
		if (pi0[0] == 1) {
			if (lt){
				return 0.5 * (1 + temp.erf());
			}else{
				return 0.5 * temp.erfc();
			}
		} else {
			if (lt){
				return 0.5 * (1 + temp.erf()) * pi0[0];
			}else{
				return 0.5 * temp.erfc() * pi0[0];
			}
		}
	}else{
		Eigen::ArrayXXd temp = (mu0.transpose().colwise().replicate(x.size()).colwise() - x) / stdev / -std::sqrt(2); 
		if (lt){
			return (0.5 * (1 + temp.erf())).matrix() * pi0;
		}else{
			return (0.5 * temp.erfc()).matrix() * pi0;
		}
	}
}

inline Eigen::VectorXd pnpnorm_(const Eigen::VectorXd &x,
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0,
	const double& stdev, const bool &lt = true, const bool& lg = false){
	if (lg){
		return pnpnormorigin(x, mu0, pi0, stdev, lt).array().log();
	}else{
		return pnpnormorigin(x, mu0, pi0, stdev, lt);
	}
}

// Simple implementaion of density of one-parameter normal mixture.
inline Eigen::MatrixXd dnormcarraylog(const Eigen::VectorXd &x, const Eigen::VectorXd &mu0, const double &n){
	Eigen::MatrixXd temp = mu0.transpose().colwise().replicate(x.size());
	Eigen::MatrixXd stdev = (1 - temp.array().square()) / std::sqrt(n);
	return ((temp.colwise() - x).array() / stdev.array()).square() * -.5 - M_LN_SQRT_2PI - stdev.array().log();
}

inline Eigen::VectorXd dnormcarraylogs(const Eigen::VectorXd &x, const double &mu0, const double &n){
	return ((x.array() - mu0) * std::sqrt(n) / (1 - mu0 * mu0)).square() * -.5 - M_LN_SQRT_2PI - std::log((1 - mu0 * mu0) / std::sqrt(n));
}

inline Eigen::MatrixXd dnormcarray(const Eigen::VectorXd &x,
	const Eigen::VectorXd &mu0, const double& n, const bool& lg = false){
	if (lg){
		return dnormcarraylog(x, mu0, n);
	}else{
		return dnormcarraylog(x, mu0, n).array().exp();
	}
}

inline Eigen::VectorXd dnpnormcorigin(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0, const double &n){
	if (mu0.size() == 1){
		Eigen::VectorXd ans = dnormcarraylogs(x, mu0[0], n).array().exp();
		if (pi0[0] == 1) {return ans;} else {return ans * pi0[0];}
	}else{
		return dnormcarray(x, mu0, n) * pi0;	
	}
}

inline Eigen::VectorXd dnpnormc_(const Eigen::VectorXd &x,
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0,
	const double& n, const bool& lg = false){
	if (lg){
		return dnpnormcorigin(x, mu0, pi0, n).array().log();
	}else{
		return dnpnormcorigin(x, mu0, pi0, n);
	}
}

// Simple implementaion of density of discete normal.
// inline Eigen::MatrixXd ddiscnormarray(const Eigen::VectorXd &x,
// 	const Eigen::VectorXd &mu0, const double &stdev, const double &h, const double &lg = false){
// 	Eigen::MatrixXd lx = pnormarray(x, mu0 - Eigen::VectorXd::Constant(mu0.size(), h), stdev, true, true);
// 	Eigen::MatrixXd ly = pnormarray(x, mu0, stdev, true, true);
// 	Eigen::MatrixXd ans(x.size(), mu0.size());
// 	std::transform(lx.data(), lx.data() + lx.size(), ly.data(), ans.data(), 
// 		[](const double &x, const double &y){return R::logspace_sub(x, y);});
// 	if (lg){
// 		return ans;
// 	}else{
// 		return ans.array().exp();
// 	}
// }

inline Eigen::MatrixXd ddiscnormarray(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0,
	const double &stdev, const double &h, const bool &lg = false){
	// Trapezoid rule. Relative error <= 1e-6 / 12;
	int N = std::max(std::ceil(std::max(x.maxCoeff() - mu0.minCoeff(), mu0.maxCoeff() - x.minCoeff()) * 1e3 * std::pow(h, 1.5)), 5.);
	double delta = h / N;
	Eigen::MatrixXd ans(x.size(), mu0.size());
	ans.noalias() = (dnormarray(x, mu0, stdev) + dnormarray(x, mu0 - Eigen::VectorXd::Constant(mu0.size(), h), stdev)) * 0.5;
	for (int i = 1; i < N; i++){
		ans.noalias() += dnormarray(x, mu0 - Eigen::VectorXd::Constant(mu0.size(), delta * i), stdev);
	}
	if (lg){
		return (ans * delta).array().log();
	}else{
		return ans * delta;
	}
}

inline Eigen::VectorXd dnpdiscnormorigin(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0, const double &stdev, const double &h){
	return ddiscnormarray(x, mu0, stdev, h) * pi0;
}

inline Eigen::VectorXd dnpdiscnorm_(const Eigen::VectorXd &x,
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0,
	const double& stdev, const double &h, const bool& lg = false){
	if (lg){
		return dnpdiscnormorigin(x, mu0, pi0, stdev, h).array().log();
	}else{
		return dnpdiscnormorigin(x, mu0, pi0, stdev, h);
	}
}

// Simple implementaion of CDF of discete normal.
inline Eigen::MatrixXd pdiscnormarray(const Eigen::VectorXd &x,
	const Eigen::VectorXd &mu0, const double &stdev, const double &h, const bool &lt = true, const bool &lg = false){
	if (x.size() > mu0.size()){
		return pnormarray(x, mu0 - Eigen::VectorXd::Constant(mu0.size(), h), stdev, lt, lg);
	}else{
		return pnormarray(x + Eigen::VectorXd::Constant(x.size(), h), mu0, stdev, lt, lg);
	}
}

inline Eigen::VectorXd pnpdiscnorm_(const Eigen::VectorXd &x,
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0,
	const double& stdev, const double &h, const bool &lt = true, const bool& lg = false){
	if (lg){
		return pnpnormorigin(x, mu0 - Eigen::VectorXd::Constant(mu0.size(), h), pi0, stdev, lt).array().log();
	}else{
		return pnpnormorigin(x, mu0 - Eigen::VectorXd::Constant(mu0.size(), h), pi0, stdev, lt);
	}
}

// Simple implementaion of density of t mixture.
struct dtptr{
	dtptr(const double &mu_, const double &n_) : mu(mu_), n(n_) {};

	const double operator()(const double & x) const {
		return R::dnt(x, n, mu, false);
	}

	double mu, n;
};

inline Eigen::MatrixXd dtarrayorigin(const Eigen::VectorXd &x,
	const Eigen::VectorXd &mu0, const double& n){
	Eigen::MatrixXd ans(x.size(), mu0.size());
	for (auto i = mu0.size() - 1; i >= 0; i--){
		std::transform(x.data(), x.data() + x.size(), ans.col(i).data(), dtptr(mu0[i], n));
	}
	return ans;
}

inline Eigen::MatrixXd dtarray(const Eigen::VectorXd &x,
	const Eigen::VectorXd &mu0, const double& n, const bool& lg = false){
	if (lg){
		return dtarrayorigin(x, mu0, n).array().log();
	}else{
		return dtarrayorigin(x, mu0, n);
	}
}

inline Eigen::VectorXd dnptorigin(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0, const double &n){
	if (mu0.size() == 1){
		Eigen::VectorXd ans(x.size());
		std::transform(x.data(), x.data() + x.size(), ans.data(), dtptr(mu0[0], n));
		if (pi0[0] == 1) {return ans;} else {return ans * pi0[0];}
	}else{
		Eigen::MatrixXd ans(x.size(), mu0.size());
		for (auto i = mu0.size() - 1; i >= 0; i--){
			std::transform(x.data(), x.data() + x.size(), ans.col(i).data(), dtptr(mu0[i], n));
		}
		return ans * pi0;	
	}
}

inline Eigen::VectorXd dnpt_(const Eigen::VectorXd &x,
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0,
	const double& n, const bool& lg = false){
	if (lg){
		return dnptorigin(x, mu0, pi0, n).array().log();
	}else{
		return dnptorigin(x, mu0, pi0, n);
	}
}

// class for comparison
// sort mixing distribution
class comparemu0
{
private:
	Eigen::VectorXd mu0;
public:
	comparemu0(const Eigen::VectorXd &mu0_) : mu0(mu0_) {};

	const bool operator()(const int & x, const int & y) const {return mu0[x] < mu0[y];};
};

inline void sort1(Eigen::VectorXd &x){
	std::sort(x.data(), x.data() + x.size(), std::less<double>());
}

inline void sort2(Eigen::VectorXd &x){
	std::sort(x.data(), x.data() + x.size(), std::greater<double>());
}

#endif