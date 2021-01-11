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
		Eigen::VectorXd ans(x.size());
		std::transform(x.data(), x.data() + x.size(), ans.data(), pnormptr(mu0[0], stdev, lt));
		if (pi0[0] == 1) {return ans;} else {return ans * pi0[0];}
	}else{
		Eigen::MatrixXd ans(x.size(), mu0.size());
		for (auto i = mu0.size() - 1; i >= 0; i--){
			std::transform(x.data(), x.data() + x.size(), ans.col(i).data(), pnormptr(mu0[i], stdev, lt));
		}
		return ans * pi0;
	}
}

inline Eigen::VectorXd pnpnormlog(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0, 
	const double &stdev = 1, const bool &lt = true){
	Eigen::VectorXd ans(x.size());
	if (mu0.size() == 1){
		std::transform(x.data(), x.data() + x.size(), ans.data(), pnormptr(mu0[0], stdev, lt, true));
		if (pi0[0] != 1) {ans = ans + Eigen::VectorXd::Constant(x.size(), std::log1p(pi0[0] - 1));}
	}else{
		Eigen::MatrixXd temp = pnormarray(x, mu0, stdev, lt, true);
		ans = temp.col(0) + Eigen::VectorXd::Constant(x.size(), std::log1p(pi0[0] - 1));
		for (int i = 1; i < mu0.size(); i++){
			ans = (ans + log1pexp(temp.col(i) + Eigen::VectorXd::Constant(x.size(), std::log1p(pi0[i] - 1)) - ans)).eval();
		}
	}
	return ans;
}

inline Eigen::VectorXd pnpnorm_(const Eigen::VectorXd &x,
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0,
	const double& stdev, const bool &lt = true, const bool& lg = false){
	if (lg){
		return pnpnormlog(x, mu0, pi0, stdev, lt);
	}else{
		return pnpnormorigin(x, mu0, pi0, stdev, lt);
	}
}

// Simple implementaion of density of one-parameter normal mixture.
struct dnormcptr{
	dnormcptr(const double &mu_, const double &n_) : mu(mu_), n(n_) {};

	const double operator()(const double & x) const {
		return R::dnorm4(x, mu, (1 - mu * mu) / std::sqrt(n), false);
	}

	double mu, n;
};

inline Eigen::MatrixXd dnormcarrayorigin(const Eigen::VectorXd &x,
	const Eigen::VectorXd &mu0, const double& n){
	Eigen::MatrixXd ans(x.size(), mu0.size());
	for (auto i = mu0.size() - 1; i >= 0; i--){
		std::transform(x.data(), x.data() + x.size(), ans.col(i).data(), dnormcptr(mu0[i], n));
	}
	return ans;
}

inline Eigen::MatrixXd dnormcarray(const Eigen::VectorXd &x,
	const Eigen::VectorXd &mu0, const double& n, const bool& lg = false){
	if (lg){
		return dnormcarrayorigin(x, mu0, n).array().log();
	}else{
		return dnormcarrayorigin(x, mu0, n);
	}
}

inline Eigen::VectorXd dnpnormcorigin(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0, const double &n){
	if (mu0.size() == 1){
		Eigen::VectorXd ans(x.size());
		std::transform(x.data(), x.data() + x.size(), ans.data(), dnormcptr(mu0[0], n));
		if (pi0[0] == 1) {return ans;} else {return ans * pi0[0];}
	}else{
		Eigen::MatrixXd ans(x.size(), mu0.size());
		for (auto i = mu0.size() - 1; i >= 0; i--){
			std::transform(x.data(), x.data() + x.size(), ans.col(i).data(), dnormcptr(mu0[i], n));
		}
		return ans * pi0;	
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