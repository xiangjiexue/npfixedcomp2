// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include "../inst/include/miscfuns.h"

// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]
Eigen::MatrixXd matsubset(const Eigen::MatrixXd &A, const Eigen::VectorXi &index){
	// This function should only be used before Eigen 3.4.
	// In Eigen 3.4 there is a built-in function for slicing.
	int L = (index.array() > 0).count();
	Eigen::MatrixXd temp(index.size(), L);
	int j = 0;
	for (int i = 0; i < index.size(); i++){
		if (index[i] > 0){
			temp.col(j) = A.col(i);
			j++;
		} 
	}
	Eigen::MatrixXd ans(L, L);
	j = 0;
	for (auto i = 0; i < index.size(); i++){
		if (index[i] > 0){
			ans.row(j) = temp.row(i);
			j++;
		}
	}
	return ans;
}

// [[Rcpp::export]]
Eigen::VectorXd vecsubset(const Eigen::VectorXd &b, const Eigen::VectorXi &index){
	// This function should only be used before Eigen 3.4.
	// In Eigen 3.4 there is a built-in function for slicing.
	int L = (index.array() > 0).count();
	Eigen::VectorXd ans(L);
	int j = 0;
	for (int i = 0; i < index.size(); i++){
		if (index[i] > 0){
			ans[j] = b[i];
			j++;
		} 
	}
	return ans;
}

void vecsubassign(Eigen::VectorXd &x, const Eigen::VectorXd &y, const Eigen::VectorXi &index){
	// y is of size index.sum();
	// This function should only be used before Eigen 3.4.
	// In Eigen 3.4 there is a built-in function for slicing.
	int j = 0;
	for (int i = 0; i < x.size(); i++){
		if (index[i] > 0){
			x[i] = y[j];
			j++;
		}
	}
}

Eigen::VectorXd nnls(const Eigen::MatrixXd &A, const Eigen::VectorXd &b){
	// fnnls by Bros and Jong (1997)
	int n = A.cols();
	Eigen::VectorXi p = Eigen::VectorXi::Zero(n);
	Eigen::VectorXd x = Eigen::VectorXd::Zero(n), ZX = A.transpose() * b, s(n), one = Eigen::VectorXd::Ones(n);
	Eigen::MatrixXd ZZ = A.transpose() * A;
	Eigen::VectorXd w = ZX - ZZ * x;
	int maxind, iter = 0;
	double alpha, tol = std::max(ZX.array().abs().maxCoeff(), ZZ.array().abs().maxCoeff()) * 10 * std::numeric_limits<double>::epsilon();
	while ((p.array() == 0).count() > 0 & (p.array() == 0).select(w, -1 * one).maxCoeff(&maxind) > tol & iter < 3 * n){
		p[maxind] = 1;
		s.setZero();
		vecsubassign(s, matsubset(ZZ, p).llt().solve(vecsubset(ZX, p)), p);
		while ((p.array() > 0).select(s, one).minCoeff() <= 0){
			alpha = vecsubset(x.cwiseQuotient(x - s), ((p.array() > 0).select(s, one).array() <= 0).cast<int>()).minCoeff();
			x.noalias() += alpha * (s - x);
			p.array() *= (x.array() > 0).cast<int>();
			s.setZero();
			vecsubassign(s, matsubset(ZZ, p).llt().solve(vecsubset(ZX, p)), p);
		}
		x = s;
		w = ZX - ZZ * x;
		iter++;
	}
	return x;
}

// [[Rcpp::export]]
Eigen::VectorXd pnnlssum(const Eigen::MatrixXd &A, const Eigen::VectorXd &b, const double &sum){
	return pnnlssum_(A, b, sum);
}


// [[Rcpp::export]]
Eigen::VectorXd pnnqp(const Eigen::MatrixXd &q, const Eigen::VectorXd &p, const double &sum){
	return pnnqp_(q, p, sum);
}

//' The density and the distribution function of non-parametric normal distribution
//'
//' \code{dnpnorm} gives the density, \code{pnpnorm} gives the distribution function.
//'
//' @title non-parametric normal distribution
//' @param x vector of observations, vector of quantiles
//' @param mu0 the vector of support points
//' @param stdev standard deviation.
//' @param pi0 the vector of weights correponding to the support points
//' @param lt logical; if TRUE, the lower probability is computed
//' @param lg logical; if TRUE, the result will be given in log scale.
//' @rdname npnorm
//' @export
// [[Rcpp::export]]
Eigen::VectorXd dnpnorm(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0, 
	const double &stdev = 1, const bool &lg = false){
	return dnpnorm_(x, mu0, pi0, stdev, lg);
}

// [[Rcpp::export]]
Eigen::MatrixXd dnormarray_(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0,
	const double &stdev = 1, const bool &lg = false){
	return dnormarray(x, mu0, stdev, lg);
}

//' @rdname npnorm
//' @export
// [[Rcpp::export]]
Eigen::VectorXd pnpnorm(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0, 
	const double &stdev = 1, const bool &lt = true, const bool &lg = false){
	return pnpnorm_(x, mu0, pi0, stdev, lt, lg);
}

// [[Rcpp::export]]
Eigen::MatrixXd pnormarray_(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0,
	const double &stdev = 1, const bool &lt = true, const bool &lg = false){
	return pnormarray(x, mu0, stdev, lt, lg);
}

//' The density and the distribution function of non-parametric one-parameter normal distribution
//'
//' \code{dnpnormc} gives the density
//'
//' @title non-parametric one-parameter normal distribution
//' @param x vector of observations, vector of quantiles
//' @param mu0 the vector of support points
//' @param n number of observations (not the length of x)
//' @param pi0 the vector of weights correponding to the support points
//' @param lg logical; if TRUE, the result will be given in log scale.
//' @rdname npnormc
//' @export
// [[Rcpp::export]]
Eigen::VectorXd dnpnormc(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0, 
	const double &n, const bool &lg = false){
	return dnpnormc_(x, mu0, pi0, n, lg);
}

// [[Rcpp::export]]
Eigen::MatrixXd dnormcarray_(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0,
	const double &n, const bool &lg = false){
	return dnormcarray(x, mu0, n, lg);
}

//' The density and the distribution function of non-parametric t distribution
//'
//' \code{dnpnorm} gives the density.
//'
//' @title non-parametric t distribution
//' @param x vector of observations, vector of quantiles
//' @param mu0 the vector of support points
//' @param df degree of freedom.
//' @param pi0 the vector of weights correponding to the support points
//' @param lg logical; if TRUE, the result will be given in log scale.
//' @rdname npt
//' @export
// [[Rcpp::export]]
Eigen::VectorXd dnpt(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0, 
	const double &df, const bool &lg = false){
	return dnpt_(x, mu0, pi0, df, lg);
}

// [[Rcpp::export]]
Eigen::MatrixXd dtarray_(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0,
	const double &df, const bool &lg = false){
	return dtarray(x, mu0, df, lg);
}

//' The density and the distribution function of non-parametric discrete normal distribution
//'
//' \code{dnpdiscnorm} gives the density and \code{pnpdiscnorm} gives the CDF
//'
//' @title non-parametric discrete normal distribution
//' @param x vector of observations, vector of quantiles
//' @param mu0 the vector of support points
//' @param pi0 the vector of weights correponding to the support points
//' @param stdev standard deviation
//' @param h the bin width
//' @param lt logical; if TRUE, the result will be given in lower tail.
//' @param lg logical; if TRUE, the result will be given in log scale.
//' @rdname npdiscnorm
//' @export
// [[Rcpp::export]]
Eigen::VectorXd dnpdiscnorm(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0, 
	const double &stdev, const double &h, const bool &lg = false){
	return dnpdiscnorm_(x, mu0, pi0, stdev, h, lg);
}

// [[Rcpp::export]]
Eigen::MatrixXd ddiscnormarray_(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0,
	const double &stdev, const double &h, const bool &lg = false){
	return ddiscnormarray(x, mu0, stdev, h, lg);
}

//' @rdname npdiscnorm
//' @export
// [[Rcpp::export]]
Eigen::VectorXd pnpdiscnorm(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0, 
	const double &stdev, const double &h, const bool &lt = true, const bool &lg = false){
	return pnpdiscnorm_(x, mu0, pi0, stdev, h, lt, lg);
}

// [[Rcpp::export]]
Eigen::MatrixXd pdiscnormarray_(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0,
	const double &stdev, const double &h, const bool &lt = true, const bool &lg = false){
	return pdiscnormarray(x, mu0, stdev, h, lt, lg);
}

