// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include "../inst/include/miscfuns.h"

// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]
Eigen::VectorXd pnnlssum(const Eigen::MatrixXd &A, 
	const Eigen::VectorXd &b, const double &sum){
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

