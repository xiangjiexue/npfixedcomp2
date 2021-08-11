// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include "../inst/include/miscfuns.h"
#include "../inst/include/npfixedcomp.h"
#include "../inst/include/densityND.h"

// [[Rcpp::depends(RcppEigen)]]

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
//' One-dimensional case: \code{dnpnorm} gives the density, \code{pnpnorm} gives the distribution function.
//'
//' Multi-dimensional case: \code{dnpnormND} gives the density.
//'
//' @title non-parametric normal distribution
//' @param x vector of observations, vector of quantiles for 1D and a m-by-n matrix for n-dimensional case. 
//' @param mu0 the vector of support points for 1D and a k-by-n matrix for n-dimensional case.
//' @param stdev standard deviation for 1D and covariance matrix for multi-dimensional case.
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
//' \code{dnpt} gives the density, \code{pnpt} gives the distribution function.
//'
//' @title non-parametric t distribution
//' @param x vector of observations, vector of quantiles
//' @param mu0 the vector of support points
//' @param df degree of freedom.
//' @param pi0 the vector of weights correponding to the support points
//' @param lt logical; if TRUE, the lower probability is computed
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

//' @rdname npt
//' @export
// [[Rcpp::export]]
Eigen::VectorXd pnpt(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0, 
	const double &df, const bool &lt = true, const bool &lg = false){
	return pnpt_(x, mu0, pi0, df, lt, lg);
}

// [[Rcpp::export]]
Eigen::MatrixXd ptarray_(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0,
	const double &df, const bool &lt = true, const bool &lg = false){
	return ptarray(x, mu0, df, lt, lg);
}

//' The density and the distribution function of non-parametric discrete normal distribution
//'
//' \code{dnpdiscnorm} gives the density and \code{pnpdiscnorm} gives the CDF.
//'
//' In this implementation, the support space is ..., -h, 0, h, ...
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

//' The density and the distribution function of non-parametric poisson distribution
//'
//' \code{dnppois} gives the density, \code{pnppois} gives the distribution function.
//'
//' It is important to note that \code{pnppois} and its underlying function did not use
//' logspaceadd type implementation, the accuracy may be affected. 
//'
//' @title non-parametric normal distribution
//' @param x vector of observations, vector of quantiles
//' @param mu0 the vector of support points
//' @param stdev structure parameter, passed but not referenced, for compatibility
//' @param pi0 the vector of weights correponding to the support points
//' @param lt logical; if TRUE, the lower probability is computed
//' @param lg logical; if TRUE, the result will be given in log scale.
//' @rdname nppois
//' @export
// [[Rcpp::export]]
Eigen::VectorXd dnppois(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0, 
	const double &stdev = 1, const bool &lg = false){
	return dnppois_(x, mu0, pi0, stdev, lg);
}

// [[Rcpp::export]]
Eigen::MatrixXd dpoisarray_(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0,
	const double &stdev = 1, const bool &lg = false){
	return dpoisarray(x, mu0, stdev, lg);
}

//' @rdname nppois
//' @export
// [[Rcpp::export]]
Eigen::VectorXd pnppois(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0, 
	const double &stdev = 1, const bool &lt = true, const bool &lg = false){
	return pnppois_(x, mu0, pi0, stdev, lt, lg);
}

// [[Rcpp::export]]
Eigen::MatrixXd ppoisarray_(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0,
	const double &stdev = 1, const bool &lt = true, const bool &lg = false){
	return ppoisarray(x, mu0, stdev, lt, lg);
}

//' The density of non-parametric discrete non-central t distribution
//'
//' \code{dnpdisct} gives the density .
//'
//' In this implementation, the support space is ..., -h, 0, h, ...
//'
//' @title non-parametric discrete non-central t distribution
//' @param x vector of observations, vector of quantiles
//' @param mu0 the vector of support points
//' @param pi0 the vector of weights correponding to the support points
//' @param df the degree of freedom
//' @param h the bin width
//' @param lg logical; if TRUE, the result will be given in log scale.
//' @rdname npdisct
//' @export
// [[Rcpp::export]]
Eigen::VectorXd dnpdisct(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0, 
	const double &df, const double &h, const bool &lg = false){
	return dnpdisct_(x, mu0, pi0, df, h, lg);
}

// [[Rcpp::export]]
Eigen::MatrixXd ddisctarray_(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0,
	const double &df, const double &h, const bool &lg = false){
	return ddisctarray(x, mu0, df, h, lg);
}

// [[Rcpp::export]]
Eigen::VectorXd log1mexp_(const Eigen::VectorXd &x){
	return log1mexp(x);
}

// [[Rcpp::export]]
Eigen::VectorXd logspaceadd_(const Eigen::VectorXd &lx, const Eigen::VectorXd &ly){
	return logspaceadd(lx, ly);
}

//' @rdname npnorm
//' @export
// [[Rcpp::export]]
Eigen::VectorXd dnpnormND(const Eigen::MatrixXd &x, 
	const Eigen::MatrixXd &mu0, const Eigen::VectorXd &pi0,
	const Eigen::MatrixXd &stdev, const bool &lg = false){
	return dnpnormND_(x, mu0, pi0, stdev, lg);
}

// [[Rcpp::export]]
Eigen::MatrixXd dnormNDarray_(const Eigen::MatrixXd &x, 
	const Eigen::MatrixXd &mu0,
	const Eigen::MatrixXd &stdev, const bool &lg = false){
	return dnormNDarray(x, mu0, stdev, lg);
}