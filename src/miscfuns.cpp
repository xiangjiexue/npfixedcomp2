// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include "../inst/include/miscfuns.h"

// [[Rcpp::depends(RcppEigen)]]

//' @export
// [[Rcpp::export]]
Eigen::VectorXd pnnlssum(const Eigen::MatrixXd &A, 
	const Eigen::VectorXd &b, const double &sum){
	return pnnlssum_(A, b, sum);
}

//' @export
// [[Rcpp::export]]
Eigen::VectorXd pnnqp(const Eigen::MatrixXd &q, const Eigen::VectorXd &p, const double &sum){
	return pnnqp_(q, p, sum);
}

//' @export
// [[Rcpp::export]]
Eigen::VectorXd diff(const Eigen::VectorXd &x){
	return diff_(x);
}

//' @export
// [[Rcpp::export]]
Eigen::VectorXd dnpnorm(const Eigen::VectorXd &x, 
	const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0, 
	const double &stdev = 1, const bool &lg = false){
	return dnpnorm_(x, mu0, pi0, stdev, lg);
}