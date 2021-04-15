// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include "../inst/include/npfixedcomp.h"

// [[Rcpp::depends(RcppEigen)]]

class nppoisll : public npfixedcomp
{
public:
	Eigen::VectorXd weights;

	nppoisll(const Eigen::VectorXd &data_, const Eigen::VectorXd &weights_, const Eigen::VectorXd &mu0fixed_, const Eigen::VectorXd &pi0fixed_,
		const double &beta_, const Eigen::VectorXd &initpt_, const Eigen::VectorXd &initpr_, 
		const Eigen::VectorXd &gridpoints_) : npfixedcomp(data_, mu0fixed_, pi0fixed_, beta_, initpt_, initpr_, gridpoints_){
		this->setprecompute();
		this->weights.resize(this->len);
		this->weights = weights_;
		family = "nppois";
		flag = "d0";
	}

	double lossfunction(const Eigen::VectorXd &maps) const{
		return ((maps + this->precompute).array().log() * this->weights.array()).sum() * -1.;
	}

	Eigen::VectorXd mapping(const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0) const{
		return dnppois_(this->data, mu0, pi0, this->beta);
	}

	void gradfun(const double &mu, const Eigen::VectorXd &dens,
		double &ansd0, double &ansd1, const bool &d0, const bool &d1) const{
		Eigen::VectorXd fullden = this->weights.cwiseQuotient(dens + this->precompute);
		double scale = 1 - this->pi0fixed.sum();
		if (d0){
			ansd0 = (dens - dnppois_(this->data, mu, scale, this->beta)).dot(fullden);
		}
	}

	void gradfunvec(const Eigen::VectorXd &mu, const Eigen::VectorXd &dens,
		Eigen::VectorXd &ansd0, Eigen::VectorXd &ansd1, const bool &d0, const bool &d1) const{
		ansd0.resize(mu.size());
		ansd1.resize(mu.size());
		Eigen::VectorXd fullden = this->weights.cwiseQuotient(dens + this->precompute);
		double scale = 1 - this->pi0fixed.sum();
		if (d0){
			ansd0 = Eigen::VectorXd::Constant(mu.size(), dens.dot(fullden));
			ansd0.noalias() -= dpoisarray(this->data, mu, this->beta).transpose() * fullden * scale;
		}
	}

	void computeweights(const Eigen::VectorXd &mu0, Eigen::VectorXd &pi0, const Eigen::VectorXd &dens) const{
		Eigen::VectorXd fp = dens + this->precompute;
		Eigen::MatrixXd sp = dpoisarray(this->data, mu0, this->beta), tp = sp.array().colwise() / fp.array();

		Eigen::VectorXd nw = pnnlssum_(tp.cwiseProduct(this->weights.cwiseSqrt().replicate(1, mu0.size())), 
			(Eigen::VectorXd::Constant(this->len, 2) - this->precompute.cwiseQuotient(fp)).cwiseProduct(this->weights.cwiseSqrt()), 1. - this->pi0fixed.sum());
		this->checklossfun2(sp * nw - dens, pi0, nw - pi0, tp.transpose() * this->weights, dens);
	}

	double hypofun(const double &ll, const double &minloss) const{
		return ll - minloss;
	}

	double familydensity(const double &x, const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0) const{
		return dnppois_(Eigen::VectorXd::Constant(1, x), mu0, pi0, this->beta).coeff(0);
	}
};

// [[Rcpp::export]]
Rcpp::List nppoisll_(const Eigen::VectorXd &data, const Eigen::VectorXd &weights, const Eigen::VectorXd &mu0fixed, const Eigen::VectorXd &pi0fixed,
	const double &beta, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &maxit = 100, const int &verbose = 0){
	nppoisll f(data, weights, mu0fixed, pi0fixed, beta, initpt, initpr, gridpoints);
	f.computemixdist(tol, maxit, verbose);
	return f.result;
}

// [[Rcpp::export]]
Rcpp::List estpi0nppoisll_(const Eigen::VectorXd &data, const Eigen::VectorXd &weights,
	const double &beta, const double &val, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &verbose = 0){
	nppoisll f(data, weights, Eigen::VectorXd::Zero(1), Eigen::VectorXd::Ones(1), beta, initpt, initpr, gridpoints);
	f.estpi0(val, tol, verbose);
	return f.result;
}
