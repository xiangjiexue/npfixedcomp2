// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include "../inst/include/npfixedcomp.h"

// [[Rcpp::depends(RcppEigen)]]

class npnormad : public npfixedcomp
{
public:
	Eigen::VectorXd w1, w2;

	npnormad(const Eigen::VectorXd &data_, const Eigen::VectorXd &mu0fixed_, const Eigen::VectorXd &pi0fixed_,
		const double &beta_, const Eigen::VectorXd &initpt_, const Eigen::VectorXd &initpr_, 
		const Eigen::VectorXd &gridpoints_) : npfixedcomp(data_, mu0fixed_, pi0fixed_, beta_, initpt_, initpr_, gridpoints_){
		// data should be sorted beforehand
		this->w1.resize(this->len);
		this->w1 = Eigen::VectorXd::LinSpaced(this->len, 1., 2. * this->len - 1.);
		this->w2.resize(this->len);
		this->w2 = this->w1.reverse();
		this->setprecompute();
		family = "npnorm";
		flag = "d1";
	}

	double lossfunction(const Eigen::VectorXd &maps) const{
		Eigen::VectorXd temp = maps + this->precompute;
		return (w1.array() * temp.array().log() + w2.array() * (temp.array() * -1.).log1p()).mean() * -1.; 
	}

	Eigen::VectorXd mapping(const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0) const{
		return pnpnorm_(this->data, mu0, pi0, this->beta);
	}

	void gradfun(const double &mu, const Eigen::VectorXd &dens,
		double &ansd0, double &ansd1, const bool &d0, const bool &d1) const{
		double scale = 1 - this->pi0fixed.sum();
		Eigen::VectorXd fullden = dens + this->precompute;
		Eigen::VectorXd one = Eigen::VectorXd::Ones(this->len);
		if (d0){
			Eigen::VectorXd temp = pnpnorm_(this->data, Eigen::VectorXd::Constant(1, mu), Eigen::VectorXd::Constant(1, scale), this->beta) + this->precompute;
			ansd0 = (temp.cwiseQuotient(fullden).cwiseProduct(this->w1) + (one - temp).cwiseQuotient(one - fullden).cwiseProduct(this->w2)).mean() * -1 + 2 * this->len;
		}

		if (d1){
			ansd1 = dnpnorm_(this->data, Eigen::VectorXd::Constant(1, mu), Eigen::VectorXd::Ones(1), this->beta).cwiseProduct(this->w1.cwiseQuotient(fullden) - this->w2.cwiseQuotient(one - fullden)).mean() * scale;
		}
	}

	// void gradfunvec(const Eigen::VectorXd &mu, const Eigen::VectorXd &dens,
	// 	Eigen::VectorXd &ansd0, Eigen::VectorXd &ansd1) const{
	// 	ansd0.resize(mu.size());
	// 	ansd1.resize(mu.size());
	// 	Eigen::VectorXd fullden = dens + this->precompute;
	// 	double scale = 1 - this->pi0fixed.sum();
	// 	ansd0 = fullden.transpose() * (pnormarray(this->data, mu, this->beta) * scale - dens.rowwise().replicate(mu.size())) * 2;
	// 	ansd1 = fullden.transpose() * dnormarray(this->data, mu, this->beta) * -2 * scale;
	// }

	void computeweights(Eigen::VectorXd &mu0, Eigen::VectorXd &pi0, 
		const Eigen::VectorXd &dens, const Eigen::VectorXd &newpoints) const{
		Eigen::VectorXd mu0new(mu0.size() + newpoints.size());
		Eigen::VectorXd pi0new(pi0.size() + newpoints.size());
		mu0new.head(mu0.size()) = mu0; mu0new.tail(newpoints.size()) = newpoints;
		pi0new.head(pi0.size()) = pi0; pi0new.tail(newpoints.size()) = Eigen::VectorXd::Zero(newpoints.size());
		
		sortmix(mu0new, pi0new);

		Eigen::MatrixXd sf = (pnormarray(this->data, mu0new, this->beta) * (1. - this->pi0fixed.sum())).array().colwise() + this->precompute.array();
		Eigen::VectorXd sp = dens + this->precompute;
		Eigen::MatrixXd S = sf.array().colwise() / sp.array();
		Eigen::MatrixXd U = (Eigen::MatrixXd::Ones(this->len, mu0new.size()) - sf).array().colwise() / (Eigen::VectorXd::Ones(this->len) - sp).array();
		Eigen::VectorXd S2 = S.transpose() * this->w1 + U.transpose() * this->w2;

		Eigen::VectorXd nw = pnnqp_(S.transpose() * this->w1.asDiagonal() * S + U.transpose() * this->w2.asDiagonal() * U,
			-2. * S2, 1.) * (1. - this->pi0fixed.sum());

		this->checklossfun(mu0new, pi0new, nw - pi0new, S2 / (1. - pi0fixed.sum()));
		this->collapse(mu0new, pi0new);

		mu0.lazyAssign(mu0new);
		pi0.lazyAssign(pi0new);
	}

	double extrafun() const{
		return -this->len;
	}
};

// [[Rcpp::export]]
Rcpp::List npnormad_(const Eigen::VectorXd &data, const Eigen::VectorXd &mu0fixed, const Eigen::VectorXd &pi0fixed,
	const double &beta, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &maxit = 100, const int &verbose = 0){
	npnormad f(data, mu0fixed, pi0fixed, beta, initpt, initpr, gridpoints);
	f.computemixdist(tol, maxit, verbose);
	return f.result;
}