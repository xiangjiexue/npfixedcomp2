// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include "../inst/include/npfixedcomp.h"

// [[Rcpp::depends(RcppEigen)]]

class npnormll : public npfixedcomp
{
public:
	Eigen::VectorXd precompute;

	npnormll(const Eigen::VectorXd &data_, const Eigen::VectorXd &mu0fixed_, const Eigen::VectorXd &pi0fixed_,
		const double &beta_, const Eigen::VectorXd &initpt_, const Eigen::VectorXd &initpr_, 
		const Eigen::VectorXd &gridpoints_) : npfixedcomp(data_, mu0fixed_, pi0fixed_, beta_, initpt_, initpr_, gridpoints_){
		this->precompute.resize(this->len);
		this->precompute = dnpnorm_(this->data, this->mu0fixed, this->pi0fixed, this->beta);
		this->family = "npnorm";
		this->flag = "d1";
	}

	double lossfunction(const Eigen::VectorXd &maps) const{
		return (maps + this->precompute).array().log().sum() * -1.;
	}

	Eigen::VectorXd mapping(const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0) const{
		return dnpnorm_(this->data, mu0, pi0, this->beta);
	}

	void gradfun(const double &mu, const Eigen::VectorXd &dens,
		double &ansd0, double &ansd1) const{
		Eigen::VectorXd fullden = (dens + this->precompute).cwiseInverse();
		double scale = 1 - this->pi0fixed.sum();
		Eigen::VectorXd temp = dnpnorm_(this->data, Eigen::VectorXd::Constant(1, mu), Eigen::VectorXd::Constant(1, scale), this->beta).cwiseProduct(fullden);
		ansd0 = dens.dot(fullden) - temp.sum(); // (dens - temp).dot(fullden);
		// ansd1 = (Eigen::VectorXd::Constant(this->len, mu) - this->data).cwiseProduct(temp).dot(fullden) / (this->beta * this->beta);
		ansd1 = (temp.sum() * mu - this->data.dot(temp)) / (this->beta * this->beta);
	}

	void gradfunvec(const Eigen::VectorXd &mu, const Eigen::VectorXd &dens,
		Eigen::VectorXd &ansd0, Eigen::VectorXd &ansd1) const{
		ansd0.resize(mu.size());
		ansd1.resize(mu.size());
		Eigen::VectorXd fullden = (dens + this->precompute).cwiseInverse();
		double scale = 1 - this->pi0fixed.sum();
		Eigen::MatrixXd temp = dnormarray(this->data, mu, this->beta);
		ansd0 = Eigen::VectorXd::Constant(mu.size(), dens.dot(fullden)) - temp.transpose() * fullden * scale;
		ansd1 = fullden.transpose() * (mu.transpose().colwise().replicate(this->len) - this->data.rowwise().replicate(mu.size())).cwiseProduct(temp)  / (this->beta * this->beta) * scale;
	}

	void computeweights(Eigen::VectorXd &mu0, Eigen::VectorXd &pi0, 
		const Eigen::VectorXd &dens, const Eigen::VectorXd &newpoints) const{
		Eigen::VectorXd mu0new(mu0.size() + newpoints.size());
		Eigen::VectorXd pi0new(pi0.size() + newpoints.size());
		mu0new.head(mu0.size()) = mu0; mu0new.tail(newpoints.size()) = newpoints;
		pi0new.head(pi0.size()) = pi0; pi0new.tail(newpoints.size()) = Eigen::VectorXd::Zero(newpoints.size());

		sortmix(mu0new, pi0new);

		Eigen::MatrixXd sp = dnormarray(this->data, mu0new, this->beta);
		Eigen::VectorXd fp = dens + this->precompute;
		sp = sp.array().colwise() / fp.array();
		Eigen::VectorXd nw = pnnlssum_(sp, Eigen::VectorXd::Constant(this->len, 2.) - this->precompute.cwiseQuotient(fp), 1. - this->pi0fixed.sum());
		this->checklossfun(mu0new, pi0new, nw - pi0new, sp.colwise().sum());
		this->collapse(mu0new, pi0new);

		mu0.lazyAssign(mu0new);
		pi0.lazyAssign(pi0new);
	}
};

// [[Rcpp::export]]
Rcpp::List npnormll_(const Eigen::VectorXd &data, const Eigen::VectorXd &mu0fixed, const Eigen::VectorXd &pi0fixed,
	const double &beta, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &maxit = 100, const int &verbose = 0){
	npnormll f(data, mu0fixed, pi0fixed, beta, initpt, initpr, gridpoints);
	f.computemixdist(tol, maxit, verbose);
	return f.result;
}