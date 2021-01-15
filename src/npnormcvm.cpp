// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include "../inst/include/npfixedcomp.h"

// [[Rcpp::depends(RcppEigen)]]

class npnormcvm : public npfixedcomp
{
public:
	Eigen::VectorXd precompute;

	npnormcvm(const Eigen::VectorXd &data_, const Eigen::VectorXd &mu0fixed_, const Eigen::VectorXd &pi0fixed_,
		const double &beta_, const Eigen::VectorXd &initpt_, const Eigen::VectorXd &initpr_, 
		const Eigen::VectorXd &gridpoints_) : npfixedcomp(data_, mu0fixed_, pi0fixed_, beta_, initpt_, initpr_, gridpoints_){
		// data should be sorted beforehand
		this->precompute.resize(this->len);
		this->precompute = Eigen::VectorXd::LinSpaced(this->len, 1, 2 * this->len - 1) / 2 / this->len - 
			pnpnorm_(this->data, this->mu0fixed, this->pi0fixed, this->beta);
		family = "npnorm";
		flag = "d1";
	}

	double lossfunction(const Eigen::VectorXd &maps) const{
		return (maps - this->precompute).squaredNorm();
	}

	Eigen::VectorXd mapping(const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0) const{
		return pnpnorm_(this->data, mu0, pi0, this->beta);
	}

	void gradfun(const double &mu, const Eigen::VectorXd &dens,
		double &ansd0, double &ansd1) const{
		Eigen::VectorXd fullden = dens - this->precompute;
		double scale = 1 - this->pi0fixed.sum();
		ansd0 = (pnpnorm_(this->data, Eigen::VectorXd::Constant(1, mu), Eigen::VectorXd::Constant(1, scale), this->beta) - dens).dot(fullden) * 2;
		ansd1 = dnpnorm_(this->data, Eigen::VectorXd::Constant(1, mu), Eigen::VectorXd::Ones(1), this->beta).dot(fullden) * -2 * scale;
	}

	void gradfunvec(const Eigen::VectorXd &mu, const Eigen::VectorXd &dens,
		Eigen::VectorXd &ansd0, Eigen::VectorXd &ansd1) const{
		ansd0.resize(mu.size());
		ansd1.resize(mu.size());
		Eigen::VectorXd fullden = dens - this->precompute;
		double scale = 1 - this->pi0fixed.sum();
		ansd0 = fullden.transpose() * (pnormarray(this->data, mu, this->beta) * scale - dens.rowwise().replicate(mu.size())) * 2;
		ansd1 = fullden.transpose() * dnormarray(this->data, mu, this->beta) * -2 * scale;
	}

	void computeweights(Eigen::VectorXd &mu0, Eigen::VectorXd &pi0, 
		const Eigen::VectorXd &dens, const Eigen::VectorXd &newpoints) const{
		Eigen::VectorXd mu0new(mu0.size() + newpoints.size());
		mu0new.head(mu0.size()) = mu0; mu0new.tail(newpoints.size()) = newpoints;

		Eigen::MatrixXd sp = pnormarray(this->data, mu0new, this->beta);
		Eigen::VectorXd pi0new = pnnlssum_(sp, this->precompute, 1. - this->pi0fixed.sum());

		sortmix(mu0new, pi0new);

		this->collapse(mu0new, pi0new);

		mu0.lazyAssign(mu0new);
		pi0.lazyAssign(pi0new);
	}

	double extrafun() const{
		return 1 / 12 / this->len;
	}
};

// [[Rcpp::export]]
Rcpp::List npnormcvm_(const Eigen::VectorXd &data, const Eigen::VectorXd &mu0fixed, const Eigen::VectorXd &pi0fixed,
	const double &beta, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &maxit = 100, const int &verbose = 0){
	npnormcvm f(data, mu0fixed, pi0fixed, beta, initpt, initpr, gridpoints);
	f.computemixdist(tol, maxit, verbose);
	return f.result;
}