// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include "../inst/include/npfixedcomp.h"

// [[Rcpp::depends(RcppEigen)]]

class npnormcll : public npfixedcomp
{
public:

	npnormcll(const Eigen::VectorXd &data_, const Eigen::VectorXd &mu0fixed_, const Eigen::VectorXd &pi0fixed_,
		const double &beta_, const Eigen::VectorXd &initpt_, const Eigen::VectorXd &initpr_, 
		const Eigen::VectorXd &gridpoints_) : npfixedcomp(data_, mu0fixed_, pi0fixed_, beta_, initpt_, initpr_, gridpoints_){
		this->setprecompute();
		family = "npnormc";
		flag = "d0";
	}

	double lossfunction(const Eigen::VectorXd &maps) const{
		return (maps + this->precompute).array().log().sum() * -1.;
	}

	Eigen::VectorXd mapping(const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0) const{
		return dnpnormc_(this->data, mu0, pi0, this->beta);
	}

	void gradfun(const double &mu, const Eigen::VectorXd &dens,
		double &ansd0, double &ansd1, const bool &d0, const bool &d1) const{
		Eigen::VectorXd fullden = (dens + this->precompute).cwiseInverse();
		double scale = 1 - this->pi0fixed.sum();
		if (d0){
			ansd0 = (dens - dnpnormc_(this->data, Eigen::VectorXd::Constant(1, mu), Eigen::VectorXd::Constant(1, scale), this->beta)).dot(fullden);
		}

		if (d1){
			ansd1 = 0;
		}
		
		// Eigen::VectorXd temp2 = ((this->beta + 4) * std::pow(mu, 3.) -  3 * this->beta * mu * mu * this->data.array() +
		// 	mu * (2 * this->beta * this->data.array().square() + this->beta - 2) - this->beta * this->data.array() - 
		// 	2 * std::pow(mu, 5.)) / std::pow(1 - mu * mu, 3.0);
		// ansd1 = temp2.cwiseProduct(temp).dot(fullden) * scale;
	}

	void gradfunvec(const Eigen::VectorXd &mu, const Eigen::VectorXd &dens,
		Eigen::VectorXd &ansd0, Eigen::VectorXd &ansd1, const bool &d0, const bool &d1) const{
		ansd0.resize(mu.size());
		ansd1.resize(mu.size());
		Eigen::VectorXd fullden = (dens + this->precompute).cwiseInverse();
		double scale = 1 - this->pi0fixed.sum();
		if (d0){
			ansd0 = Eigen::VectorXd::Constant(mu.size(), dens.dot(fullden));
			ansd0.noalias() -= dnormcarray(this->data, mu, this->beta).transpose() * fullden * scale;
		}
	}

	void computeweights(Eigen::VectorXd &mu0, Eigen::VectorXd &pi0, 
		const Eigen::VectorXd &dens, const Eigen::VectorXd &newpoints) const{
		Eigen::VectorXd mu0new(mu0.size() + newpoints.size());
		Eigen::VectorXd pi0new(pi0.size() + newpoints.size());
		mu0new.head(mu0.size()) = mu0; mu0new.tail(newpoints.size()) = newpoints;
		pi0new.head(pi0.size()) = pi0; pi0new.tail(newpoints.size()) = Eigen::VectorXd::Zero(newpoints.size());

		sortmix(mu0new, pi0new);

		Eigen::VectorXd fp = dens + this->precompute;
		Eigen::MatrixXd sp = dnormcarray(this->data, mu0new, this->beta).array().colwise() / fp.array();
		Eigen::VectorXd nw = pnnlssum_(sp, Eigen::VectorXd::Constant(this->len, 2.) - this->precompute.cwiseQuotient(fp), 1. - this->pi0fixed.sum());
		this->checklossfun(mu0new, pi0new, nw - pi0new, sp.colwise().sum());
		this->collapse(mu0new, pi0new);

		mu0.lazyAssign(mu0new);
		pi0.lazyAssign(pi0new);
	}

	double hypofun(const double &ll, const double &minloss) const{
		return ll - minloss;
	}

	double familydensity(const double &x, const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0) const{
		return dnpnormc_(Eigen::VectorXd::Constant(1, x), mu0, pi0, this->beta)[0];
	}
};

// [[Rcpp::export]]
Rcpp::List npnormcll_(const Eigen::VectorXd &data, const Eigen::VectorXd &mu0fixed, const Eigen::VectorXd &pi0fixed,
	const double &beta, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &maxit = 100, const int &verbose = 0){
	npnormcll f(data, mu0fixed, pi0fixed, beta, initpt, initpr, gridpoints);
	f.computemixdist(tol, maxit, verbose);
	return f.result;
}

// [[Rcpp::export]]
Rcpp::List estpi0npnormcll_(const Eigen::VectorXd &data,
	const double &beta, const double &val, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &verbose = 0){
	npnormcll f(data, Eigen::VectorXd::Zero(1), Eigen::VectorXd::Ones(1), beta, initpt, initpr, gridpoints);
	f.estpi0(val, tol, verbose);
	return f.result;
}
