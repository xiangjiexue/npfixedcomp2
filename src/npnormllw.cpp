// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include "../inst/include/npfixedcomp.h"

// [[Rcpp::depends(RcppEigen)]]

class npnormllw : public npfixedcomp
{
public:
	Eigen::VectorXd precompute;
	Eigen::VectorXd weights;
	double h;

	npnormllw(const Eigen::VectorXd &data_, const Eigen::VectorXd &weights_, const Eigen::VectorXd &mu0fixed_, const Eigen::VectorXd &pi0fixed_,
		const double &beta_, const double &h_, const Eigen::VectorXd &initpt_, const Eigen::VectorXd &initpr_, 
		const Eigen::VectorXd &gridpoints_) : npfixedcomp(data_, mu0fixed_, pi0fixed_, beta_, initpt_, initpr_, gridpoints_){
		this->weights.resize(this->len);
		this->weights = weights_;
		this->h = h_;
		this->precompute.resize(this->len);
		this->precompute = dnpdiscnorm_(this->data, this->mu0fixed, this->pi0fixed, this->beta, this->h);
		this->family = "npnorm";
		this->flag = "d0";
	}

	double lossfunction(const Eigen::VectorXd &maps) const{
		return ((maps + this->precompute).array().log() * this->weights.array()).sum() * -1.;
	}

	Eigen::VectorXd mapping(const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0) const{
		return dnpdiscnorm_(this->data, mu0, pi0, this->beta, this->h);
	}

	void gradfun(const double &mu, const Eigen::VectorXd &dens,
		double &ansd0, double &ansd1) const{
		Eigen::VectorXd fullden = (dens + this->precompute).cwiseInverse();
		double scale = 1 - this->pi0fixed.sum();
		Eigen::VectorXd temp = dnpdiscnorm_(this->data, Eigen::VectorXd::Constant(1, mu), Eigen::VectorXd::Constant(1, scale), this->beta, this->h);
		ansd0 = (dens - temp).cwiseProduct(this->weights).dot(fullden);
		ansd1 = 0;
	}

	// void gradfunvec(const Eigen::VectorXd &mu, const Eigen::VectorXd &dens,
	// 	Eigen::VectorXd &ansd0, Eigen::VectorXd &ansd1) const{
	// 	ansd0.resize(mu.size());
	// 	ansd1.resize(mu.size());
	// 	Eigen::VectorXd fullden = (dens + this->precompute).cwiseInverse();
	// 	double scale = 1 - this->pi0fixed.sum();
	// 	Eigen::MatrixXd temp = dnormarray(this->data, mu, this->beta);
	// 	ansd0 = fullden.transpose() * (dens.rowwise().replicate(mu.size()) - temp * scale);
	// 	ansd1 = fullden.transpose() * (mu.transpose().colwise().replicate(this->len) - this->data.rowwise().replicate(mu.size())).cwiseProduct(temp)  / (this->beta * this->beta) * scale;
	// }

	void computeweights(Eigen::VectorXd &mu0, Eigen::VectorXd &pi0, 
		const Eigen::VectorXd &dens, const Eigen::VectorXd &newpoints) const{
		Eigen::VectorXd mu0new(mu0.size() + newpoints.size());
		Eigen::VectorXd pi0new(pi0.size() + newpoints.size());
		mu0new.head(mu0.size()) = mu0; mu0new.tail(newpoints.size()) = newpoints;
		pi0new.head(pi0.size()) = pi0; pi0new.tail(newpoints.size()) = Eigen::VectorXd::Zero(newpoints.size());

		sortmix(mu0new, pi0new);

		Eigen::MatrixXd sp = ddiscnormarray(this->data, mu0new, this->beta, this->h);
		Eigen::VectorXd fp = dens + this->precompute;
		sp = sp.array().colwise() / fp.array();
		Eigen::VectorXd nw = pnnlssum_(sp.array().colwise() * this->weights.array().sqrt(), 
			(2. - this->precompute.array() / fp.array()) * this->weights.array().sqrt(), 1. - this->pi0fixed.sum());
		this->checklossfun(mu0new, pi0new, nw - pi0new, sp.transpose() * this->weights);
		this->collapse(mu0new, pi0new);

		mu0.lazyAssign(mu0new);
		pi0.lazyAssign(pi0new);
	}

	void print(const int &level = 0) const{
		Rcpp::Rcout<<"mu0fixed: "<<this->mu0fixed.transpose()<<std::endl;
		Rcpp::Rcout<<"pi0fixed: "<<this->pi0fixed.transpose()<<std::endl;
		Rcpp::Rcout<<"beta: "<<this->beta<<std::endl;
		Rcpp::Rcout<<"length: "<<this->len<<std::endl;
		Rcpp::Rcout<<"gridpoints: "<<this->gridpoints.transpose()<<std::endl;
		Rcpp::Rcout<<"initial loss: "<<this->lossfunction(this->mapping(initpt, initpr))<<std::endl;

		if (level == 1){
			Rcpp::Rcout<<"data: "<<this->data.transpose()<<std::endl;
			Rcpp::Rcout<<"precompute: "<<this->precompute.transpose()<<std::endl;
		}
	}

	double extrafun() const{
		return this->weights.sum() * std::log(this->h);
	}
};

// [[Rcpp::export]]
Rcpp::List npnormllw_(const Eigen::VectorXd &data, const Eigen::VectorXd &weights, const Eigen::VectorXd &mu0fixed, const Eigen::VectorXd &pi0fixed,
	const double &beta, const double &h, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &maxit = 100, const bool &verbose = false){
	npnormllw f(data, weights, mu0fixed, pi0fixed, beta, h, initpt, initpr, gridpoints);
	f.computemixdist(tol, maxit, verbose);
	return f.result;
}