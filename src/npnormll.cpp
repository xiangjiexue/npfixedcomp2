// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include "../inst/include/npfixedcomp.h"

// [[Rcpp::depends(RcppEigen)]]

class npnormll : public npfixedcomp
{
public:

	npnormll(const Eigen::VectorXd &data_, const Eigen::VectorXd &mu0fixed_, const Eigen::VectorXd &pi0fixed_,
		const double &beta_, const Eigen::VectorXd &initpt_, const Eigen::VectorXd &initpr_, 
		const Eigen::VectorXd &gridpoints_) : npfixedcomp(data_, mu0fixed_, pi0fixed_, beta_, initpt_, initpr_, gridpoints_){
		this->setprecompute();
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
		double &ansd0, double &ansd1, const bool &d0, const bool &d1) const{
		Eigen::VectorXd fullden = (dens + this->precompute).cwiseInverse();
		double scale = 1 - this->pi0fixed.sum();
		Eigen::VectorXd temp = dnpnorm_(this->data, Eigen::VectorXd::Constant(1, mu), Eigen::VectorXd::Constant(1, scale), this->beta).cwiseProduct(fullden);
		if (d0){
			ansd0 = dens.dot(fullden) - temp.sum(); // (dens - temp).dot(fullden);
		}

		if (d1){
		// ansd1 = (Eigen::VectorXd::Constant(this->len, mu) - this->data).cwiseProduct(temp).dot(fullden) / (this->beta * this->beta);
			ansd1 = (temp.sum() * mu - this->data.dot(temp)) / (this->beta * this->beta);
		}
	}

	void gradfunvec(const Eigen::VectorXd &mu, const Eigen::VectorXd &dens,
		Eigen::VectorXd &ansd0, Eigen::VectorXd &ansd1, const bool &d0, const bool &d1) const{
		ansd0.resize(mu.size());
		ansd1.resize(mu.size());
		Eigen::VectorXd fullden = (dens + this->precompute).cwiseInverse();
		double scale = 1 - this->pi0fixed.sum();
		Eigen::MatrixXd temp = dnormarray(this->data, mu, this->beta);
		if (d0){
			ansd0 = Eigen::VectorXd::Constant(mu.size(), dens.dot(fullden)) - temp.transpose() * fullden * scale;
		}

		if (d1){
			ansd1 = fullden.transpose() * (mu.transpose().colwise().replicate(this->len) - this->data.rowwise().replicate(mu.size())).cwiseProduct(temp)  / (this->beta * this->beta) * scale;
		}
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

	double hypofun(const double &ll, const double &minloss) const{
		return ll - minloss;
	}

	double familydensity(const double &x, const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0) const{
		return dnpnorm_(Eigen::VectorXd::Constant(1, x), mu0, pi0, this->beta)[0];
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

// [[Rcpp::export]]
Rcpp::List estpi0npnormll_(const Eigen::VectorXd &data,
	const double &beta, const double &val, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &verbose = 0){
	npnormll f(data, Eigen::VectorXd::Zero(1), Eigen::VectorXd::Ones(1), beta, initpt, initpr, gridpoints);
	f.estpi0(val, tol, verbose);
	return f.result;
}

// large-scale computation

class npnormllw : public npfixedcomp
{
public:
	Eigen::VectorXd weights;
	double h;

	npnormllw(const Eigen::VectorXd &data_, const Eigen::VectorXd &weights_, const Eigen::VectorXd &mu0fixed_, const Eigen::VectorXd &pi0fixed_,
		const double &beta_, const double &h_, const Eigen::VectorXd &initpt_, const Eigen::VectorXd &initpr_, 
		const Eigen::VectorXd &gridpoints_) : npfixedcomp(data_, mu0fixed_, pi0fixed_, beta_, initpt_, initpr_, gridpoints_){
		this->weights.resize(this->len);
		this->weights = weights_;
		this->h = h_;
		this->setprecompute();
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
		double &ansd0, double &ansd1, const bool &d0, const bool &d1) const{
		Eigen::VectorXd fullden = (dens + this->precompute).cwiseInverse();
		double scale = 1 - this->pi0fixed.sum();
		Eigen::VectorXd temp = dnpdiscnorm_(this->data, Eigen::VectorXd::Constant(1, mu), Eigen::VectorXd::Constant(1, scale), this->beta, this->h);
		if (d0){
			ansd0 = (dens - temp).cwiseProduct(this->weights).dot(fullden);
		}
	}

	void gradfunvec(const Eigen::VectorXd &mu, const Eigen::VectorXd &dens,
		Eigen::VectorXd &ansd0, Eigen::VectorXd &ansd1, const bool &d0, const bool &d1) const{
		ansd0.resize(mu.size());
		ansd1.resize(mu.size());
		Eigen::VectorXd fullden = this->weights.cwiseQuotient(dens + this->precompute);
		double scale = 1 - this->pi0fixed.sum();
		if (d0){
			ansd0 = Eigen::VectorXd::Constant(mu.size(), dens.dot(fullden)) - ddiscnormarray(this->data, mu, this->beta, this->h).transpose() * fullden * scale;
		}
	}

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

	double extrafun() const{
		return this->weights.sum() * std::log(this->h);
	}

	double hypofun(const double &ll, const double &minloss) const{
		return ll - minloss;
	}

	double familydensity(const double &x, const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0) const{
		return dnpdiscnorm_(Eigen::VectorXd::Constant(1, x), mu0, pi0, this->beta, this->h)[0];
	}
};

// [[Rcpp::export]]
Rcpp::List npnormllw_(const Eigen::VectorXd &data, const Eigen::VectorXd &weights, const Eigen::VectorXd &mu0fixed, const Eigen::VectorXd &pi0fixed,
	const double &beta, const double &h, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &maxit = 100, const int &verbose = 0){
	npnormllw f(data, weights, mu0fixed, pi0fixed, beta, h, initpt, initpr, gridpoints);
	f.computemixdist(tol, maxit, verbose);
	return f.result;
}

// [[Rcpp::export]]
Rcpp::List estpi0npnormllw_(const Eigen::VectorXd &data, const Eigen::VectorXd &weights,
	const double &beta, const double &h, const double &val, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &verbose = 0){
	npnormllw f(data, weights, Eigen::VectorXd::Zero(1), Eigen::VectorXd::Ones(1), beta, h, initpt, initpr, gridpoints);
	f.estpi0(val, tol, verbose);
	return f.result;
}