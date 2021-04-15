// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include "../inst/include/npfixedcomp.h"

// [[Rcpp::depends(RcppEigen)]]

class nptll : public npfixedcomp
{
public:

	nptll(const Eigen::VectorXd &data_, const Eigen::VectorXd &mu0fixed_, const Eigen::VectorXd &pi0fixed_,
		const double &beta_, const Eigen::VectorXd &initpt_, const Eigen::VectorXd &initpr_, 
		const Eigen::VectorXd &gridpoints_) : npfixedcomp(data_, mu0fixed_, pi0fixed_, beta_, initpt_, initpr_, gridpoints_){
		this->setprecompute();
		this->family = "npt";
		this->flag = "d0";
	}

	double lossfunction(const Eigen::VectorXd &maps) const{
		return (maps + this->precompute).array().log().sum() * -1.;
	}

	Eigen::VectorXd mapping(const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0) const{
		return dnpt_(this->data, mu0, pi0, this->beta);
	}

	void gradfun(const double &mu, const Eigen::VectorXd &dens,
		double &ansd0, double &ansd1, const bool &d0, const bool &d1) const{
		Eigen::VectorXd fullden = (dens + this->precompute).cwiseInverse();
		double scale = 1 - this->pi0fixed.sum();
		if (d0){
			ansd0 = (dens - dnpt_(this->data, mu, scale, this->beta)).dot(fullden);
		}
	}

	void gradfunvec(const Eigen::VectorXd &mu, const Eigen::VectorXd &dens,
		Eigen::VectorXd &ansd0, Eigen::VectorXd &ansd1, const bool &d0, const bool &d1) const{
		ansd0.resize(mu.size());
		ansd1.resize(mu.size());
		Eigen::VectorXd fullden = (dens + this->precompute).cwiseInverse();
		double scale = 1 - this->pi0fixed.sum();
		if (d0){
			ansd0 = Eigen::VectorXd::Constant(mu.size(), dens.dot(fullden));
			ansd0.noalias() -= dtarray(this->data, mu, this->beta).transpose() * fullden * scale;
		}
	}

	void computeweights(const Eigen::VectorXd &mu0, Eigen::VectorXd &pi0, const Eigen::VectorXd &dens) const{
		Eigen::VectorXd fp = dens + this->precompute;
		Eigen::MatrixXd sp = dtarray(this->data, mu0, this->beta), tp = sp.array().colwise() / fp.array();
		Eigen::VectorXd nw(pi0.size());

		if (this->len > 1e3){
			nw = pnnqp_(tp.transpose() * tp, tp.transpose() * (this->precompute.cwiseQuotient(fp) - Eigen::VectorXd::Constant(this->len, 2)), 1. - this->pi0fixed.sum());
		}else{
			nw = pnnlssum_(tp, Eigen::VectorXd::Constant(this->len, 2) - this->precompute.cwiseQuotient(fp), 1. - this->pi0fixed.sum());
		}
		this->checklossfun2(sp * nw - dens, pi0, nw - pi0, tp.colwise().sum(), dens);
	}

	double hypofun(const double &ll, const double &minloss) const{
		return ll - minloss;
	}

	double familydensity(const double &x, const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0) const{
		return dnpt_(Eigen::VectorXd::Constant(1, x), mu0, pi0, this->beta).coeff(0);
	}
};

// [[Rcpp::export]]
Rcpp::List nptll_(const Eigen::VectorXd &data, const Eigen::VectorXd &mu0fixed, const Eigen::VectorXd &pi0fixed,
	const double &beta, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &maxit = 100, const int &verbose = 0){
	nptll f(data, mu0fixed, pi0fixed, beta, initpt, initpr, gridpoints);
	f.computemixdist(tol, maxit, verbose);
	return f.result;
}

// [[Rcpp::export]]
Rcpp::List estpi0nptll_(const Eigen::VectorXd &data,
	const double &beta, const double &val, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &verbose = 0){
	nptll f(data, Eigen::VectorXd::Zero(1), Eigen::VectorXd::Ones(1), beta, initpt, initpr, gridpoints);
	f.estpi0(val, tol, verbose);
	return f.result;
}

// large-scale computation

class nptllw : public npfixedcomp
{
public:
	Eigen::VectorXd weights;
	double h;

	nptllw(const Eigen::VectorXd &data_, const Eigen::VectorXd &weights_, const Eigen::VectorXd &mu0fixed_, const Eigen::VectorXd &pi0fixed_,
		const double &beta_, const double &h_, const Eigen::VectorXd &initpt_, const Eigen::VectorXd &initpr_, 
		const Eigen::VectorXd &gridpoints_) : npfixedcomp(data_, mu0fixed_, pi0fixed_, beta_, initpt_, initpr_, gridpoints_){
		this->weights.resize(this->len);
		this->weights = weights_;
		this->h = h_;
		this->setprecompute();
		this->family = "npt";
		this->flag = "d0";
	}

	double lossfunction(const Eigen::VectorXd &maps) const{
		return ((maps + this->precompute).array().log() * this->weights.array()).sum() * -1.;
	}

	Eigen::VectorXd mapping(const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0) const{
		return dnpdisct_(this->data, mu0, pi0, this->beta, this->h);
	}

	void gradfun(const double &mu, const Eigen::VectorXd &dens,
		double &ansd0, double &ansd1, const bool &d0, const bool &d1) const{
		Eigen::VectorXd fullden = (dens + this->precompute).cwiseInverse();
		double scale = 1 - this->pi0fixed.sum();
		Eigen::VectorXd temp = dnpdisct_(this->data, mu, scale, this->beta, this->h);
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
			ansd0 = Eigen::VectorXd::Constant(mu.size(), dens.dot(fullden));
			ansd0.noalias() -= ddisctarray(this->data, mu, this->beta, this->h).transpose() * fullden * scale;
		}
	}

	void computeweights(const Eigen::VectorXd &mu0, Eigen::VectorXd &pi0, const Eigen::VectorXd &dens) const{
		Eigen::VectorXd fp = dens + this->precompute;
		Eigen::MatrixXd sp = ddisctarray(this->data, mu0, this->beta, this->h), tp = sp.array().colwise() / fp.array();
		Eigen::VectorXd nw = pnnlssum_(tp.cwiseProduct(this->weights.cwiseSqrt().replicate(1, mu0.size())), 
			(Eigen::VectorXd::Constant(this->len, 2) - this->precompute.cwiseQuotient(fp)).cwiseProduct(this->weights.cwiseSqrt()), 1. - this->pi0fixed.sum());
		this->checklossfun2(sp * nw - dens, pi0, nw - pi0, tp.transpose() * this->weights, dens);
		// this->checklossfun(mu0, pi0, nw - pi0, tp.transpose() * this->weights);
	}

	double extrafun() const{
		// the likelihood value for each fit will not be accurate but the likelihood ratio should still be accurate.
		return this->weights.sum() * std::log(this->h);
	}

	double hypofun(const double &ll, const double &minloss) const{
		return ll - minloss;
	}

	double familydensity(const double &x, const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0) const{
		return dnpdisct_(Eigen::VectorXd::Constant(1, x), mu0, pi0, this->beta, this->h).coeff(0);
	}
};

// [[Rcpp::export]]
Rcpp::List nptllw_(const Eigen::VectorXd &data, const Eigen::VectorXd &weights, const Eigen::VectorXd &mu0fixed, const Eigen::VectorXd &pi0fixed,
	const double &beta, const double &h, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &maxit = 100, const int &verbose = 0){
	nptllw f(data, weights, mu0fixed, pi0fixed, beta, h, initpt, initpr, gridpoints);
	f.computemixdist(tol, maxit, verbose);
	return f.result;
}

// [[Rcpp::export]]
Rcpp::List estpi0nptllw_(const Eigen::VectorXd &data, const Eigen::VectorXd &weights,
	const double &beta, const double &h, const double &val, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &verbose = 0){
	nptllw f(data, weights, Eigen::VectorXd::Zero(1), Eigen::VectorXd::Ones(1), beta, h, initpt, initpr, gridpoints);
	f.estpi0(val, tol, verbose);
	return f.result;
}