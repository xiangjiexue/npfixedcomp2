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
			Eigen::VectorXd temp = pnpnorm_(this->data, mu, scale, this->beta) + this->precompute;
			ansd0 = (temp.cwiseQuotient(fullden).cwiseProduct(this->w1) + (one - temp).cwiseQuotient(one - fullden).cwiseProduct(this->w2)).mean() * -1 + 2 * this->len;
		}

		if (d1){
			ansd1 = dnpnorm_(this->data, mu, 1., this->beta).cwiseProduct(this->w1.cwiseQuotient(fullden) - this->w2.cwiseQuotient(one - fullden)).mean() * scale;
		}
	}

	void gradfunvec(const Eigen::VectorXd &mu, const Eigen::VectorXd &dens,
		Eigen::VectorXd &ansd0, Eigen::VectorXd &ansd1, const bool & d0, const bool &d1) const{
		ansd0.resize(mu.size());
		ansd1.resize(mu.size());
		Eigen::VectorXd fullden = dens + this->precompute;
		double scale = 1 - this->pi0fixed.sum();
		if (d0){
			Eigen::ArrayXXd temp = (pnormarray(this->data, mu, this->beta) * scale).colwise() + this->precompute;
			ansd0 = (temp.colwise() * this->w1.cwiseQuotient(fullden).array() + (1 - temp).colwise() * (this->w2.array() / (1 - fullden.array()))).colwise().mean() * -1 + 2 * this->len; 
		} 

		if (d1){
			ansd1 = dnormarray(this->data, mu, this->beta).transpose() * (this->w1.cwiseQuotient(fullden) - this->w2.cwiseQuotient(Eigen::VectorXd::Ones(this->len) - fullden)) * scale / this->len;
		}
	}

	void computeweights(const Eigen::VectorXd &mu0, Eigen::VectorXd &pi0, const Eigen::VectorXd &dens) const{
		Eigen::MatrixXd sf = (pnormarray(this->data, mu0, this->beta) * (1. - this->pi0fixed.sum())).array().colwise() + this->precompute.array();
		Eigen::VectorXd sp = dens + this->precompute;
		Eigen::MatrixXd S = sf.array().colwise() / sp.array();
		Eigen::MatrixXd U = (1. - sf.array()).colwise() / (1. - sp.array());
		Eigen::VectorXd S2 = S.transpose() * this->w1 + U.transpose() * this->w2;

		Eigen::VectorXd nw = pnnqp_(S.transpose() * this->w1.asDiagonal() * S + U.transpose() * this->w2.asDiagonal() * U,
			-2. * S2, 1.) * (1. - this->pi0fixed.sum());
		this->checklossfun2(sf * nw - sp, pi0, nw - pi0, S2 / (1. - pi0fixed.sum()), sp);
	}

	double extrafun() const{
		return -this->len;
	}

	double hypofun(const double &ll, const double &minloss) const{
		return ll;
	}

	double familydensity(const double &x, const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0) const{
		return dnpnorm_(Eigen::VectorXd::Constant(1, x), mu0, pi0, this->beta).coeff(0);
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

// [[Rcpp::export]]
Rcpp::List estpi0npnormad_(const Eigen::VectorXd &data,
	const double &beta, const double &val, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &verbose = 0){
	npnormad f(data, Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(1), beta, initpt, initpr, gridpoints);
	f.estpi0(val, tol, verbose);
	return f.result;
}

// large-scale computation

class npnormadw : public npfixedcomp
{
public:
	Eigen::VectorXd weights, w1, w2;
	double h;

	npnormadw(const Eigen::VectorXd &data_, const Eigen::VectorXd &weights_, const Eigen::VectorXd &mu0fixed_, const Eigen::VectorXd &pi0fixed_,
		const double &beta_, const double &h_, const Eigen::VectorXd &initpt_, const Eigen::VectorXd &initpr_, 
		const Eigen::VectorXd &gridpoints_) : npfixedcomp(data_, mu0fixed_, pi0fixed_, beta_, initpt_, initpr_, gridpoints_){
		// data should be sorted beforehand
		this->weights.resize(this->len);
		this->weights = weights_;
		this->h = h_;
		this->setprecompute();
		Eigen::VectorXd ecdf(weights);
		std::partial_sum(ecdf.data(), ecdf.data() + ecdf.size(), ecdf.data(), std::plus<double>());
		this->w1.resize(this->len);
		this->w1 = (2 * ecdf - this->weights).cwiseProduct(this->weights);
		this->w2.resize(this->len);
		this->w2 = 2 * this->weights.sum() * this->weights - this->w1;
		family = "npnorm";
		flag = "d1";
	}

	double lossfunction(const Eigen::VectorXd &maps) const{
		Eigen::VectorXd temp = maps + this->precompute;
		return (w1.array() * temp.array().log() + w2.array() * (temp.array() * -1.).log1p()).sum() / this->weights.sum() * -1.; 
	}

	Eigen::VectorXd mapping(const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0) const{
		return pnpdiscnorm_(this->data, mu0, pi0, this->beta, this->h);
	}

	void gradfun(const double &mu, const Eigen::VectorXd &dens,
		double &ansd0, double &ansd1, const bool &d0, const bool &d1) const{
		double scale = 1 - this->pi0fixed.sum();
		Eigen::VectorXd fullden = dens + this->precompute;
		Eigen::VectorXd one = Eigen::VectorXd::Ones(this->len);
		if (d0){
			Eigen::VectorXd temp = pnpdiscnorm_(this->data, mu, scale, this->beta, this->h) + this->precompute;
			ansd0 = (temp.cwiseQuotient(fullden).cwiseProduct(this->w1) + (one - temp).cwiseQuotient(one - fullden).cwiseProduct(this->w2)).sum() / this->weights.sum() * -1 + 2 * this->weights.sum();
		}

		if (d1){
			ansd1 = dnormarray(this->data, mu - h, this->beta).cwiseProduct(this->w1.cwiseQuotient(fullden) - this->w2.cwiseQuotient(one - fullden)).sum() / this->weights.sum() * scale;
		} 
	}

	void gradfunvec(const Eigen::VectorXd &mu, const Eigen::VectorXd &dens,
		Eigen::VectorXd &ansd0, Eigen::VectorXd &ansd1, const bool &d0, const bool &d1) const{
		ansd0.resize(mu.size());
		ansd1.resize(mu.size());
		Eigen::VectorXd fullden = dens + this->precompute;
		double scale = 1 - this->pi0fixed.sum();
		if (d0){
			Eigen::ArrayXXd temp = (pdiscnormarray(this->data, mu, this->beta, this->h) * scale).colwise() + this->precompute;
			ansd0 = (temp.colwise() * this->w1.cwiseQuotient(fullden).array() + (1 - temp).colwise() * (this->w2.array() / (1 - fullden.array()))).colwise().sum() / this->weights.sum() * -1 + 2 * this->weights.sum(); 
		} 

		if (d1){
			ansd1 = dnormarray(this->data, mu - Eigen::VectorXd::Constant(mu.size(), h), this->beta).transpose() * (this->w1.cwiseQuotient(fullden) - this->w2.cwiseQuotient(Eigen::VectorXd::Ones(this->len) - fullden)) * scale / this->weights.sum();
		}
	}

	void computeweights(const Eigen::VectorXd &mu0, Eigen::VectorXd &pi0, const Eigen::VectorXd &dens) const{
		Eigen::MatrixXd sf = (pdiscnormarray(this->data, mu0, this->beta, this->h) * (1. - this->pi0fixed.sum())).array().colwise() + this->precompute.array();
		Eigen::VectorXd sp = dens + this->precompute;
		Eigen::MatrixXd S = sf.array().colwise() / sp.array();
		Eigen::MatrixXd U = (1. - sf.array()).colwise() / (1. - sp.array());
		Eigen::VectorXd S2 = S.transpose() * this->w1 + U.transpose() * this->w2;

		Eigen::VectorXd nw = pnnqp_(S.transpose() * this->w1.asDiagonal() * S + U.transpose() * this->w2.asDiagonal() * U,
			-2. * S2, 1.) * (1. - this->pi0fixed.sum());

		this->checklossfun2(sf * nw - sp, pi0, nw - pi0, S2 / (1. - pi0fixed.sum()), sp);
		// this->checklossfun(mu0, pi0, nw - pi0, S2 / (1. - pi0fixed.sum()));

	}

	double extrafun() const{
		return -this->weights.sum();
	}

	double hypofun(const double &ll, const double &minloss) const{
		return ll;
	}

	double familydensity(const double &x, const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0) const{
		return dnpdiscnorm_(Eigen::VectorXd::Constant(1, x), mu0, pi0, this->beta, this->h).coeff(0);
	}
};

// [[Rcpp::export]]
Rcpp::List npnormadw_(const Eigen::VectorXd &data, const Eigen::VectorXd &weights, const Eigen::VectorXd &mu0fixed, const Eigen::VectorXd &pi0fixed,
	const double &beta, const double &h, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &maxit = 100, const int &verbose = 0){
	npnormadw f(data, weights, mu0fixed, pi0fixed, beta, h, initpt, initpr, gridpoints);
	f.computemixdist(tol, maxit, verbose);
	return f.result;
}

// [[Rcpp::export]]
Rcpp::List estpi0npnormadw_(const Eigen::VectorXd &data, const Eigen::VectorXd &weights,
	const double &beta, const double &h, const double &val, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &verbose = 0){
	npnormadw f(data, weights, Eigen::VectorXd::Zero(1), Eigen::VectorXd::Ones(1), beta, h, initpt, initpr, gridpoints);
	f.estpi0(val, tol, verbose);
	return f.result;
}