// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include "../inst/include/npfixedcomp.h"

// [[Rcpp::depends(RcppEigen)]]

template<class Type>
class npnormad : public npfixedcomp<Type>
{
public:
	Eigen::Matrix<Type, Eigen::Dynamic, 1> w1, w2;

	npnormad(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &data_, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &mu0fixed_, 
		const Eigen::Matrix<Type, Eigen::Dynamic, 1> &pi0fixed_, const Type &beta_, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &initpt_, 
		const Eigen::Matrix<Type, Eigen::Dynamic, 1> &initpr_, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &gridpoints_) : npfixedcomp<Type>(data_, mu0fixed_, pi0fixed_, beta_, initpt_, initpr_, gridpoints_){
		// data should be sorted beforehand
		this->w1.resize(this->len);
		this->w1 = Eigen::Matrix<Type, Eigen::Dynamic, 1>::LinSpaced(this->len, 1., 2. * this->len - 1.) / this->len;
		this->w2.resize(this->len);
		this->w2 = this->w1.reverse();
		this->setprecompute();
		this->family = "npnorm";
		this->flag = "d1";
	}

	Type lossfunction(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &maps) const{
		Eigen::Matrix<Type, Eigen::Dynamic, 1> temp = maps + this->precompute;
		return (w1.array() * temp.array().log() + w2.array() * (temp.array() * -1.).log1p()).sum() * -1.; 
	}

	Eigen::Matrix<Type, Eigen::Dynamic, 1> mapping(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &mu0, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &pi0) const{
		return pnpnorm_(this->data, mu0, pi0, this->beta);
	}

	void gradfun(const Type &mu, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &dens,
		Type &ansd0, Type &ansd1, const bool &d0, const bool &d1) const{
		Type scale = 1. - this->pi0fixed.sum();
		Eigen::Matrix<Type, Eigen::Dynamic, 1> fullden = dens + this->precompute;
		Eigen::Matrix<Type, Eigen::Dynamic, 1> one = Eigen::Matrix<Type, Eigen::Dynamic, 1>::Ones(this->len);
		Eigen::Matrix<Type, Eigen::Dynamic, 1> s1 = this->w1.cwiseQuotient(fullden) - this->w2.cwiseQuotient(one - fullden);
		if (d0){
			ansd0 = (s1.dot(pnpnorm_(this->data, mu, scale, this->beta) + this->precompute) + this->w2.cwiseQuotient(one - fullden).sum()) * -1. + 2 * this->len;
		}

		if (d1){
			ansd1 = dnpnorm_(this->data, mu, scale, this->beta).dot(s1);
		}
	}

	void gradfunvec(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &mu, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &dens,
		Eigen::Matrix<Type, Eigen::Dynamic, 1> &ansd0, Eigen::Matrix<Type, Eigen::Dynamic, 1> &ansd1, const bool & d0, const bool &d1) const{
		ansd0.resize(mu.size());
		ansd1.resize(mu.size());
		Eigen::Matrix<Type, Eigen::Dynamic, 1> fullden = dens + this->precompute;
		Eigen::Matrix<Type, Eigen::Dynamic, 1> one = Eigen::Matrix<Type, Eigen::Dynamic, 1>::Ones(this->len), s1 = this->w1.cwiseQuotient(fullden) - this->w2.cwiseQuotient(one - fullden);
		Type scale = 1. - this->pi0fixed.sum();
		if (d0){
			// Eigen::ArrayXXd temp = (pnormarray(this->data, mu, this->beta) * scale).colwise() + this->precompute;
			// ansd0 = (temp.colwise() * this->w1.cwiseQuotient(fullden).array() + (1 - temp).colwise() * (this->w2.array() / (1 - fullden.array()))).colwise().mean() * -1 + 2 * this->len; 
			ansd0 = (((pnormarray(this->data, mu, this->beta) * scale).colwise() + this->precompute).transpose() * s1 +
							Eigen::Matrix<Type, Eigen::Dynamic, 1>::Constant(mu.size(), this->w2.cwiseQuotient(one - fullden).sum())) * -1. +
							2 * Eigen::Matrix<Type, Eigen::Dynamic, 1>::Constant(mu.size(), this->len);
		} 

		if (d1){
			ansd1 = dnormarray(this->data, mu, this->beta).transpose() * s1 * scale;
		}
	}

	void computeweights(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &mu0, Eigen::Matrix<Type, Eigen::Dynamic, 1> &pi0, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &dens) const{
		Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> sf = pnormarray(this->data, mu0, this->beta);
		Eigen::Matrix<Type, Eigen::Dynamic, 1> sp = dens + this->precompute;
		Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> S = sf.array().colwise() / sp.array();
		Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> U = sf.array().colwise() / (sp.array() - 1.);
		Eigen::Matrix<Type, Eigen::Dynamic, 1> S2 = S.transpose() * this->w1 + U.transpose() * this->w2;

		Eigen::Matrix<Type, Eigen::Dynamic, 1> nw = pnnqp_(S.transpose() * this->w1.asDiagonal() * S + U.transpose() * this->w2.asDiagonal() * U,
			-2. * S2 + S.transpose() * this->precompute.cwiseQuotient(sp).cwiseProduct(this->w1) + 
			U.transpose() * (Eigen::Matrix<Type, Eigen::Dynamic, 1>::Ones(this->len) - this->precompute).cwiseQuotient(Eigen::Matrix<Type, Eigen::Dynamic, 1>::Ones(this->len) - sp).cwiseProduct(this->w2),
			1. - this->pi0fixed.sum());
		this->checklossfun2(sf * nw - dens, pi0, nw - pi0, S2, dens);
	}

	Type extrafun() const{
		return -this->len;
	}

	Type hypofun(const Type &ll, const Type &minloss) const{
		return ll;
	}

	Type familydensity(const Type &x, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &mu0, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &pi0) const{
		return dnpnorm_(Eigen::Matrix<Type, Eigen::Dynamic, 1>::Constant(1, x), mu0, pi0, this->beta).coeff(0);
	}
};

// double precision impl
// [[Rcpp::export]]
Rcpp::List npnormad_(const Eigen::VectorXd &data, const Eigen::VectorXd &mu0fixed, const Eigen::VectorXd &pi0fixed,
	const double &beta, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &maxit = 100, const int &verbose = 0){
	npnormad<double> f(data, mu0fixed, pi0fixed, beta, initpt, initpr, gridpoints);
	f.computemixdist(tol, maxit, verbose);
	return f.get_ans();
}

// [[Rcpp::export]]
Rcpp::List estpi0npnormad_(const Eigen::VectorXd &data,
	const double &beta, const double &val, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &verbose = 0){
	npnormad<double> f(data, Eigen::Matrix<double, 1, 1>::Zero(1), Eigen::Matrix<double, 1, 1>::Zero(1), beta, initpt, initpr, gridpoints);
	f.estpi0(val, tol, verbose);
	return f.get_ans();
}

// [[Rcpp::export]]
Eigen::VectorXd gfnpnormad(const Eigen::VectorXd &data, const Eigen::VectorXd &mu0fixed, const Eigen::VectorXd &pi0fixed,
	const double &beta, const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0, const Eigen::VectorXd &gridpoints){
	npnormad<double> f(data, mu0fixed, pi0fixed, beta, mu0, pi0, gridpoints);
	Eigen::VectorXd ans(gridpoints.size()), dummy(gridpoints.size());
	f.gradfunvec(gridpoints, f.mapping(mu0, pi0), ans, dummy, true, false);
	return ans;
}

// large-scale computation

template<class Type>
class npnormadw : public npfixedcomp<Type>
{
public:
	Eigen::Matrix<Type, Eigen::Dynamic, 1> weights, w1, w2;
	Type h;

	npnormadw(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &data_, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &weights_, 
		const Eigen::Matrix<Type, Eigen::Dynamic, 1> &mu0fixed_, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &pi0fixed_,
		const Type &beta_, const Type &h_, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &initpt_, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &initpr_, 
		const Eigen::Matrix<Type, Eigen::Dynamic, 1> &gridpoints_) : npfixedcomp<Type>(data_, mu0fixed_, pi0fixed_, beta_, initpt_, initpr_, gridpoints_){
		// data should be sorted beforehand
		this->weights.resize(this->len);
		this->weights = weights_;
		this->h = h_;
		this->setprecompute();
		Eigen::Matrix<Type, Eigen::Dynamic, 1> ecdf(weights);
		std::partial_sum(ecdf.data(), ecdf.data() + ecdf.size(), ecdf.data(), std::plus<Type>());
		this->w1.resize(this->len);
		this->w1 = (2 * ecdf - this->weights).cwiseProduct(this->weights) / this->weights.sum();
		this->w2.resize(this->len);
		this->w2 = 2 * this->weights - this->w1;
		this->family = "npnorm";
		this->flag = "d1";
	}

	Type lossfunction(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &maps) const{
		Eigen::Matrix<Type, Eigen::Dynamic, 1> temp = maps + this->precompute;
		return (w1.array() * temp.array().log() + w2.array() * (temp.array() * -1.).log1p()).sum() * -1.; 
	}

	Eigen::Matrix<Type, Eigen::Dynamic, 1> mapping(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &mu0, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &pi0) const{
		return pnpdiscnorm_(this->data, mu0, pi0, this->beta, this->h);
	}

	void gradfun(const Type &mu, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &dens,
		Type &ansd0, Type &ansd1, const bool &d0, const bool &d1) const{
		Type scale = 1 - this->pi0fixed.sum();
		Eigen::Matrix<Type, Eigen::Dynamic, 1> fullden = dens + this->precompute;
		Eigen::Matrix<Type, Eigen::Dynamic, 1> one = Eigen::Matrix<Type, Eigen::Dynamic, 1>::Ones(this->len);
		Eigen::Matrix<Type, Eigen::Dynamic, 1> s1 = this->w1.cwiseQuotient(fullden) - this->w2.cwiseQuotient(one - fullden);
		if (d0){
			ansd0 = (s1.dot(pnpdiscnorm_(this->data, mu, scale, this->beta, this->h) + this->precompute) + this->w2.cwiseQuotient(one - fullden).sum()) * -1. + 2 * this->weights.sum();
		}

		if (d1){
			ansd1 = dnpnorm_(this->data, mu - h, scale, this->beta).dot(s1);
		}
	}

	void gradfunvec(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &mu, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &dens,
		Eigen::Matrix<Type, Eigen::Dynamic, 1> &ansd0, Eigen::Matrix<Type, Eigen::Dynamic, 1> &ansd1, const bool &d0, const bool &d1) const{
		ansd0.resize(mu.size());
		ansd1.resize(mu.size());
		Eigen::Matrix<Type, Eigen::Dynamic, 1> fullden = dens + this->precompute;
		Eigen::Matrix<Type, Eigen::Dynamic, 1> one = Eigen::Matrix<Type, Eigen::Dynamic, 1>::Ones(this->len), s1 = this->w1.cwiseQuotient(fullden) - this->w2.cwiseQuotient(one - fullden);
		Type scale = 1 - this->pi0fixed.sum();
		if (d0){
			// Eigen::ArrayXXd temp = (pnormarray(this->data, mu, this->beta) * scale).colwise() + this->precompute;
			// ansd0 = (temp.colwise() * this->w1.cwiseQuotient(fullden).array() + (1 - temp).colwise() * (this->w2.array() / (1 - fullden.array()))).colwise().mean() * -1 + 2 * this->len; 
			ansd0 = ((pdiscnormarray(this->data, mu, this->beta, this->h) * scale).colwise() + this->precompute).transpose() * s1 * -1. + 
				Eigen::Matrix<Type, Eigen::Dynamic, 1>::Constant(mu.size(), 2. * this->weights.sum() - this->w2.cwiseQuotient(one - fullden).sum());
		} 

		if (d1){
			ansd1 = dnormarray(this->data, mu - Eigen::Matrix<Type, Eigen::Dynamic, 1>::Constant(mu.size(), h), this->beta).transpose() * s1 * scale;
		}
	}

	void computeweights(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &mu0, Eigen::Matrix<Type, Eigen::Dynamic, 1> &pi0, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &dens) const{
		Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> sf = pdiscnormarray(this->data, mu0, this->beta, this->h);
		Eigen::Matrix<Type, Eigen::Dynamic, 1> sp = dens + this->precompute;
		Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> S = sf.array().colwise() / sp.array();
		Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> U = sf.array().colwise() / (sp.array() - 1.);
		Eigen::Matrix<Type, Eigen::Dynamic, 1> S2 = S.transpose() * this->w1 + U.transpose() * this->w2;

		Eigen::Matrix<Type, Eigen::Dynamic, 1> nw = pnnqp_(S.transpose() * this->w1.asDiagonal() * S + U.transpose() * this->w2.asDiagonal() * U,
			-2. * S2 + S.transpose() * this->precompute.cwiseQuotient(sp).cwiseProduct(this->w1) + 
			U.transpose() * (Eigen::Matrix<Type, Eigen::Dynamic, 1>::Ones(this->len) - this->precompute).cwiseQuotient(Eigen::Matrix<Type, Eigen::Dynamic, 1>::Ones(this->len) - sp).cwiseProduct(this->w2),
			1. - this->pi0fixed.sum());
		this->checklossfun2(sf * nw - dens, pi0, nw - pi0, S2, dens);

	}

	Type extrafun() const{
		return -this->weights.sum();
	}

	Type hypofun(const Type &ll, const Type &minloss) const{
		return ll;
	}

	Type familydensity(const Type &x, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &mu0, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &pi0) const{
		return dnpdiscnorm_(Eigen::Matrix<Type, Eigen::Dynamic, 1>::Constant(1, x), mu0, pi0, this->beta, this->h).coeff(0);
	}
};

// double precision impl
// [[Rcpp::export]]
Rcpp::List npnormadw_(const Eigen::VectorXd &data, const Eigen::VectorXd &weights, const Eigen::VectorXd &mu0fixed, const Eigen::VectorXd &pi0fixed,
	const double &beta, const double &h, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &maxit = 100, const int &verbose = 0){
	npnormadw<double> f(data, weights, mu0fixed, pi0fixed, beta, h, initpt, initpr, gridpoints);
	f.computemixdist(tol, maxit, verbose);
	return f.get_ans();
}

// [[Rcpp::export]]
Rcpp::List estpi0npnormadw_(const Eigen::VectorXd &data, const Eigen::VectorXd &weights,
	const double &beta, const double &h, const double &val, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &verbose = 0){
	npnormadw<double> f(data, weights, Eigen::Matrix<double, 1, 1>::Zero(1), Eigen::Matrix<double, 1, 1>::Ones(1), beta, h, initpt, initpr, gridpoints);
	f.estpi0(val, tol, verbose);
	return f.get_ans();
}