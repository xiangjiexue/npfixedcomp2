// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include "../inst/include/npfixedcomp.h"

// [[Rcpp::depends(RcppEigen)]]

template<class Type>
class npnormcvm : public npfixedcomp<Type>
{
public:

	npnormcvm(const Eigen::VectorX<Type> &data_, const Eigen::VectorX<Type> &mu0fixed_, 
		const Eigen::VectorX<Type> &pi0fixed_, const Type &beta_, const Eigen::VectorX<Type> &initpt_, 
		const Eigen::VectorX<Type> &initpr_, const Eigen::VectorX<Type> &gridpoints_) : npfixedcomp<Type>(data_, mu0fixed_, pi0fixed_, beta_, initpt_, initpr_, gridpoints_){
		// data should be sorted beforehand
		this->setprecompute();
		this->family = "npnorm";
		this->flag = "d1";
	}

	void setprecompute() const{
		this->precompute.resize(this->len);
		this->precompute = Eigen::VectorX<Type>::LinSpaced(this->len, 1., 2. * this->len - 1.) / (2. * this->len) - 
			mapping(this->mu0fixed, this->pi0fixed);
	}

	Type lossfunction(const Eigen::VectorX<Type> &maps) const{
		return (maps - this->precompute).squaredNorm();
	}

	Eigen::VectorX<Type> mapping(const Eigen::VectorX<Type> &mu0, const Eigen::VectorX<Type> &pi0) const{
		return pnpnorm_(this->data, mu0, pi0, this->beta);
	}

	void gradfun(const Type &mu, const Eigen::VectorX<Type> &dens,
		Type &ansd0, Type &ansd1, const bool &d0, const bool &d1) const{
		Eigen::VectorX<Type> fullden = dens - this->precompute;
		Type scale = 1. - this->pi0fixed.sum();
		if (d0){
			ansd0 = (pnpnorm_(this->data, mu, scale, this->beta) - dens).dot(fullden) * 2.;	
		}
		if (d1){
			ansd1 = dnpnorm_(this->data, mu, 1., this->beta).dot(fullden) * (-2. * scale);
		}
	}

	void gradfunvec(const Eigen::VectorX<Type> &mu, const Eigen::VectorX<Type> &dens,
		Eigen::VectorX<Type> &ansd0, Eigen::VectorX<Type> &ansd1, const bool &d0, const bool &d1) const{
		ansd0.resize(mu.size());
		ansd1.resize(mu.size());
		Eigen::VectorX<Type> fullden = dens - this->precompute;
		Type scale = 1. - this->pi0fixed.sum();
		if (d0){
			// ansd0 = (pnormarray(this->data, mu, this->beta).transpose() * fullden * scale - Eigen::VectorXd::Constant(mu.size(), fullden.dot(dens))) * 2;
			ansd0 = pnormarray(this->data, mu, this->beta).transpose() * fullden * (scale * 2.) - Eigen::VectorX<Type>::Constant(mu.size(), fullden.dot(dens) * 2.);
		}
		if (d1){
			ansd1 = dnormarray(this->data, mu, this->beta).transpose() * fullden * -2 * scale;
		}
	}

	void computeweights(const Eigen::VectorX<Type> &mu0, Eigen::VectorX<Type> &pi0, const Eigen::VectorX<Type> &dens) const{
		Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> fp = pnormarray(this->data, mu0, this->beta);
		if (this->len > 1e3){
			pi0 = pnnqp_(fp.transpose() * fp, fp.transpose() * this->precompute * -1., 1. - this->pi0fixed.sum());
		}else{
			pi0 = pnnlssum_(pnormarray(this->data, mu0, this->beta), this->precompute, 1. - this->pi0fixed.sum());
		}
	}

	Type extrafun() const{
		return 1 / 12 / this->len;
	}

	Type hypofun(const Type &ll, const Type &minloss) const{
		return ll;
	}

	Type familydensity(const Type &x, const Eigen::VectorX<Type> &mu0, const Eigen::VectorX<Type> &pi0) const{
		return dnpnorm_(Eigen::VectorX<Type>::Constant(1, x), mu0, pi0, this->beta).coeff(0);
	}
};

// [[Rcpp::export]]
Rcpp::List npnormcvm_(const Eigen::VectorXd &data, const Eigen::VectorXd &mu0fixed, const Eigen::VectorXd &pi0fixed,
	const double &beta, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &maxit = 100, const int &verbose = 0){
	npnormcvm<double> f(data, mu0fixed, pi0fixed, beta, initpt, initpr, gridpoints);
	f.computemixdist(tol, maxit, verbose);
	return f.get_ans();
}

// [[Rcpp::export]]
Rcpp::List estpi0npnormcvm_(const Eigen::VectorXd &data,
	const double &beta, const double &val, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &verbose = 0){
	npnormcvm<double> f(data, Eigen::Matrix<double, 1, 1>::Zero(1), Eigen::Matrix<double, 1, 1>::Zero(1), beta, initpt, initpr, gridpoints);
	f.estpi0(val, tol, verbose);
	return f.get_ans();
}

// [[Rcpp::export]]
Eigen::VectorXd gfnpnormcvm(const Eigen::VectorXd &data, const Eigen::VectorXd &mu0fixed, const Eigen::VectorXd &pi0fixed,
	const double &beta, const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0, const Eigen::VectorXd &gridpoints){
	npnormcvm<double> f(data, mu0fixed, pi0fixed, beta, mu0, pi0, gridpoints);
	Eigen::VectorXd ans(gridpoints.size()), dummy(gridpoints.size());
	f.gradfunvec(gridpoints, f.mapping(mu0, pi0), ans, dummy, true, false);
	return ans;
}

// large-scale computation

template<class Type>
class npnormcvmw : public npfixedcomp<Type>
{
public:
	Eigen::VectorX<Type> weights;
	Type h;

	npnormcvmw(const Eigen::VectorX<Type> &data_, const Eigen::VectorX<Type> &weights_, 
		const Eigen::VectorX<Type> &mu0fixed_, const Eigen::VectorX<Type> &pi0fixed_,
		const Type &beta_, const Type &h_, const Eigen::VectorX<Type> &initpt_, const Eigen::VectorX<Type> &initpr_, 
		const Eigen::VectorX<Type> &gridpoints_) : npfixedcomp<Type>(data_, mu0fixed_, pi0fixed_, beta_, initpt_, initpr_, gridpoints_){
		// data should be sorted beforehand
		this->weights.resize(this->len);
		this->weights = weights_;
		this->h = h_;
		this->setprecompute();
		this->family = "npnorm";
		this->flag = "d1";
	}

	void setprecompute() const{
		this->precompute.resize(this->len);
		Eigen::VectorX<Type> ecdf(weights);
		std::partial_sum(ecdf.data(), ecdf.data() + ecdf.size(), ecdf.data(), std::plus<Type>());
		this->precompute = (ecdf - this->weights / 2) / this->weights.sum() - 
			mapping(this->mu0fixed, this->pi0fixed);
	}

	Type lossfunction(const Eigen::VectorX<Type> &maps) const{
		return (maps - this->precompute).cwiseAbs2().dot(this->weights);
	}

	Eigen::VectorX<Type> mapping(const Eigen::VectorX<Type> &mu0, const Eigen::VectorX<Type> &pi0) const{
		return pnpdiscnorm_(this->data, mu0, pi0, this->beta, this->h);
	}

	void gradfun(const Type &mu, const Eigen::VectorX<Type> &dens,
		Type &ansd0, Type &ansd1, const bool &d0, const bool &d1) const{
		Eigen::VectorX<Type> fullden = (dens - this->precompute).cwiseProduct(this->weights);
		Type scale = 1. - this->pi0fixed.sum();
		if (d0){
			ansd0 = (pnpdiscnorm_(this->data, mu, scale, this->beta, this->h) - dens).dot(fullden) * 2;
		}

		if (d1){
			ansd1 = ddiscnormarray(this->data, mu, this->beta, this->h).dot(fullden) * -2 * scale;
		}
	}

	void gradfunvec(const Eigen::VectorX<Type> &mu, const Eigen::VectorX<Type> &dens,
		Eigen::VectorX<Type> &ansd0, Eigen::VectorX<Type> &ansd1, const bool &d0, const bool &d1) const{
		ansd0.resize(mu.size());
		ansd1.resize(mu.size());
		Eigen::VectorX<Type> fullden = (dens - this->precompute).cwiseProduct(this->weights);
		Type scale = 1. - this->pi0fixed.sum();
		if (d0){
			ansd0 = (pdiscnormarray(this->data, mu, this->beta, this->h) * scale - dens.rowwise().replicate(mu.size())).transpose() * fullden * 2;
		}
		if (d1){
			ansd1 = ddiscnormarray(this->data, mu, this->beta, this->h).transpose() * fullden * -2 * scale;
		}
	}

	void computeweights(const Eigen::VectorX<Type> &mu0, Eigen::VectorX<Type> &pi0, const Eigen::VectorX<Type> &dens) const{
		pi0 = pnnlssum_(pdiscnormarray(this->data, mu0, this->beta, this->h).cwiseProduct(this->weights.cwiseSqrt().replicate(1, mu0.size())),
			this->precompute.cwiseProduct(this->weights.cwiseSqrt()), 1. - this->pi0fixed.sum());
	}

	Type extrafun() const{
		Eigen::VectorX<Type> ecdf(weights);
		std::partial_sum(ecdf.data(), ecdf.data() + ecdf.size(), ecdf.data(), std::plus<Type>());
		return this->weights.sum() / 3. - ((ecdf - this->weights / 2.) / this->weights.sum()).cwiseAbs2().dot(this->weights);
	}

	Type hypofun(const Type &ll, const Type &minloss) const{
		return ll;
	}

	Type familydensity(const Type &x, const Eigen::VectorX<Type> &mu0, const Eigen::VectorX<Type> &pi0) const{
		return dnpdiscnorm_(Eigen::VectorX<Type>::Constant(1, x), mu0, pi0, this->beta, this->h).coeff(0);
	}
};

// [[Rcpp::export]]
Rcpp::List npnormcvmw_(const Eigen::VectorXd &data, const Eigen::VectorXd &weights, const Eigen::VectorXd &mu0fixed, const Eigen::VectorXd &pi0fixed,
	const double &beta, const double &h, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &maxit = 100, const int &verbose = 0){
	npnormcvmw<double> f(data, weights, mu0fixed, pi0fixed, beta, h, initpt, initpr, gridpoints);
	f.computemixdist(tol, maxit, verbose);
	return f.get_ans();
}

// [[Rcpp::export]]
Rcpp::List estpi0npnormcvmw_(const Eigen::VectorXd &data, const Eigen::VectorXd &weights,
	const double &beta, const double &h, const double &val, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &verbose = 0){
	npnormcvmw<double> f(data, weights, Eigen::Matrix<double, 1, 1>::Zero(1), Eigen::Matrix<double, 1, 1>::Ones(1), beta, h, initpt, initpr, gridpoints);
	f.estpi0(val, tol, verbose);
	return f.get_ans();
}