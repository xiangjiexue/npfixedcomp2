// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include "../inst/include/npfixedcomp.h"

// [[Rcpp::depends(RcppEigen)]]

template<class Type>
class nppoisll : public npfixedcomp<Type>
{
public:
	Eigen::VectorX<Type> weights;

	nppoisll(const Eigen::VectorX<Type> &data_, const Eigen::VectorX<Type> &weights_, 
		const Eigen::VectorX<Type> &mu0fixed_, const Eigen::VectorX<Type> &pi0fixed_,
		const Type &beta_, const Eigen::VectorX<Type> &initpt_, const Eigen::VectorX<Type> &initpr_, 
		const Eigen::VectorX<Type> &gridpoints_) : npfixedcomp<Type>(data_, mu0fixed_, pi0fixed_, beta_, initpt_, initpr_, gridpoints_){
		this->setprecompute();
		this->weights.resize(this->len);
		this->weights = weights_;
		this->family = "nppois";
		this->flag = "d0";
	}

	Type lossfunction(const Eigen::VectorX<Type> &maps) const{
		return ((maps + this->precompute).array().log() * this->weights.array()).sum() * -1.;
	}

	Eigen::VectorX<Type> mapping(const Eigen::VectorX<Type> &mu0, const Eigen::VectorX<Type> &pi0) const{
		return dnppois_(this->data, mu0, pi0, this->beta);
	}

	void gradfun(const Type &mu, const Eigen::VectorX<Type> &dens,
		Type &ansd0, Type &ansd1, const bool &d0, const bool &d1) const{
		Eigen::VectorX<Type> fullden = this->weights.cwiseQuotient(dens + this->precompute);
		Type scale = 1. - this->pi0fixed.sum();
		if (d0){
			ansd0 = (dens - dnppois_(this->data, mu, scale, this->beta)).dot(fullden);
		}
	}

	void gradfunvec(const Eigen::VectorX<Type> &mu, const Eigen::VectorX<Type> &dens,
		Eigen::VectorX<Type> &ansd0, Eigen::VectorX<Type> &ansd1, const bool &d0, const bool &d1) const{
		ansd0.resize(mu.size());
		ansd1.resize(mu.size());
		Eigen::VectorX<Type> fullden = this->weights.cwiseQuotient(dens + this->precompute);
		Type scale = 1. - this->pi0fixed.sum();
		if (d0){
			ansd0 = Eigen::VectorX<Type>::Constant(mu.size(), dens.dot(fullden));
			ansd0.noalias() -= dpoisarray(this->data, mu, this->beta).transpose() * fullden * scale;
		}
	}

	void computeweights(const Eigen::VectorX<Type> &mu0, Eigen::VectorX<Type> &pi0, const Eigen::VectorX<Type> &dens) const{
		Eigen::VectorX<Type> fp = dens + this->precompute;
		Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> sp = dpoisarray(this->data, mu0, this->beta), tp = sp.array().colwise() / fp.array();

		Eigen::VectorX<Type> nw = pnnlssum_(tp.cwiseProduct(this->weights.cwiseSqrt().replicate(1, mu0.size())), 
			(Eigen::VectorX<Type>::Constant(this->len, 2) - this->precompute.cwiseQuotient(fp)).cwiseProduct(this->weights.cwiseSqrt()), 1. - this->pi0fixed.sum());
		this->checklossfun2(sp * nw - dens, pi0, nw - pi0, tp.transpose() * this->weights, dens);
	}

	Type hypofun(const Type &ll, const Type &minloss) const{
		return ll - minloss;
	}

	Type familydensity(const Type &x, const Eigen::VectorX<Type> &mu0, const Eigen::VectorX<Type> &pi0) const{
		return dnppois_(Eigen::VectorX<Type>::Constant(1, x), mu0, pi0, this->beta).coeff(0);
	}
};

// [[Rcpp::export]]
Rcpp::List nppoisll_(const Eigen::VectorXd &data, const Eigen::VectorXd &weights, const Eigen::VectorXd &mu0fixed, const Eigen::VectorXd &pi0fixed,
	const double &beta, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &maxit = 100, const int &verbose = 0){
	nppoisll<double> f(data, weights, mu0fixed, pi0fixed, beta, initpt, initpr, gridpoints);
	f.computemixdist(tol, maxit, verbose);
	return f.get_ans();
}

// [[Rcpp::export]]
Rcpp::List estpi0nppoisll_(const Eigen::VectorXd &data, const Eigen::VectorXd &weights,
	const double &beta, const double &val, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &verbose = 0){
	nppoisll<double> f(data, weights, Eigen::Matrix<double, 1, 1>::Zero(1), Eigen::Matrix<double, 1, 1>::Ones(1), beta, initpt, initpr, gridpoints);
	f.estpi0(val, tol, verbose);
	return f.get_ans();
}
