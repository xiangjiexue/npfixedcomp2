// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include "../inst/include/npfixedcomp.h"

// [[Rcpp::depends(RcppEigen)]]

template<class Type>
class npnormcll : public npfixedcomp<Type>
{
public:

	npnormcll(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &data_, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &mu0fixed_, 
		const Eigen::Matrix<Type, Eigen::Dynamic, 1> &pi0fixed_, const Type &beta_, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &initpt_, 
		const Eigen::Matrix<Type, Eigen::Dynamic, 1> &initpr_, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &gridpoints_) : npfixedcomp<Type>(data_, mu0fixed_, pi0fixed_, beta_, initpt_, initpr_, gridpoints_){
		this->setprecompute();
		this->family = "npnormc";
		this->flag = "d0";
	}

	Type lossfunction(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &maps) const{
		return (maps + this->precompute).array().log().sum() * -1.;
	}

	Eigen::Matrix<Type, Eigen::Dynamic, 1> mapping(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &mu0, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &pi0) const{
		return dnpnormc_(this->data, mu0, pi0, this->beta);
	}

	void gradfun(const Type &mu, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &dens,
		Type &ansd0, Type &ansd1, const bool &d0, const bool &d1) const{
		Eigen::Matrix<Type, Eigen::Dynamic, 1> fullden = (dens + this->precompute).cwiseInverse();
		Type scale = 1. - this->pi0fixed.sum();
		if (d0){
			ansd0 = (dens - dnpnormc_(this->data, mu, scale, this->beta)).dot(fullden);
		}
		
		// Eigen::VectorXd temp2 = ((this->beta + 4) * std::pow(mu, 3.) -  3 * this->beta * mu * mu * this->data.array() +
		// 	mu * (2 * this->beta * this->data.array().square() + this->beta - 2) - this->beta * this->data.array() - 
		// 	2 * std::pow(mu, 5.)) / std::pow(1 - mu * mu, 3.0);
		// ansd1 = temp2.cwiseProduct(temp).dot(fullden) * scale;
	}

	void gradfunvec(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &mu, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &dens,
		Eigen::Matrix<Type, Eigen::Dynamic, 1> &ansd0, Eigen::Matrix<Type, Eigen::Dynamic, 1> &ansd1, const bool &d0, const bool &d1) const{
		ansd0.resize(mu.size());
		ansd1.resize(mu.size());
		Eigen::Matrix<Type, Eigen::Dynamic, 1> fullden = (dens + this->precompute).cwiseInverse();
		Type scale = 1. - this->pi0fixed.sum();
		if (d0){
			ansd0 = Eigen::Matrix<Type, Eigen::Dynamic, 1>::Constant(mu.size(), dens.dot(fullden));
			ansd0.noalias() -= dnormcarray(this->data, mu, this->beta).transpose() * fullden * scale;
		}
	}

	void computeweights(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &mu0, Eigen::Matrix<Type, Eigen::Dynamic, 1> &pi0, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &dens) const{
		Eigen::Matrix<Type, Eigen::Dynamic, 1> fp = dens + this->precompute;
		Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> sp = dnormcarray(this->data, mu0, this->beta), tp = sp.array().colwise() / fp.array();
		Eigen::Matrix<Type, Eigen::Dynamic, 1> nw(pi0.size());

		if (this->len > 1e3){
			nw = pnnqp_(tp.transpose() * tp, tp.transpose() * (this->precompute.cwiseQuotient(fp) - Eigen::Matrix<Type, Eigen::Dynamic, 1>::Constant(this->len, 2)), 1. - this->pi0fixed.sum());
		}else{
			nw = pnnlssum_(tp, Eigen::Matrix<Type, Eigen::Dynamic, 1>::Constant(this->len, 2) - this->precompute.cwiseQuotient(fp), 1. - this->pi0fixed.sum());
		}
		this->checklossfun2(sp * nw - dens, pi0, nw - pi0, tp.colwise().sum(), dens);
	}

	Type hypofun(const Type &ll, const Type &minloss) const{
		return ll - minloss;
	}

	Type familydensity(const Type &x, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &mu0, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &pi0) const{
		return dnpnormc_(Eigen::Matrix<Type, Eigen::Dynamic, 1>::Constant(1, x), mu0, pi0, this->beta).coeff(0);
	}
};

// [[Rcpp::export]]
Rcpp::List npnormcll_(const Eigen::VectorXd &data, const Eigen::VectorXd &mu0fixed, const Eigen::VectorXd &pi0fixed,
	const double &beta, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &maxit = 100, const int &verbose = 0){
	npnormcll<double> f(data, mu0fixed, pi0fixed, beta, initpt, initpr, gridpoints);
	f.computemixdist(tol, maxit, verbose);
	return f.get_ans();
}

// [[Rcpp::export]]
Rcpp::List estpi0npnormcll_(const Eigen::VectorXd &data,
	const double &beta, const double &val, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &verbose = 0){
	npnormcll<double> f(data, Eigen::Matrix<double, 1, 1>::Zero(1), Eigen::Matrix<double, 1, 1>::Ones(1), beta, initpt, initpr, gridpoints);
	f.estpi0(val, tol, verbose);
	return f.get_ans();
}
