// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include "../inst/include/npfixedcomp.h"

// [[Rcpp::depends(RcppEigen)]]

class npnormcvm : public npfixedcomp
{
public:

	npnormcvm(const Eigen::VectorXd &data_, const Eigen::VectorXd &mu0fixed_, const Eigen::VectorXd &pi0fixed_,
		const double &beta_, const Eigen::VectorXd &initpt_, const Eigen::VectorXd &initpr_, 
		const Eigen::VectorXd &gridpoints_) : npfixedcomp(data_, mu0fixed_, pi0fixed_, beta_, initpt_, initpr_, gridpoints_){
		// data should be sorted beforehand
		this->setprecompute();
		family = "npnorm";
		flag = "d1";
	}

	void setprecompute() const{
		this->precompute.resize(this->len);
		this->precompute = Eigen::VectorXd::LinSpaced(this->len, 1, 2 * this->len - 1) / 2 / this->len - 
			mapping(this->mu0fixed, this->pi0fixed);
	}

	double lossfunction(const Eigen::VectorXd &maps) const{
		return (maps - this->precompute).squaredNorm();
	}

	Eigen::VectorXd mapping(const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0) const{
		return pnpnorm_(this->data, mu0, pi0, this->beta);
	}

	void gradfun(const double &mu, const Eigen::VectorXd &dens,
		double &ansd0, double &ansd1, const bool &d0, const bool &d1) const{
		Eigen::VectorXd fullden = dens - this->precompute;
		double scale = 1 - this->pi0fixed.sum();
		if (d0){
			ansd0 = (pnpnorm_(this->data, Eigen::VectorXd::Constant(1, mu), Eigen::VectorXd::Constant(1, scale), this->beta) - dens).dot(fullden) * 2;	
		}
		if (d1){
			ansd1 = dnpnorm_(this->data, Eigen::VectorXd::Constant(1, mu), Eigen::VectorXd::Ones(1), this->beta).dot(fullden) * -2 * scale;
		}
	}

	void gradfunvec(const Eigen::VectorXd &mu, const Eigen::VectorXd &dens,
		Eigen::VectorXd &ansd0, Eigen::VectorXd &ansd1, const bool &d0, const bool &d1) const{
		ansd0.resize(mu.size());
		ansd1.resize(mu.size());
		Eigen::VectorXd fullden = dens - this->precompute;
		double scale = 1 - this->pi0fixed.sum();
		if (d0){
			ansd0 = fullden.transpose() * (pnormarray(this->data, mu, this->beta) * scale - dens.rowwise().replicate(mu.size())) * 2;
		}
		if (d1){
			ansd1 = fullden.transpose() * dnormarray(this->data, mu, this->beta) * -2 * scale;
		}
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

	double hypofun(const double &ll, const double &minloss) const{
		return ll;
	}

	double familydensity(const double &x, const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0) const{
		return dnpnorm_(Eigen::VectorXd::Constant(1, x), mu0, pi0, this->beta)[0];
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

// [[Rcpp::export]]
Rcpp::List estpi0npnormcvm_(const Eigen::VectorXd &data,
	const double &beta, const double &val, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &verbose = 0){
	npnormcvm f(data, Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(1), beta, initpt, initpr, gridpoints);
	f.estpi0(val, tol, verbose);
	return f.result;
}

// large-scale computation

class npnormcvmw : public npfixedcomp
{
public:
	Eigen::VectorXd weights;
	double h;

	npnormcvmw(const Eigen::VectorXd &data_, const Eigen::VectorXd &weights_, const Eigen::VectorXd &mu0fixed_, const Eigen::VectorXd &pi0fixed_,
		const double &beta_, const double &h_, const Eigen::VectorXd &initpt_, const Eigen::VectorXd &initpr_, 
		const Eigen::VectorXd &gridpoints_) : npfixedcomp(data_, mu0fixed_, pi0fixed_, beta_, initpt_, initpr_, gridpoints_){
		// data should be sorted beforehand
		this->weights.resize(this->len);
		this->weights = weights_;
		this->h = h_;
		this->setprecompute();
		family = "npnorm";
		flag = "d1";
	}

	void setprecompute() const{
		this->precompute.resize(this->len);
		Eigen::VectorXd ecdf(weights);
		std::partial_sum(ecdf.data(), ecdf.data() + ecdf.size(), ecdf.data(), std::plus<double>());
		this->precompute = (ecdf - this->weights / 2) / this->weights.sum() - 
			mapping(this->mu0fixed, this->pi0fixed);
	}

	double lossfunction(const Eigen::VectorXd &maps) const{
		return (maps - this->precompute).cwiseAbs2().dot(this->weights);
	}

	Eigen::VectorXd mapping(const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0) const{
		return pnpdiscnorm_(this->data, mu0, pi0, this->beta, this->h);
	}

	void gradfun(const double &mu, const Eigen::VectorXd &dens,
		double &ansd0, double &ansd1, const bool &d0, const bool &d1) const{
		Eigen::VectorXd fullden = (dens - this->precompute).cwiseProduct(weights);
		double scale = 1 - this->pi0fixed.sum();
		if (d0){
			ansd0 = (pnpdiscnorm_(this->data, Eigen::VectorXd::Constant(1, mu), Eigen::VectorXd::Constant(1, scale), this->beta, this->h) - dens).dot(fullden) * 2;
		}

		if (d1){
			ansd1 = dnpdiscnorm_(this->data, Eigen::VectorXd::Constant(1, mu), Eigen::VectorXd::Ones(1), this->beta, this->h).dot(fullden) * -2 * scale;
		}
	}

	void gradfunvec(const Eigen::VectorXd &mu, const Eigen::VectorXd &dens,
		Eigen::VectorXd &ansd0, Eigen::VectorXd &ansd1, const bool &d0, const bool &d1) const{
		ansd0.resize(mu.size());
		ansd1.resize(mu.size());
		Eigen::VectorXd fullden = (dens - this->precompute).cwiseProduct(this->weights);
		double scale = 1 - this->pi0fixed.sum();
		if (d0){
			ansd0 = fullden.transpose() * (pdiscnormarray(this->data, mu, this->beta, this->h) * scale - dens.rowwise().replicate(mu.size())) * 2;
		}
		if (d1){
			ansd1 = fullden.transpose() * ddiscnormarray(this->data, mu, this->beta, this->h) * -2 * scale;
		}
	}

	void computeweights(Eigen::VectorXd &mu0, Eigen::VectorXd &pi0, 
		const Eigen::VectorXd &dens, const Eigen::VectorXd &newpoints) const{
		Eigen::VectorXd mu0new(mu0.size() + newpoints.size());
		mu0new.head(mu0.size()) = mu0; mu0new.tail(newpoints.size()) = newpoints;

		Eigen::MatrixXd sp = pdiscnormarray(this->data, mu0new, this->beta, this->h).array().colwise() * this->weights.array().sqrt();
		Eigen::VectorXd pi0new = pnnlssum_(sp, this->precompute.array() * this->weights.array().sqrt(), 1. - this->pi0fixed.sum());

		sortmix(mu0new, pi0new);

		this->collapse(mu0new, pi0new);

		mu0.lazyAssign(mu0new);
		pi0.lazyAssign(pi0new);
	}

	double extrafun() const{
		Eigen::VectorXd ecdf(weights);
		std::partial_sum(ecdf.data(), ecdf.data() + ecdf.size(), ecdf.data(), std::plus<double>());
		return this->weights.sum() / 3 - ((ecdf - this->weights / 2) / this->weights.sum()).cwiseAbs2().dot(this->weights);
	}

	double hypofun(const double &ll, const double &minloss) const{
		return ll;
	}

	double familydensity(const double &x, const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0) const{
		return dnpdiscnorm_(Eigen::VectorXd::Constant(1, x), mu0, pi0, this->beta, this->h)[0];
	}
};

// [[Rcpp::export]]
Rcpp::List npnormcvmw_(const Eigen::VectorXd &data, const Eigen::VectorXd &weights, const Eigen::VectorXd &mu0fixed, const Eigen::VectorXd &pi0fixed,
	const double &beta, const double &h, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &maxit = 100, const int &verbose = 0){
	npnormcvmw f(data, weights, mu0fixed, pi0fixed, beta, h, initpt, initpr, gridpoints);
	f.computemixdist(tol, maxit, verbose);
	return f.result;
}

// [[Rcpp::export]]
Rcpp::List estpi0npnormcvmw_(const Eigen::VectorXd &data, const Eigen::VectorXd &weights,
	const double &beta, const double &h, const double &val, const Eigen::VectorXd &initpt, const Eigen::VectorXd &initpr, const Eigen::VectorXd &gridpoints,
	const double &tol = 1e-6, const int &verbose = 0){
	npnormcvmw f(data, weights, Eigen::VectorXd::Zero(1), Eigen::VectorXd::Ones(1), beta, h, initpt, initpr, gridpoints);
	f.estpi0(val, tol, verbose);
	return f.result;
}