#ifndef npfixedcomp_h
#define npfixedcomp_h

// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include "./miscfuns.h"

inline void simplifymix(Eigen::VectorXd & mu0, Eigen::VectorXd & pi0){
	if (mu0.size() != 1) {
		int count = pi0.size() - pi0.cwiseEqual(0).count(), index = 0;
		Eigen::VectorXd mu0new(count), pi0new(count);
		for (int i = 0; i < mu0.size(); i++){
			if (pi0[i] != 0){
				mu0new[index] = mu0[i];
				pi0new[index] = pi0[i];
				index++;
			}
		}
		pi0.lazyAssign(pi0new);
		mu0.lazyAssign(mu0new);
	}
}

// This function collpse the mixing distribution
inline void collapsemix(Eigen::VectorXd &mu0, Eigen::VectorXd &pi0,
	const double &prec){
	bool foo;
	if (mu0.size() > 1){
		foo = (diff_(mu0).minCoeff() <= prec);
//		sortmix(mu0, pi0);
		double temp;
		int i;
		while (foo){
			i = 0;
			while (i < mu0.size() - 1){
				if (mu0[i + 1] - mu0[i] <= prec){
					temp = pi0[i] + pi0[i + 1];
					mu0[i] = (mu0[i] * pi0[i] + mu0[i + 1] * pi0[i + 1]) / temp;
					pi0[i + 1] = 0;
					pi0[i] = temp;
					i = i + 2;
				}else{
					i++;
				}
			}
			simplifymix(mu0, pi0);
			if (mu0.size() <= 1) {
				foo = false;
			}else{
				foo = (diff_(mu0).minCoeff() <= prec);
			}
		}
	}
}

// sort mixing distribution
class comparemu0
{
private:
	Eigen::VectorXd mu0;
public:
	comparemu0(const Eigen::VectorXd &mu0_) : mu0(mu0_) {};

	const bool operator()(const int & x, const int & y) const {return mu0[x] < mu0[y];};
};
// This is much faster than previous version
inline void sortmix(Eigen::VectorXd &mu0, Eigen::VectorXd &pi0){
	Eigen::VectorXd mu0new = mu0, pi0new = pi0;
	Eigen::VectorXi index = Eigen::VectorXi::LinSpaced(pi0.size(), 0, pi0.size() - 1);
	std::sort(index.data(), index.data() + index.size(), comparemu0(mu0));
	for (int i = 0; i < mu0.size(); i++){
		mu0[i] = mu0new[index[i]];
		pi0[i] = pi0new[index[i]];
	}
}


class npfixedcomp
{
public:
	const Eigen::Ref<const Eigen::VectorXd> data;
	const Eigen::Ref<const Eigen::VectorXd> mu0fixed;
	const Eigen::Ref<const Eigen::VectorXd> pi0fixed;
	double beta;
	// pass it from R for now
	const Eigen::Ref<const Eigen::VectorXd> initpt;
	const Eigen::Ref<const Eigen::VectorXd> initpr;
	const Eigen::Ref<const Eigen::VectorXd> gridpoints;
	int len;
	mutable Rcpp::List result;

	npfixedcomp(const Eigen::VectorXd &data_, const Eigen::VectorXd &mu0fixed_, const Eigen::VectorXd &pi0fixed_,
		const double &beta_, const Eigen::VectorXd &initpt_, const Eigen::VectorXd &initpr_, const Eigen::VectorXd &gridpoints_) : data(data_),
	mu0fixed(mu0fixed_), pi0fixed(pi0fixed_), beta(beta_), initpt(initpt_), initpr(initpr_), gridpoints(gridpoints_){
		this->len = this->data.size();
	}

	// For each method, need passing lossfunction, gradfun, computeweights

	// check loss function
	void checklossfun(Eigen::VectorXd & mu0, Eigen::VectorXd & pi0, const Eigen::VectorXd &eta,
		const Eigen::VectorXd &p, const int &maxit = 100) const{
		double llorigin = this->lossfunction(this->mapping(mu0, pi0));
		double sigma = 0.5, alpha = 1/3, u = -1.;
		double con = - p.dot(eta);
		double lhs, rhs;
		Eigen::VectorXd pi0new(pi0.size());
		do{
			u++;
			pi0new = (pi0 + std::pow(sigma, u) * eta).eval();
			lhs = this->lossfunction(this->mapping(mu0, pi0new));
			rhs = llorigin + alpha * std::pow(sigma, u) * con;
			if (lhs < rhs){
				pi0 = pi0new;
				break;
			}
			if (u + 1 > maxit){
				break;
			}
		}while(true);
	}

	// collapsing
	void collapse(Eigen::VectorXd & mu0, Eigen::VectorXd & pi0, const double &tol = 1e-6) const{
		double ll = this->lossfunction(this->mapping(mu0, pi0)), nll;
		double ntol = std::max(tol * 0.1, ll * 1e-16), prec;
		Eigen::VectorXd mu0new = mu0, pi0new = pi0;
		do {
			if (mu0.size() <= 1) break;
			prec = 10 * diff_(mu0).minCoeff();
			collapsemix(mu0new, pi0new, prec);
			nll = this->lossfunction(this->mapping(mu0new, pi0new));
			if (nll <= ll + ntol){
				pi0.lazyAssign(pi0new);
				mu0.lazyAssign(mu0new);
			}else{break;}
		} while(true);
	}

	void Brmin(double & x, double & fx, const double &lb, const double &ub, 
		const Eigen::VectorXd &dens, const double & tol = 1e-6) const{
		double fa, fb, a = lb, b = ub, duma, dumb;
		this->gradfun(a, dens, duma, fa);
		this->gradfun(b, dens, dumb, fb);

		double s = a, fs = fb, dums, c, fc, dumc;

		// this function will only be called if flb * fub < 0
		while (std::abs(fb) > tol & std::abs(fs) > tol & std::abs(b - a) > tol){
			c = (a + b) / 2;
			this->gradfun(c, dens, dumc, fc);

			if (fa != fc & fb != fc){
				s = a * fb * fc / (fa - fb) / (fa - fc) +
					b * fa * fc / (fb - fa) / (fb - fc) +
					c * fa * fb / (fc - fa) / (fc - fb);
			}else{
				s = b - fb * (b - a) / (fb - fa);
			}

			if (s > a & s < b){
				this->gradfun(s, dens, dums, fs);
			}else{
				s = c; fs = fc; dums = dumc;
			}

			if (c > s){
				std::swap(c, s);
				std::swap(fc, fs);
				std::swap(dumc, dums);
			}

			// a < s < c < b

			if (fc * fs < 0){
				a = s; fa = fs; duma = dums;
				b = c; fb = fc; dumb = dumc;
			}else{
				if (fs * fb < 0){
					a = c; fa = fc; duma = dumc;
				}else{
					b = s; fb = fs; dumb = dums;
				}
			}
		}

		if (std::abs(fb) < tol){
			x = b; fx = dumb;
		}else{
			x = s; fx = dums;
		}
	}

	Eigen::VectorXd solvegradd1(const Eigen::VectorXd &dens) const{
		Eigen::VectorXd pointsval, pointsgrad;
		this->gradfunvec(gridpoints, dens, pointsval, pointsgrad);
		int length = 0;
    	double x, fx;
    	Eigen::VectorXd ans(this->len);
    	if (pointsval.head(1)[0] < 0){
    		ans[length] = gridpoints.head(1)[0];
    		length++;
    	}
    	for (auto i = 0; i < gridpoints.size() - 1; i++){
    		if (pointsgrad[i] < 0 & pointsgrad[i + 1] > 0){
    			this->Brmin(x, fx, gridpoints[i], gridpoints[i + 1], dens);
    			if (fx < 0){
    				ans[length] = x;
    				length++;
    			}
    		}
    	}
    	if (pointsval.tail(1)[0] < 0){
    		ans[length] = gridpoints.tail(1)[0];
    		length++;
    	}
    	ans.conservativeResize(length);
    	return ans;
	}

	// compute mixing distribution
	void computemixdist(const double &tol = 1e-6, const int &maxit = 100, const bool &verbose = false) const{
		Eigen::VectorXd mu0 = initpt, pi0 = initpr * (1. - pi0fixed.sum());
		int iter = 0, convergence = 0;
		Eigen::VectorXd newpoints, dens = this->mapping(mu0, pi0);
		double closs = this->lossfunction(dens), nloss;

		do{
			newpoints.lazyAssign(this->solvegradd1(dens));
			this->computeweights(mu0, pi0, dens, newpoints); // return using mu0, pi0? remember to sort
			iter++;
			dens = this->mapping(mu0, pi0);
			nloss = this->lossfunction(dens);

			if (verbose){
				Rcpp::Rcout<<"Iteration: "<<iter<<" with loss "<<nloss<<std::endl;
				Rcpp::Rcout<<"new points: "<<newpoints.transpose()<<std::endl;
				Rcpp::Rcout<<"support points: "<<mu0.transpose()<<std::endl;
				Rcpp::Rcout<<"probabilities: "<<pi0.transpose()<<std::endl;
			}

			if (closs - nloss < tol){
				convergence = 0;
				break;
			}

			if (iter > maxit){
				convergence = 1;
				break;
			}

			closs = nloss;
		}while(true);

		Eigen::VectorXd mu0new(mu0.size() + mu0fixed.size());
		Eigen::VectorXd pi0new(pi0.size() + pi0fixed.size());
		mu0new.head(mu0.size()) = mu0; mu0new.tail(mu0fixed.size()) = mu0fixed;
		pi0new.head(pi0.size()) = pi0; pi0new.tail(pi0fixed.size()) = pi0fixed;

		sortmix(mu0new, pi0new);
		Eigen::VectorXd maxgrad, maxgrad2;
		this->gradfunvec(mu0, dens, maxgrad, maxgrad2);
		double dd0, dd02;
		this->gradfun(0, dens, dd0, dd02);

		Rcpp::List r = Rcpp::List::create(Rcpp::Named("iter") = iter,
			Rcpp::Named("family") = "npnorm",
			Rcpp::Named("min.gradient") = maxgrad.minCoeff(),
			Rcpp::Named("beta") = this->beta,
			Rcpp::Named("mix") = Rcpp::List::create(Rcpp::Named("pt") = mu0new, Rcpp::Named("pr") = pi0new),
			Rcpp::Named("ll") = closs,
			Rcpp::Named("dd0") = dd0,
			Rcpp::Named("convergence") = convergence);

		this->result = Rcpp::clone(r);
	}

	// functions to each specific type
	virtual void print(const int & level = 0) const{
		Rcpp::Rcout<<"mu0fixed: "<<this->mu0fixed.transpose()<<std::endl;
		Rcpp::Rcout<<"pi0fixed: "<<this->pi0fixed.transpose()<<std::endl;
		Rcpp::Rcout<<"beta: "<<this->beta<<std::endl;
		Rcpp::Rcout<<"length: "<<this->len<<std::endl;
		Rcpp::Rcout<<"gridpoints: "<<this->gridpoints.transpose()<<std::endl;
		Rcpp::Rcout<<"initial loss: "<<this->lossfunction(this->mapping(initpt, initpr))<<std::endl;

		if (level == 1){
			Rcpp::Rcout<<"data: "<<this->data.transpose()<<std::endl;
		}
	}

	virtual Eigen::VectorXd mapping(const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0) const{
		return Eigen::VectorXd::Zero(1);
	}

	virtual double lossfunction(const Eigen::VectorXd & maps) const{
		return 0;
	}

	virtual void gradfun(const double &mu, const Eigen::VectorXd &dens,
		double &ansd0, double & ansd1) const{}

	// vectorised function for gradfund1 (can be overwriten)
	virtual void gradfunvec(const Eigen::VectorXd &mu, const Eigen::VectorXd &dens, 
		Eigen::VectorXd &ansd0, Eigen::VectorXd &ansd1) const{
		ansd0.resize(mu.size());
		ansd1.resize(mu.size());
		for (auto i = 0; i < mu.size(); i++){
			this->gradfun(mu[i], dens, ansd0[i], ansd1[i]);
		}
	}

	virtual void computeweights(Eigen::VectorXd &mu0, Eigen::VectorXd &pi0, 
		const Eigen::VectorXd &dens, const Eigen::VectorXd &newpoints) const{}

};

#endif