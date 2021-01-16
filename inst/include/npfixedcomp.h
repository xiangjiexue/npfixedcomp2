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

inline double newmin(const Eigen::Vector3d &x, const Eigen::Vector3d &fx){
	// v is the best so far (v = x[2], fv = fx[2])
	// u = x[0], fu = fx[0], w = x[1], fw = fx[1]
	// page 272 of Scientific Computing: An Introductory Survey
	double p = (x[2] - x[0]) * (x[2] - x[0]) * (fx[2] - fx[1]) - (x[2] - x[1]) * (x[2] - x[1]) * (fx[2] - fx[0]);
	double q = 2 * ((x[2] - x[0]) * (fx[2] - fx[1]) - (x[2] - x[1]) * (fx[2] - fx[0]));
	return x[2] - p / q;
}


class npfixedcomp
{
public:
	const Eigen::Ref<const Eigen::VectorXd> data;
	mutable Eigen::VectorXd mu0fixed;
	mutable Eigen::VectorXd pi0fixed;
	double beta;
	// pass it from R for now
	mutable Eigen::VectorXd initpt;
	mutable Eigen::VectorXd initpr;
	const Eigen::Ref<const Eigen::VectorXd> gridpoints;
	int len;
	mutable Rcpp::List result;
	std::string family, flag;
	mutable Eigen::VectorXd precompute;

	npfixedcomp(const Eigen::VectorXd &data_, const Eigen::VectorXd &mu0fixed_, const Eigen::VectorXd &pi0fixed_,
		const double &beta_, const Eigen::VectorXd &initpt_, const Eigen::VectorXd &initpr_, const Eigen::VectorXd &gridpoints_) : data(data_),
	mu0fixed(mu0fixed_), pi0fixed(pi0fixed_), beta(beta_), initpt(initpt_), initpr(initpr_), gridpoints(gridpoints_){
		this->len = this->data.size();
	}

	void setprecompute() const{
		this->precompute.resize(this->len);
		this->precompute = mapping(this->mu0fixed, this->pi0fixed);
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
		this->gradfun(a, dens, duma, fa, false, true);
		this->gradfun(b, dens, dumb, fb, false, true);

		double s = a, fs = fa, dums, c = a, fc = fa, dumc;

		// this function will only be called if flb * fub < 0
		while (std::abs(fc) > tol & std::abs(fs) > tol & std::abs(b - a) > tol){
			c = (a + b) / 2;
			this->gradfun(c, dens, dumc, fc, false, true);

			if (fa != fc & fb != fc){
				s = a * fb * fc / (fa - fb) / (fa - fc) +
					b * fa * fc / (fb - fa) / (fb - fc) +
					c * fa * fb / (fc - fa) / (fc - fb);
			}else{
				s = b - fb * (b - a) / (fb - fa);
			}

			if (s > a & s < b){
				this->gradfun(s, dens, dums, fs, false, true);
			}else{
				s = c; fs = fc;
			}

			if (c > s){
				std::swap(c, s);
				std::swap(fc, fs);
			}

			// a < c < s < b

			if (fc * fs < 0){
				a = c; fa = fc; 
				b = s; fb = fs; 
			}else{
				if (fs * fb < 0){
					a = s; fa = fs; 
				}else{
					b = c; fb = fc;
				}
			}
		}

		if (std::abs(fc) < tol){
			this->gradfun(c, dens, dumc, fc, true, false);
			x = c; fx = dumc;
		}else{
			this->gradfun(s, dens, dums, fs, true, false);
			x = s; fx = dums;
		}
	}

	Eigen::VectorXd solvegradd1(const Eigen::VectorXd &dens) const{
		Eigen::VectorXd pointsval, pointsgrad;
		this->gradfunvec(gridpoints, dens, pointsval, pointsgrad, false, true);
		int length = 0;
    	double x, fx;
    	Eigen::VectorXd ans(this->len);
    	this->gradfun(gridpoints.head(1)[0], dens, pointsval[0], pointsgrad[0], true, false);
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
    	this->gradfun(gridpoints.tail(1)[0], dens, pointsval[pointsval.size() - 1], pointsgrad[pointsgrad.size() - 1], true, false);
    	if (pointsval.tail(1)[0] < 0){
    		ans[length] = gridpoints.tail(1)[0];
    		length++;
    	}
    	ans.conservativeResize(length);
    	return ans;
	}

	void Dfmin(double & x, double & fx, const Eigen::Vector3d &x1, const Eigen::Vector3d &fx1,
		const Eigen::VectorXd &dens, const double &tol = 1e-6) const{
		double newpoint, fnewpoint, dummy, lb = x1[0], ub = x1[1];
		Eigen::Vector3d xx(x1), fxx(fx1);
		// ensure tail has the smallest fxx.
		if (fxx[0] < fxx[1]){
			std::swap(xx[0], xx[1]);
			std::swap(fxx[0], fxx[1]);
		}	
		while ((xx.maxCoeff() - xx.minCoeff()) > tol){
			newpoint = newmin(xx, fxx);
			if (newpoint < lb & newpoint > ub){
				Eigen::MatrixXd A(3, 3);
				A.col(0) = xx.array().square(); A.col(1) = xx; A.col(2).setOnes();
				Eigen::VectorXd x1 = A.inverse() * fxx;
				newpoint = -x1[1] / x1[0] * 0.5;
			}
			this->gradfun(newpoint, dens, fnewpoint, dummy, true, false);
			std::swap(xx[0], xx[1]);
			std::swap(fxx[0], fxx[1]);
			std::swap(xx[1], xx[2]);
			std::swap(fxx[1], fxx[2]);
			xx[2] = newpoint;
			fxx[2] = fnewpoint;
		}

		x = xx[2]; fx = fxx[2];
	}

	Eigen::VectorXd solvegradd0(const Eigen::VectorXd &dens) const{
		Eigen::VectorXd pointsval, pointsgrad; // pointsgrad not referenced.
		this->gradfunvec(gridpoints, dens, pointsval, pointsgrad, true, false);
		int length = 0;
    	double x, fx;
    	Eigen::VectorXd ans(this->len);
    	if (pointsval.head(1)[0] < 0){
    		ans[length] = gridpoints.head(1)[0];
    		length++;
    	}
    	Eigen::Vector3d inputx(3), inputfx(3);
    	for (auto i = 0; i < gridpoints.size() - 2; i++){
    		if ((pointsval[i + 1] - pointsval[i] < 0) & (pointsval[i + 2] - pointsval[i + 1] > 0)){
    			inputx << gridpoints[i], gridpoints[i + 2], gridpoints[i + 1];
    			inputfx << pointsval[i], pointsval[i + 2], pointsval[i + 1];
    			this->Dfmin(x, fx, inputx, inputfx, dens);
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

	Eigen::VectorXd solvegrad(const Eigen::VectorXd &dens) const{
		if (this->flag == "d1"){
			return solvegradd1(dens);
		}else{
			return solvegradd0(dens);
		}
	}

	// compute mixing distribution
	void computemixdist(const double &tol = 1e-6, const int &maxit = 100, const int &verbose = 0) const{
		Eigen::VectorXd mu0 = initpt, pi0 = initpr * (1. - pi0fixed.sum());
		int iter = 0, convergence = 0;
		Eigen::VectorXd newpoints, dens = this->mapping(mu0, pi0);
		double closs = this->lossfunction(dens), nloss;

		do{
			newpoints.lazyAssign(this->solvegrad(dens));

			if (verbose >= 1){
				Rcpp::Rcout<<"Iteration: "<<iter<<" with loss "<<nloss<<std::endl;
				Rcpp::Rcout<<"support points: "<<mu0.transpose()<<std::endl;
				Rcpp::Rcout<<"probabilities: "<<pi0.transpose()<<std::endl;
			}
			
			if (verbose >= 2){
				Rcpp::Rcout<<"new points: "<<newpoints.transpose()<<std::endl;
				Eigen::VectorXd pointsval, pointsgrad;
				this->gradfunvec(newpoints, dens, pointsval, pointsgrad, true, true);
				Rcpp::Rcout<<"gradient: "<<pointsval.transpose()<<std::endl;
				Rcpp::Rcout<<"gradient derivative: "<<pointsgrad.transpose()<<std::endl;
			}

			this->computeweights(mu0, pi0, dens, newpoints); // return using mu0, pi0? remember to sort
			iter++;
			dens = this->mapping(mu0, pi0);
			nloss = this->lossfunction(dens);

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
		this->gradfunvec(mu0, dens, maxgrad, maxgrad2, true, false);

		Rcpp::List r = Rcpp::List::create(Rcpp::Named("iter") = iter,
			Rcpp::Named("family") = family,
			Rcpp::Named("min.gradient") = maxgrad.minCoeff(),
			Rcpp::Named("beta") = this->beta,
			Rcpp::Named("mix") = Rcpp::List::create(Rcpp::Named("pt") = mu0new, Rcpp::Named("pr") = pi0new),
			Rcpp::Named("ll") = closs + this->extrafun(),
			Rcpp::Named("flag") = flag,
			Rcpp::Named("convergence") = convergence);

		this->result = Rcpp::clone(r);
	}

	void estpi0(const double &val, const double &tol = 1e-6, const int &verbose = 0) const{
		// this is only suitable for estimate the probability of location at 0.
		this->mu0fixed.resize(1);
		this->mu0fixed.setZero();
		this->pi0fixed.resize(1);
		this->pi0fixed.setZero();
		this->setprecompute();
		this->computemixdist();
		double minloss = result["ll"], ll, maxgrad, maxgrad2;;
		Eigen::VectorXd dens = this->mapping(Eigen::VectorXd::Zero(1), Eigen::VectorXd::Ones(1));
		if (this->lossfunction(dens) - minloss < val){
			this->gradfun(0, dens, maxgrad, maxgrad2, true, false);
			Rcpp::List r = Rcpp::List::create(Rcpp::Named("iter") = 0,
				Rcpp::Named("family") = family,
				Rcpp::Named("min.gradient") = maxgrad,
				Rcpp::Named("beta") = this->beta,
				Rcpp::Named("mix") = Rcpp::List::create(Rcpp::Named("pt") = 0, Rcpp::Named("pr") = 1),
				Rcpp::Named("ll") = this->lossfunction(dens) + this->extrafun(),
				Rcpp::Named("flag") = flag,
				Rcpp::Named("convergence") = 0);

			this->result = Rcpp::clone(r);
		}else{
			double lb = 0, ub = 1, flb = minloss, fub = this->lossfunction(dens);
			Rcpp::List mix = this->result["mix"];
			Rcpp::NumericVector tmu0 = mix["pt"], tpi0 = mix["pr"];
			Eigen::Map<Eigen::VectorXd> mu1(tmu0.begin(), tmu0.length());
			Eigen::Map<Eigen::VectorXd> pi1(tpi0.begin(), tpi0.length());	
			double sp = dnpnorm_(Eigen::VectorXd::Zero(1), mu1, pi1, this->beta)[0] * M_SQRT2 * M_SQRT_PI * this->beta;
			this->pi0fixed = Eigen::VectorXd::Constant(1, sp);
			this->setprecompute();
			this->computemixdist();
			ll = this->result["ll"];
			int iter = 1;
			while (std::abs(ll - minloss - val) > tol & std::abs(ub - lb) > tol){
				if (verbose >= 1){
					Rcpp::Rcout<<"Iter: "<<iter<<" lower: "<<lb<<" upper: "<<ub<<std::endl;
					Rcpp::Rcout<<"current val: "<<sp<<" fval: "<<ll<<std::endl;
				}

				if (ll - minloss - val < 0 & sp > lb){
					lb = sp; flb = ll;
				}
				if (ll - minloss - val > 0 & sp < ub){
					ub = sp; fub = ll;
				}
				sp = (lb + ub) / 2;

				mix = this->result["mix"];
				tmu0 = mix["pt"]; tpi0 = mix["pr"];
				new (&mu1) Eigen::Map<Eigen::VectorXd>(tmu0.begin(), tmu0.length());
				new (&pi1) Eigen::Map<Eigen::VectorXd>(tpi0.begin(), tpi0.length());
				initpt.resize(mu1.size());
				initpt = mu1;
				initpr.resize(pi1.size());
				initpr = pi1;
				this->pi0fixed = Eigen::VectorXd::Constant(1, sp);
				this->setprecompute();
				this->computemixdist();
				ll = this->result["ll"];

				Eigen::MatrixXd A(3, 3);
				A << lb * lb, lb, 1, 
					 sp * sp, sp, 1, 
					 ub * ub, ub,  1;
				Eigen::Vector3d b;
				b<<flb - minloss - val, ll - minloss - val, fub - minloss - val;
				Eigen::Vector3d x1 = A.inverse() * b;

				if (ll - minloss - val < 0){
					lb = sp; flb = ll;
				}
				if (ll - minloss - val > 0){
					ub = sp; fub = ll;
				}
				;
				sp = (-x1[1] + std::sqrt(x1[1] * x1[1] - 4 * x1[0] * x1[2])) / 2 / x1[0];

				mix = this->result["mix"];
				tmu0 = mix["pt"]; tpi0 = mix["pr"];
				new (&mu1) Eigen::Map<Eigen::VectorXd>(tmu0.begin(), tmu0.length());
				new (&pi1) Eigen::Map<Eigen::VectorXd>(tpi0.begin(), tpi0.length());
				initpt.resize(mu1.size());
				initpt = mu1;
				initpr.resize(pi1.size());
				initpr = pi1;
				this->pi0fixed = Eigen::VectorXd::Constant(1, sp);
				this->setprecompute();
				this->computemixdist();
				ll = this->result["ll"];
				iter+=2;
			}
		}
	}

	// functions to each specific type

	virtual Eigen::VectorXd mapping(const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0) const{
		return Eigen::VectorXd::Zero(1);
	}

	virtual double lossfunction(const Eigen::VectorXd & maps) const{
		return 0;
	}

	virtual void gradfun(const double &mu, const Eigen::VectorXd &dens,
		double &ansd0, double & ansd1, const bool &d0, const bool &d1) const{}

	// vectorised function for gradfund1 (can be overwriten)
	virtual void gradfunvec(const Eigen::VectorXd &mu, const Eigen::VectorXd &dens, 
		Eigen::VectorXd &ansd0, Eigen::VectorXd &ansd1, const bool &d0, const bool& d1) const{
		ansd0.resize(mu.size());
		ansd1.resize(mu.size());
		double *ansd0ptr = ansd0.data(), *ansd1ptr = ansd1.data();
		for (auto muptr = mu.data(); muptr < mu.data() + mu.size(); muptr++, ansd0ptr++, ansd1ptr++){
			this->gradfun(*muptr, dens, *ansd0ptr, *ansd1ptr, d0, d1);
		}
	}

	virtual void computeweights(Eigen::VectorXd &mu0, Eigen::VectorXd &pi0, 
		const Eigen::VectorXd &dens, const Eigen::VectorXd &newpoints) const{}

	virtual double extrafun() const{
		return 0;
	}

};

#endif