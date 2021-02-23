#ifndef npfixedcomp_h
#define npfixedcomp_h

// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include "./miscfuns.h"

inline void simplifymix(Eigen::VectorXd & mu0, Eigen::VectorXd & pi0){
	if (mu0.size() != 1) {
		Eigen::VectorXi index = index2num((pi0.array().abs() > 1e-14).cast<int>());
		Eigen::VectorXd mu0new = indexing(mu0, index);
		Eigen::VectorXd pi0new = indexing(pi0, index);
		// int count = pi0.size() - pi0.cwiseEqual(0).count(), index = 0;
		// Eigen::VectorXd mu0new(count), pi0new(count);
		// for (int i = 0; i < mu0.size(); i++){
		// 	if (pi0[i] != 0){
		// 		mu0new[index] = mu0[i];
		// 		pi0new[index] = pi0[i];
		// 		index++;
		// 	}
		// }
		pi0.lazyAssign(pi0new);
		mu0.lazyAssign(mu0new);
	}
}

inline int bisearchsorted(const Eigen::VectorXd &mu0, const int &lowerindex, const int &upperindex, const double &targetval){
	// assuming mu0 is a sorted array
	if (upperindex == lowerindex){
		return upperindex;
	}else{
		int tempindex = std::ceil((lowerindex + upperindex) / 2.);
		if (mu0[tempindex] < targetval){
			return bisearchsorted(mu0, tempindex, upperindex, targetval);
			// it is possible that tempindex is the one required.
		}else{
			return bisearchsorted(mu0, lowerindex, tempindex - 1, targetval);
		}
	}
}

// This function collpse the mixing distribution
inline void collapsemix(Eigen::VectorXd &mu0, Eigen::VectorXd &pi0,
	const double &prec){
	double temppr, temppt;
	bool foo = (diff_(mu0).minCoeff() <= prec);
	// update: blocked operations.
	int i, strid;
	while (foo){
		i = 0;
		while (i < mu0.size() - 1){
			strid = bisearchsorted(mu0, i, mu0.size() - 1, mu0[i] + prec);
			if (strid > i){
				int step = strid - i + 1;
				temppr = pi0.segment(i, step).sum();
				temppt = mu0.segment(i, step).dot(pi0.segment(i, step)) / temppr;
				mu0[i] = temppt;
				pi0[i] = temppr;
				pi0.segment(i + 1, step - 1).setZero();
				i = strid + 1;
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

// This is much faster than previous version
inline void sortmix(Eigen::VectorXd &mu0, Eigen::VectorXd &pi0){
	Eigen::VectorXd mu0new = mu0, pi0new = pi0;
	Eigen::VectorXi index = Eigen::VectorXi::LinSpaced(pi0.size(), 0, pi0.size() - 1);
	std::sort(index.data(), index.data() + index.size(), comparemu0(mu0));
	// for (int i = 0; i < mu0.size(); i++){
	// 	mu0[i] = mu0new[index[i]];
	// 	pi0[i] = pi0new[index[i]];
	// }
	mu0 = indexing(mu0new, index);
	pi0 = indexing(pi0new, index);
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

	virtual void setprecompute() const{
		this->precompute.resize(this->len);
		this->precompute = mapping(this->mu0fixed, this->pi0fixed);
	}

	// For each method, need passing lossfunction, gradfun, computeweights

	// check loss function
	void checklossfun(const Eigen::VectorXd & mu0, Eigen::VectorXd & pi0, const Eigen::VectorXd &eta,
		const Eigen::VectorXd &p, const int &maxit = 10) const{
		double llorigin = this->lossfunction(this->mapping(mu0, pi0));
		double sigma = 2., alpha = 0.3333;
		int u = -1;
		double con = - p.dot(eta);
		double lhs, rhs;
		Eigen::VectorXd pi0new(pi0.size());
		do{
			u++;
			sigma *= 0.5;
			pi0new = pi0 + sigma * eta;
			lhs = this->lossfunction(this->mapping(mu0, pi0new));
			rhs = llorigin + alpha * sigma * con;
			if (lhs < rhs){
				pi0 = pi0new;
				break;
			}
			if (u + 1 > maxit){
				break;
			}
		}while(true);
	}

	virtual void checklossfun2(const Eigen::VectorXd &diff, Eigen::VectorXd &pi0, const Eigen::VectorXd &eta,
		const Eigen::VectorXd &p, const Eigen::VectorXd &dens) const{
		double llorigin = this->lossfunction(dens);
		double sigma = 2., alpha = 0.3333;
		double con = - p.dot(eta);
		double lhs, rhs;
		Eigen::VectorXd ans(pi0);
		do{
			sigma *= .5;
			lhs = this->lossfunction(dens + sigma * diff);
			rhs = llorigin + alpha * sigma * con;
			if (lhs < rhs){
				ans = pi0 + sigma * eta;
				break;
			}
			if (sigma < 0.001){
				break;
			}
		}while(true);
		pi0 = ans;
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
		simplifymix(mu0, pi0);
	}

	double Brmin(const double &lb, const double &ub, 
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
			// this->gradfun(c, dens, dumc, fc, true, false);
			return c; // fx = dumc;
		}else{
			// this->gradfun(s, dens, dums, fs, true, false);
			return s; // fx = dums;
		}
	}

	Eigen::VectorXd solvegradd1(const Eigen::VectorXd &dens, const double & tol = 1e-6) const{
		Eigen::VectorXd pointsval, pointsgrad;
		this->gradfunvec(this->gridpoints, dens, pointsval, pointsgrad, false, true);
		const int L = this->gridpoints.size();
		Eigen::VectorXi index = index2num((pointsgrad.head(L - 1).array() < 0).cast<int>() * (pointsgrad.tail(L - 1).array() > 0).cast<int>());
		Eigen::VectorXd ans(index.size());
		if (index.size() > 0){
			ans = Eigen::VectorXd::NullaryExpr(index.size(),
				[this, &index, &dens, &tol](Eigen::Index row){
					return this->Brmin(this->gridpoints[index[row]], this->gridpoints[index[row] + 1], dens, tol);
				});
			// double x;
			// for (auto i = 0; i < ans.size(); ++i){
			// 	this->Brmin(x, gridpoints[index[i]], gridpoints[index[i] + 1], dens);
			// 	ans[i] = x;
			// }
			this->gradfunvec(ans, dens, pointsval, pointsgrad, true, false);
			Eigen::VectorXd anstemp = indexing(ans, index2num((pointsval.array() < 0).cast<int>()));
			ans.lazyAssign(anstemp);
		}
		double pv2,ps2;
		this->gradfun(this->gridpoints[0], dens, pv2, ps2, true, false);
		if (pv2 < 0 & pointsgrad[0] > 0){
			ans.conservativeResize(ans.size() + 1);
			ans.tail<1>() = gridpoints.head<1>();
		}
		this->gradfun(this->gridpoints.tail<1>()[0], dens, pv2, ps2, true, false);
		if (pv2 < 0 & pointsgrad.tail<1>()[0] < 0){
			ans.conservativeResize(ans.size() + 1);
			ans.tail<1>() = gridpoints.tail<1>();
		}

		return ans;
	}

	double Dfmin(const Eigen::Vector3d &x1, const Eigen::Vector3d &fx1,
		const Eigen::VectorXd &dens, const double &tol = 1e-6) const{
		double lb = x1[0], ub = x1[2], newpoint, fnewpoint, dummy;
		Eigen::Vector3d xx(x1), fxx(fx1);

		// ensure tail has the smallest fxx.
		if (fxx[0] < fxx[1]){
			std::swap(xx[0], xx[1]);
			std::swap(fxx[0], fxx[1]);
		}
		if (fxx[1] < fxx[2]){
			std::swap(xx[1], xx[2]);
			std::swap(fxx[1], fxx[2]);
		}	

		while (ub - lb > tol){
			newpoint = newmin(xx, fxx);
			newpoint = (std::isnan(newpoint) | (newpoint < lb) | (newpoint > ub)) ? (lb + ub) / 2 : newpoint;
			this->gradfun(newpoint, dens, fnewpoint, dummy, true, false);

			if (fnewpoint > fxx[2]){
				// xx[2] still the local min, replacing bounds
				if (newpoint > xx[2]){
					ub = newpoint;
					fxx = (xx.array() > newpoint).select(Eigen::Vector3d::Constant(fnewpoint), fxx);
					xx = (xx.array() > newpoint).select(Eigen::Vector3d::Constant(newpoint), xx);
				}else{
					lb = newpoint;
					fxx = (xx.array() < newpoint).select(Eigen::Vector3d::Constant(fnewpoint), fxx);
					xx = (xx.array() < newpoint).select(Eigen::Vector3d::Constant(newpoint), xx);
				}
			}else{
				// newpoint is the new min, replacing bounds using xx[2]
				if (xx[2] > newpoint){
					ub = xx[2];
					fxx = (xx.array() > xx[2]).select(Eigen::Vector3d::Constant(fxx[2]), fxx);
					xx = (xx.array() > xx[2]).select(Eigen::Vector3d::Constant(xx[2]), xx);
					xx[2] = newpoint;
					fxx[2] = fnewpoint;
				}else{
					lb = xx[2];
					fxx = (xx.array() < xx[2]).select(Eigen::Vector3d::Constant(fxx[2]), fxx);
					xx = (xx.array() < xx[2]).select(Eigen::Vector3d::Constant(xx[2]), xx);
					xx[2] = newpoint;
					fxx[2] = fnewpoint;
				}
			}
		}

		if (fxx[2] < 0){
			return xx[2];
		}else{
			return NAN;
		}
	}

	Eigen::VectorXd solvegradd0(const Eigen::VectorXd &dens, const double &tol = 1e-6) const{
		Eigen::VectorXd pointsval, pointsgrad; // pointsgrad not referenced.
		this->gradfunvec(this->gridpoints, dens, pointsval, pointsgrad, true, false);
		Eigen::VectorXd temp = diff_(pointsval);
		const int L = temp.size();
		Eigen::VectorXi index = index2num((temp.head(L - 1).array() < 0).cast<int>() * (temp.tail(L - 1).array() > 0).cast<int>());
		Eigen::VectorXd ans(index.size());
		if (index.size() > 0){
			ans = Eigen::VectorXd::NullaryExpr(index.size(),
				[this, &index, &dens, &pointsval, &tol](Eigen::Index row){
					return this->Dfmin(this->gridpoints.segment<3>(index[row]), pointsval.segment<3>(index[row]), dens, tol);
				});
			Eigen::VectorXd anstemp = indexing(ans, index2num(1 - ans.array().isNaN().cast<int>()));
			ans.lazyAssign(anstemp);
			// double x, fx;
			// for (auto ptr = index.data(); ptr < index.data() + index.size(); ptr++){
			// 	// inputx << gridpoints[index[i]], gridpoints[index[i] + 2], gridpoints[index[i] + 1];
   //  			// inputfx << pointsval[index[i]], pointsval[index[i] + 2], pointsval[index[i] + 1];
   //  			this->Dfmin(x, fx, gridpoints.segment<3>(*ptr), pointsval.segment<3>(*ptr), dens);
			// 	if (fx < 0){
   //  				ans[length] = x;
   //  				length++;
   //  			}
			// }
			// ans.conservativeResize(length);
		}
		if (pointsval[0] < 0 & diff_(pointsval.head<2>())[0] > 0){
			ans.conservativeResize(ans.size() + 1);
			ans.tail<1>() = gridpoints.head<1>();
		}
		if (pointsval.tail<1>()[0] < 0 & diff_(pointsval.tail<2>())[0] < 0){
			ans.conservativeResize(ans.size() + 1);
			ans.tail<1>() = gridpoints.tail<1>();			
		}
		return ans;
	}

	Eigen::VectorXd solvegrad(const Eigen::VectorXd &dens, const double & tol = 1e-6) const{
		if (this->flag == "d1"){
			return solvegradd1(dens, tol);
		}else{
			return solvegradd0(dens, tol);
		}
	}

	// compute mixing distribution
	void computemixdist(const double &tol = 1e-6, const int &maxit = 100, const int &verbose = 0) const{
		Eigen::VectorXd mu0 = initpt, pi0 = initpr * (1. - pi0fixed.sum());
		int iter = 0, convergence = 0;
		Eigen::VectorXd newpoints, dens = this->mapping(mu0, pi0);
		double closs = this->lossfunction(dens), nloss;

		do{
			newpoints.lazyAssign(this->solvegrad(dens, tol));

			mu0.conservativeResize(mu0.size() + newpoints.size());
			pi0.conservativeResize(pi0.size() + newpoints.size());
			mu0.tail(newpoints.size()) = newpoints;
			pi0.tail(newpoints.size()) = Eigen::VectorXd::Zero(newpoints.size());
			sortmix(mu0, pi0);

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
				if (this->flag == "d1") {Rcpp::Rcout<<"gradient derivative: "<<pointsgrad.transpose()<<std::endl;}
				Rcpp::Rcout<<"loss:"<<this->lossfunction(this->mapping(mu0, pi0))<<std::endl;
				
			}

			this->computeweights(mu0, pi0, dens); // return using mu0, pi0? remember to sort
			if (verbose >= 2){
				Rcpp::Rcout<<"After computeweights"<<std::endl;
				Rcpp::Rcout<<"support points: "<<mu0.transpose()<<std::endl;
				Rcpp::Rcout<<"probabilities: "<<pi0.transpose()<<std::endl;
				Rcpp::Rcout<<"loss:"<<this->lossfunction(this->mapping(mu0, pi0))<<std::endl;
			}
			this->collapse(mu0, pi0);
			if (verbose >= 2){
				Rcpp::Rcout<<"After collapse"<<std::endl;
				Rcpp::Rcout<<"support points: "<<mu0.transpose()<<std::endl;
				Rcpp::Rcout<<"probabilities: "<<pi0.transpose()<<std::endl;
				Rcpp::Rcout<<"loss:"<<this->lossfunction(this->mapping(mu0, pi0))<<std::endl;
			}
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
		this->computemixdist(tol);
		double minloss = result["ll"], ll;
		Eigen::VectorXd dens = this->mapping(Eigen::VectorXd::Zero(1), Eigen::VectorXd::Ones(1));
		if (this->hypofun(this->lossfunction(dens) + this->extrafun(), minloss) < val){
			double maxgrad, maxgrad2;
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
			double lb = 0, ub = 1, flb = minloss, fub = this->lossfunction(dens) + this->extrafun();
			Rcpp::List mix = this->result["mix"];
			Rcpp::NumericVector tmu0 = mix["pt"], tpi0 = mix["pr"];
			Eigen::Map<Eigen::VectorXd> mu1(tmu0.begin(), tmu0.length());
			Eigen::Map<Eigen::VectorXd> pi1(tpi0.begin(), tpi0.length());	
			double sp = this->familydensity(0, mu1, pi1) / this->familydensity(0, Eigen::VectorXd::Zero(1), Eigen::VectorXd::Ones(1));
			this->pi0fixed = Eigen::VectorXd::Constant(1, sp);
			this->setprecompute();
			this->computemixdist();
			ll = this->result["ll"];
			int iter = 1;
			Eigen::Matrix3d A;
			Eigen::Vector3d b, x1;
			while (std::abs(this->hypofun(ll, minloss) - val) > tol & std::abs(ub - lb) > tol){
				if (verbose >= 1){
					Rcpp::Rcout<<"Iter: "<<iter<<" lower: "<<lb<<" upper: "<<ub<<std::endl;
					Rcpp::Rcout<<"current val: "<<sp<<" fval: "<<ll<<std::endl;
				}

				if (this->hypofun(ll, minloss) - val < 0 & sp > lb){
					lb = sp; flb = ll;
				}
				if (this->hypofun(ll, minloss) - val > 0 & sp < ub){
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

				A << lb * lb, lb, 1., 
					 sp * sp, sp, 1., 
					 ub * ub, ub, 1.;
				b<<this->hypofun(flb, minloss) - val, this->hypofun(ll, minloss) - val, this->hypofun(fub, minloss) - val;
				x1 = A.inverse() * b;

				if (this->hypofun(ll, minloss) - val < 0){
					lb = sp; flb = ll;
				}
				if (this->hypofun(ll, minloss) - val > 0){
					ub = sp; fub = ll;
				}
				;
				sp = (-x1[1] + std::sqrt(x1[1] * x1[1] - 4 * x1[0] * x1[2])) / 2 / x1[0];
				sp = (std::isnan(sp) | (sp < lb) | (sp > ub)) ? (lb + ub) / 2 : sp;

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
				iter+=1;
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

	virtual void computeweights(const Eigen::VectorXd &mu0, Eigen::VectorXd &pi0, const Eigen::VectorXd &dens) const{}

	virtual double extrafun() const{
		return 0;
	}

	virtual double hypofun(const double &ll, const double &minloss) const{
		// likelihood methods use ll - minloss
		// distance methods use ll.
		return 0;
	}

	virtual double familydensity(const double &x, const Eigen::VectorXd &mu0, const Eigen::VectorXd &pi0) const{
		return 0;
	}

};

#endif