#ifndef npfixedcomp_h
#define npfixedcomp_h

// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include "./miscfuns.h"

template<class Type>
inline void simplifymix(Eigen::Matrix<Type, Eigen::Dynamic, 1>& mu0, 
	Eigen::Matrix<Type, Eigen::Dynamic, 1> & pi0){
	if (mu0.size() != 1) {
		Eigen::VectorXi index = index2num((pi0.array().abs() > 1e-14).template cast<int>());
		Eigen::Matrix<Type, Eigen::Dynamic, 1> mu0new = indexing(mu0, index);
		Eigen::Matrix<Type, Eigen::Dynamic, 1> pi0new = indexing(pi0, index);
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

// This function collpse the mixing distribution
template<class Type>
inline void collapsemix(Eigen::Matrix<Type, Eigen::Dynamic, 1> &mu0, Eigen::Matrix<Type, Eigen::Dynamic, 1> &pi0,
	const Type &prec){
	bool foo;
	if (mu0.size() > 1){
		foo = (diff_(mu0).minCoeff() <= prec);
//		sortmix(mu0, pi0);
		Type temp;
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
template<class Type>
inline void sortmix(Eigen::Matrix<Type, Eigen::Dynamic, 1> &mu0, Eigen::Matrix<Type, Eigen::Dynamic, 1> &pi0){
	Eigen::Matrix<Type, Eigen::Dynamic, 1> mu0new = mu0, pi0new = pi0;
	Eigen::VectorXi index = Eigen::VectorXi::LinSpaced(pi0.size(), 0, pi0.size() - 1);
	std::sort(index.data(), index.data() + index.size(), comparemu0<Type>(mu0));
	// for (int i = 0; i < mu0.size(); i++){
	// 	mu0[i] = mu0new[index[i]];
	// 	pi0[i] = pi0new[index[i]];
	// }
	mu0 = indexing(mu0new, index);
	pi0 = indexing(pi0new, index);
}

template<class Type>
inline Type newmin(const Eigen::Matrix<Type, 3, 1> &x, const Eigen::Matrix<Type, 3, 1> &fx){
	// v is the best so far (v = x[2], fv = fx[2])
	// u = x[0], fu = fx[0], w = x[1], fw = fx[1]
	// page 272 of Scientific Computing: An Introductory Survey
	Type p = (x[2] - x[0]) * (x[2] - x[0]) * (fx[2] - fx[1]) - (x[2] - x[1]) * (x[2] - x[1]) * (fx[2] - fx[0]);
	Type q = 2 * ((x[2] - x[0]) * (fx[2] - fx[1]) - (x[2] - x[1]) * (fx[2] - fx[0]));
	return x[2] - p / q;
}

template<class Type>
class npfixedcomp
{
public:
	const Eigen::Ref<const Eigen::Matrix<Type, Eigen::Dynamic, 1> > data;
	mutable Eigen::Matrix<Type, Eigen::Dynamic, 1> mu0fixed;
	mutable Eigen::Matrix<Type, Eigen::Dynamic, 1> pi0fixed;
	Type beta;
	// pass it from R for now
	mutable Eigen::Matrix<Type, Eigen::Dynamic, 1> initpt;
	mutable Eigen::Matrix<Type, Eigen::Dynamic, 1> initpr;
	const Eigen::Ref<const Eigen::Matrix<Type, Eigen::Dynamic, 1> > gridpoints;
	int len;
	mutable int convergence, iter = 0;
	mutable Rcpp::List result;
	std::string family, flag;
	mutable Eigen::Matrix<Type, Eigen::Dynamic, 1> precompute;
	mutable Eigen::Matrix<Type, Eigen::Dynamic, 1> resultpt;
	mutable Eigen::Matrix<Type, Eigen::Dynamic, 1> resultpr;

	npfixedcomp(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &data_, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &mu0fixed_, 
		const Eigen::Matrix<Type, Eigen::Dynamic, 1> &pi0fixed_, const Type &beta_, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &initpt_,
		 const Eigen::Matrix<Type, Eigen::Dynamic, 1> &initpr_, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &gridpoints_) : data(data_),
	mu0fixed(mu0fixed_), pi0fixed(pi0fixed_), beta(beta_), initpt(initpt_), initpr(initpr_), gridpoints(gridpoints_){
		this->len = this->data.size();
	}

	virtual void setprecompute() const{
		this->precompute.resize(this->len);
		this->precompute = mapping(this->mu0fixed, this->pi0fixed);
	}

	// For each method, need passing lossfunction, gradfun, computeweights

	// check loss function
	void checklossfun(const Eigen::Matrix<Type, Eigen::Dynamic, 1> & mu0, Eigen::Matrix<Type, Eigen::Dynamic, 1> & pi0, 
		const Eigen::Matrix<Type, Eigen::Dynamic, 1> &eta, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &p, const int &maxit = 10) const{
		Type llorigin = this->lossfunction(this->mapping(mu0, pi0));
		Type sigma = 2., alpha = 0.3333;
		int u = -1;
		Type con = - p.dot(eta);
		Type lhs, rhs;
		Eigen::Matrix<Type, Eigen::Dynamic, 1> pi0new(pi0.size());
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

	virtual void checklossfun2(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &diff, Eigen::Matrix<Type, Eigen::Dynamic, 1> &pi0, 
		const Eigen::Matrix<Type, Eigen::Dynamic, 1> &eta, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &p, 
		const Eigen::Matrix<Type, Eigen::Dynamic, 1> &dens) const{
		Type llorigin = this->lossfunction(dens);
		Type sigma = 2., alpha = 0.3333;
		Type con = - p.dot(eta);
		Type lhs, rhs;
		Eigen::Matrix<Type, Eigen::Dynamic, 1> ans(pi0);
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
	void collapse(Eigen::Matrix<Type, Eigen::Dynamic, 1> & mu0, Eigen::Matrix<Type, Eigen::Dynamic, 1> & pi0, const Type &tol = 1e-6) const{
		Type ll = this->lossfunction(this->mapping(mu0, pi0)), nll;
		Type ntol = std::max(tol * 0.1, ll * 1e-16), prec;
		Eigen::Matrix<Type, Eigen::Dynamic, 1> mu0new = mu0, pi0new = pi0;
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

	Type Brmin(const Type &lb, const Type &ub, 
		const Eigen::Matrix<Type, Eigen::Dynamic, 1> &dens, const Type & tol = 1e-6) const{
		Type fa, fb, a = lb, b = ub, duma, dumb;
		this->gradfun(a, dens, duma, fa, false, true);
		this->gradfun(b, dens, dumb, fb, false, true);

		Type s = a, fs = fa, dums, c = a, fc = fa, dumc;

		// this function will only be called if flb * fub < 0
		while (std::fabs(fc) > tol & std::fabs(fs) > tol & std::fabs(b - a) > tol){
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

		if (std::fabs(fc) < tol){
			// this->gradfun(c, dens, dumc, fc, true, false);
			return c; // fx = dumc;
		}else{
			// this->gradfun(s, dens, dums, fs, true, false);
			return s; // fx = dums;
		}
	}

	Eigen::Matrix<Type, Eigen::Dynamic, 1> solvegradd1(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &dens, const Type & tol = 1e-6) const{
		Eigen::Matrix<Type, Eigen::Dynamic, 1> pointsval, pointsgrad;
		this->gradfunvec(this->gridpoints, dens, pointsval, pointsgrad, false, true);
		const int L = this->gridpoints.size();
		Eigen::VectorXi index = index2num((pointsgrad.head(L - 1).array() < 0).template cast<int>() * (pointsgrad.tail(L - 1).array() > 0).template cast<int>());
		Eigen::Matrix<Type, Eigen::Dynamic, 1> ans(index.size());
		if (index.size() > 0){
			ans = Eigen::Matrix<Type, Eigen::Dynamic, 1>::NullaryExpr(index.size(),
				[this, &index, &dens, &tol](Eigen::Index row){
					return this->Brmin(this->gridpoints[index[row]], this->gridpoints[index[row] + 1], dens, tol);
				});
			// double x;
			// for (auto i = 0; i < ans.size(); ++i){
			// 	this->Brmin(x, gridpoints[index[i]], gridpoints[index[i] + 1], dens);
			// 	ans[i] = x;
			// }
			this->gradfunvec(ans, dens, pointsval, pointsgrad, true, false);
			Eigen::Matrix<Type, Eigen::Dynamic, 1> anstemp = indexing(ans, index2num((pointsval.array() < 0).template cast<int>()));
			ans.lazyAssign(anstemp);
		}
		Type pv2,ps2;
		this->gradfun(this->gridpoints[0], dens, pv2, ps2, true, false);
		if ((pv2 < 0) & (pointsgrad[0] > 0)){
			ans.conservativeResize(ans.size() + 1);
			ans.template tail<1>() = gridpoints.template head<1>();
		}
		this->gradfun(this->gridpoints.template tail<1>()[0], dens, pv2, ps2, true, false);
		if ((pv2 < 0) & (pointsgrad.template tail<1>()[0] < 0)){
			ans.conservativeResize(ans.size() + 1);
			ans.template tail<1>() = gridpoints.template tail<1>();
		}

		return ans;
	}

	Type Dfmin(const Eigen::Matrix<Type, 3, 1> &x1, const Eigen::Matrix<Type, 3, 1> &fx1,
		const Eigen::Matrix<Type, Eigen::Dynamic, 1> &dens, const Type &tol = 1e-6) const{
		Type lb = x1[0], ub = x1[2], newpoint, fnewpoint, dummy;
		Eigen::Matrix<Type, 3, 1> xx(x1), fxx(fx1);

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
			if (std::isnan(newpoint) | (newpoint < lb) | (newpoint > ub)){
				if (std::fabs(xx[0] - xx[2]) < std::fabs(xx[1] - xx[2])){
					newpoint = (xx[1] + xx[2]) / 2.;
				}else{
					newpoint = (xx[0] + xx[2]) / 2.;
				}
			}
			this->gradfun(newpoint, dens, fnewpoint, dummy, true, false);

			if (fnewpoint > fxx[2]){
				// xx[2] still the local min, replacing bounds
				if (newpoint > xx[2]){
					ub = newpoint;
					fxx = (xx.array() > newpoint).select(Eigen::Matrix<Type, 3, 1>::Constant(fnewpoint), fxx);
					xx = (xx.array() > newpoint).select(Eigen::Matrix<Type, 3, 1>::Constant(newpoint), xx);
				}else{
					lb = newpoint;
					fxx = (xx.array() < newpoint).select(Eigen::Matrix<Type, 3, 1>::Constant(fnewpoint), fxx);
					xx = (xx.array() < newpoint).select(Eigen::Matrix<Type, 3, 1>::Constant(newpoint), xx);
				}
			}else{
				// newpoint is the new min, replacing bounds using xx[2]
				if (xx[2] > newpoint){
					ub = xx[2];
					fxx = (xx.array() > xx[2]).select(Eigen::Matrix<Type, 3, 1>::Constant(fxx[2]), fxx);
					xx = (xx.array() > xx[2]).select(Eigen::Matrix<Type, 3, 1>::Constant(xx[2]), xx);
					xx[2] = newpoint;
					fxx[2] = fnewpoint;
				}else{
					lb = xx[2];
					fxx = (xx.array() < xx[2]).select(Eigen::Matrix<Type, 3, 1>::Constant(fxx[2]), fxx);
					xx = (xx.array() < xx[2]).select(Eigen::Matrix<Type, 3, 1>::Constant(xx[2]), xx);
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

	Eigen::Matrix<Type, Eigen::Dynamic, 1> solvegradd0(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &dens, const Type &tol = 1e-6) const{
		Eigen::Matrix<Type, Eigen::Dynamic, 1> pointsval, pointsgrad; // pointsgrad not referenced.
		this->gradfunvec(this->gridpoints, dens, pointsval, pointsgrad, true, false);
		Eigen::Matrix<Type, Eigen::Dynamic, 1> temp = diff_(pointsval);
		const int L = temp.size();
		Eigen::VectorXi index = index2num((temp.head(L - 1).array() < 0).template cast<int>() * (temp.tail(L - 1).array() > 0).template cast<int>());
		Eigen::Matrix<Type, Eigen::Dynamic, 1> ans(index.size());
		if (index.size() > 0){
			ans = Eigen::Matrix<Type, Eigen::Dynamic, 1>::NullaryExpr(index.size(),
				[this, &index, &dens, &pointsval, &tol](Eigen::Index row){
					return this->Dfmin(this->gridpoints.template segment<3>(index[row]), pointsval.template segment<3>(index[row]), dens, tol);
				});
			Eigen::Matrix<Type, Eigen::Dynamic, 1> anstemp = indexing(ans, index2num(1 - ans.array().isNaN().template cast<int>()));
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
		if ((pointsval[0] < 0) & (diff_(pointsval.template head<2>())[0] > 0)){
			ans.conservativeResize(ans.size() + 1);
			ans.template tail<1>() = gridpoints.template head<1>();
		}
		if ((pointsval.template tail<1>()[0] < 0) & (diff_(pointsval.template tail<2>())[0] < 0)){
			ans.conservativeResize(ans.size() + 1);
			ans.template tail<1>() = gridpoints.template tail<1>();			
		}
		return ans;
	}

	Eigen::Matrix<Type, Eigen::Dynamic, 1> solvegrad(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &dens, const Type & tol = 1e-6) const{
		if (this->flag == "d1"){
			return solvegradd1(dens, tol);
		}else{
			return solvegradd0(dens, tol);
		}
	}

	// compute mixing distribution
	void computemixdist(const Type &tol = 1e-6, const int &maxit = 100, const int &verbose = 0) const{
		Eigen::Matrix<Type, Eigen::Dynamic, 1> mu0 = initpt, pi0 = initpr * (1. - pi0fixed.sum());
		// int iter = 0, convergence = 0;
		this->iter = 0;
		Eigen::Matrix<Type, Eigen::Dynamic, 1> newpoints, dens = this->mapping(mu0, pi0);
		Type closs = this->lossfunction(dens), nloss;

		do{
			newpoints.lazyAssign(this->solvegrad(dens, tol));

			mu0.conservativeResize(mu0.size() + newpoints.size());
			pi0.conservativeResize(pi0.size() + newpoints.size());
			mu0.tail(newpoints.size()) = newpoints;
			pi0.tail(newpoints.size()) = Eigen::Matrix<Type, Eigen::Dynamic, 1>::Zero(newpoints.size());
			sortmix(mu0, pi0);

			if (verbose >= 1){
				Rcpp::Rcout<<"Iteration: "<<iter<<" with loss "<<nloss<<std::endl;
				Rcpp::Rcout<<"support points: "<<mu0.transpose()<<std::endl;
				Rcpp::Rcout<<"probabilities: "<<pi0.transpose()<<std::endl;
			}
			
			if (verbose >= 2){
				Rcpp::Rcout<<"new points: "<<newpoints.transpose()<<std::endl;
				Eigen::Matrix<Type, Eigen::Dynamic, 1> pointsval, pointsgrad;
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
			this->iter++;
			dens = this->mapping(mu0, pi0);
			nloss = this->lossfunction(dens);

			if (closs - nloss < tol){
				this->convergence = 0;
				break;
			}

			if (this->iter > maxit){
				this->convergence = 1;
				break;
			}

			closs = nloss;
		}while(true);

		this->resultpt.resize(mu0.size());
		this->resultpr.resize(pi0.size());
		this->resultpt = mu0;
		this->resultpr = pi0;
	}

	Rcpp::List get_ans() const{
		Eigen::Matrix<Type, Eigen::Dynamic, 1> mu0new(this->resultpt.size() + mu0fixed.size());
		Eigen::Matrix<Type, Eigen::Dynamic, 1> pi0new(this->resultpr.size() + pi0fixed.size());
		mu0new.head(this->resultpt.size()) = this->resultpt; mu0new.tail(mu0fixed.size()) = mu0fixed;
		pi0new.head(this->resultpr.size()) = this->resultpr; pi0new.tail(pi0fixed.size()) = pi0fixed;

		sortmix(mu0new, pi0new);
		Eigen::Matrix<Type, Eigen::Dynamic, 1> maxgrad, maxgrad2;
		this->gradfunvec(this->resultpt, this->mapping(this->resultpt, this->resultpr), maxgrad, maxgrad2, true, false);

		return Rcpp::List::create(Rcpp::Named("iter") = iter,
			Rcpp::Named("family") = family,
			Rcpp::Named("min.gradient") = maxgrad.minCoeff(),
			Rcpp::Named("beta") = this->beta,
			Rcpp::Named("mix") = Rcpp::List::create(Rcpp::Named("pt") = mu0new, Rcpp::Named("pr") = pi0new),
			Rcpp::Named("ll") = this->lossfunction(this->mapping(this->resultpt, this->resultpr)) + this->extrafun(),
			Rcpp::Named("flag") = this->flag,
			Rcpp::Named("convergence") = this->convergence);
	}

	void estpi0(const Type &val, const Type &tol = 1e-6, const int &verbose = 0) const{
		// this is only suitable for estimate the probability of location at 0.
		this->mu0fixed.resize(1);
		this->mu0fixed.setZero();
		this->pi0fixed.resize(1);
		this->pi0fixed.setZero();
		this->setprecompute();
		this->computemixdist(tol);
		Type minloss = this->lossfunction(this->mapping(this->resultpt, this->resultpr)) + this->extrafun(), ll;
		Eigen::Matrix<Type, Eigen::Dynamic, 1> dens = this->mapping(Eigen::Matrix<Type, 1, 1>::Zero(1), Eigen::Matrix<Type, 1, 1>::Ones(1));
		if (this->hypofun(this->lossfunction(dens) + this->extrafun(), minloss) < val){
			this->resultpt.resize(1);
			this->resultpr.resize(1);
			this->resultpt[0] = 0;
			this->resultpr[0] = 1;
		}else{
			Type lb = 0, ub = 1, flb = minloss, fub = this->lossfunction(dens) + this->extrafun();
			// Rcpp::List mix = this->result["mix"];
			// Rcpp::NumericVector tmu0 = mix["pt"], tpi0 = mix["pr"];
			// Eigen::Map<Eigen::VectorXd> mu1(tmu0.begin(), tmu0.length());
			// Eigen::Map<Eigen::VectorXd> pi1(tpi0.begin(), tpi0.length());	
			Type sp = this->familydensity(0, this->resultpt, this->resultpr) / this->familydensity(0, Eigen::Matrix<Type, 1, 1>::Zero(1), Eigen::Matrix<Type, 1, 1>::Ones(1));
			this->pi0fixed.setConstant(sp);
			this->setprecompute();
			this->computemixdist();
			ll = this->lossfunction(this->mapping(this->resultpt, this->resultpr)) + this->extrafun();
			int iter = 1;
			Eigen::Matrix<Type, 3, 3> A;
			Eigen::Matrix<Type, 3, 1> b, x1;
			while ((std::fabs(this->hypofun(ll, minloss) - val) > tol) & (std::fabs(ub - lb) > tol)){
				if (verbose >= 1){
					Rcpp::Rcout<<"Iter: "<<iter<<" lower: "<<lb<<" upper: "<<ub<<std::endl;
					Rcpp::Rcout<<"current val: "<<sp<<" fval: "<<ll<<std::endl;
				}

				if ((this->hypofun(ll, minloss) - val < 0) & (sp > lb)){
					lb = sp; flb = ll;
				}
				if ((this->hypofun(ll, minloss) - val > 0) & (sp < ub)){
					ub = sp; fub = ll;
				}
				sp = (lb + ub) / 2;

				this->initpt.resize(this->resultpt.size());
				this->initpt = this->resultpt;
				this->initpr.resize(this->resultpr.size());
				this->initpr = this->resultpr;
				this->pi0fixed.setConstant(sp);
				this->setprecompute();
				this->computemixdist();
				ll = this->lossfunction(this->mapping(this->resultpt, this->resultpr)) + this->extrafun();

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

				this->initpt.resize(this->resultpt.size());
				this->initpt = this->resultpt;
				this->initpr.resize(this->resultpr.size());
				this->initpr = this->resultpr;
				this->pi0fixed.setConstant(sp);
				this->setprecompute();
				this->computemixdist();
				ll = this->lossfunction(this->mapping(this->resultpt, this->resultpr)) + this->extrafun();
				iter++;
			}
		}
	}

	// functions to each specific type

	virtual Eigen::Matrix<Type, Eigen::Dynamic, 1> mapping(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &mu0, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &pi0) const{
		return Eigen::Matrix<Type, 1, 1>::Zero(1);
	}

	virtual Type lossfunction(const Eigen::Matrix<Type, Eigen::Dynamic, 1> & maps) const{
		return 0;
	}

	virtual void gradfun(const Type &mu, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &dens,
		Type &ansd0, Type & ansd1, const bool &d0, const bool &d1) const{}

	// vectorised function for gradfund1 (can be overwriten)
	virtual void gradfunvec(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &mu, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &dens, 
		Eigen::Matrix<Type, Eigen::Dynamic, 1> &ansd0, Eigen::Matrix<Type, Eigen::Dynamic, 1> &ansd1, const bool &d0, const bool& d1) const{
		ansd0.resize(mu.size());
		ansd1.resize(mu.size());
		Type *ansd0ptr = ansd0.data(), *ansd1ptr = ansd1.data();
		for (auto muptr = mu.data(); muptr < mu.data() + mu.size(); muptr++, ansd0ptr++, ansd1ptr++){
			this->gradfun(*muptr, dens, *ansd0ptr, *ansd1ptr, d0, d1);
		}
	}

	virtual void computeweights(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &mu0, 
		Eigen::Matrix<Type, Eigen::Dynamic, 1> &pi0, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &dens) const{}

	virtual Type extrafun() const{
		return 0;
	}

	virtual Type hypofun(const Type &ll, const Type &minloss) const{
		// likelihood methods use ll - minloss
		// distance methods use ll.
		return 0;
	}

	virtual Type familydensity(const Type &x, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &mu0, const Eigen::Matrix<Type, Eigen::Dynamic, 1> &pi0) const{
		return 0;
	}

};

// begin of 2D implementation

template<class Type>
inline void simplifymix2D(Eigen::Matrix<Type, Eigen::Dynamic, 2>& mu0, 
	Eigen::Matrix<Type, Eigen::Dynamic, 1> & pi0){
	if (mu0.rows() != 1) {
		Eigen::VectorXi index = index2num((pi0.array().abs() > 1e-14).template cast<int>());
		Eigen::Matrix<Type, Eigen::Dynamic, 2> mu0new = indexing(mu0, index, Eigen::VectorXi::LinSpaced(2, 0, 1));
		Eigen::Matrix<Type, Eigen::Dynamic, 1> pi0new = indexing(pi0, index);
		pi0.lazyAssign(pi0new);
		mu0.lazyAssign(mu0new);
	}
}

// This function collpse the mixing distribution
template<class Type>
inline void collapsemix2D(Eigen::Matrix<Type, Eigen::Dynamic, 2> &mu0, Eigen::Matrix<Type, Eigen::Dynamic, 1> &pi0, const Type &prec){
	bool foo;
	if (mu0.rows() > 1){
		Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> distmat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>::NullaryExpr(mu0.rows(), mu0.rows(), [&mu0](Eigen::Index i, Eigen::Index j){
			return (mu0.row(i) - mu0.row(j)).norm();
		});
		distmat.diagonal() = Eigen::Matrix<Type, Eigen::Dynamic, 1>::Constant(mu0.rows(), std::numeric_limits<Type>::infinity());
		int mini, minj;
		foo = (distmat.array() <= prec).any();
		Type temp;
		while (foo){
			distmat.minCoeff(&mini, &minj);
			temp = pi0[mini] + pi0[minj];
			mu0.row(mini) = (mu0.row(mini) * pi0[mini] + mu0.row(minj) * pi0[minj]) / temp;
			pi0[minj] = 0;
			pi0[mini] = temp;
			simplifymix2D(mu0, pi0);
			if (mu0.rows() <= 1) {
				foo = false;
			}else{
				distmat.lazyAssign(Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>::NullaryExpr(mu0.rows(), mu0.rows(), [&mu0](Eigen::Index i, Eigen::Index j){
					return (mu0.row(i) - mu0.row(j)).norm();
				}));
				distmat.diagonal() = Eigen::Matrix<Type, Eigen::Dynamic, 1>::Constant(mu0.rows(), std::numeric_limits<Type>::infinity());
				foo = (distmat.array() <= prec).any();
			}
		}
	}
}

#endif