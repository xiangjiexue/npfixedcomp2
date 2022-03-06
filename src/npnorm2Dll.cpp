// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include "../inst/include/npfixedcomp.h"
#include "../inst/include/densityND.h"
#include "../inst/include/LBFGSB.h"

// [[Rcpp::depends(RcppEigen)]]

template<class Type>
class npnorm2Dll
{
public:
	const Eigen::Matrix<Type, Eigen::Dynamic, 2> data;
	mutable Eigen::Matrix<Type, Eigen::Dynamic, 2> mu0fixed;
	mutable Eigen::VectorX<Type> pi0fixed;
	const Eigen::Matrix<Type, 2, 2> beta;
	// pass it from R for now
	mutable Eigen::Matrix<Type, Eigen::Dynamic, 2> initpt;
	mutable Eigen::VectorX<Type> initpr;
	const Eigen::Matrix<Type, Eigen::Dynamic, 2> gridpoints;
	int len;
	mutable int convergence, iter = 0;
	mutable Rcpp::List result;
	std::string family, flag;
	mutable Eigen::VectorX<Type> precompute;
	mutable Eigen::VectorX<Type> dens;
	mutable Eigen::Matrix<Type, Eigen::Dynamic, 2> resultpt;
	mutable Eigen::VectorX<Type> resultpr;

	npnorm2Dll(const Eigen::Matrix<Type, Eigen::Dynamic, 2> &data_, const Eigen::Matrix<Type, Eigen::Dynamic, 2> &mu0fixed_, 
		const Eigen::VectorX<Type> &pi0fixed_, const Eigen::Matrix<Type, 2, 2> &beta_, const Eigen::Matrix<Type, Eigen::Dynamic, 2> &initpt_,
		const Eigen::VectorX<Type> &initpr_, const Eigen::Matrix<Type, Eigen::Dynamic, 2> &gridpoints_) : data(data_),
		mu0fixed(mu0fixed_), pi0fixed(pi0fixed_), beta(beta_), initpt(initpt_), initpr(initpr_), gridpoints(gridpoints_) {
		this->len = data.rows();
		this->setprecompute();
		this->family = "npnorm2D";
		this->flag = "d1";
	}

	void setprecompute() const{
		this->precompute.resize(this->len);
		this->precompute = mapping(this->mu0fixed, this->pi0fixed);
	}

	void setdens(const Eigen::VectorX<Type> & dens) const{
		this->dens.resize(dens.size());
		this->dens = dens;
	}

	Type lossfunction(const Eigen::VectorX<Type> &maps) const{
		return (maps + this->precompute).array().log().sum() * -1.;
	}

	Eigen::VectorX<Type> mapping(const Eigen::Matrix<Type, Eigen::Dynamic, 2> &mu0, const Eigen::VectorX<Type> &pi0) const{
		return dnpnormND_(this->data, mu0, pi0, this->beta);
	}

	void gradfun(const Eigen::Matrix<Type, 1, 2> &mu, const Eigen::VectorX<Type> &dens, 
		Type &ansd0, Eigen::Matrix<Type, 1, 2> & ansd1, const bool &d0, const bool &d1) const{
		Eigen::VectorX<Type> fullden = (dens + this->precompute).cwiseInverse();
		Eigen::Matrix<Type, 1, 1> scale(1, 1);
		scale(0, 0) = 1. - this->pi0fixed.sum();
		Eigen::VectorX<Type> temp = dnpnormND_(this->data, mu, scale, this->beta);
		if (d0)
			ansd0 = (dens - temp).dot(fullden);
		if (d1)
			ansd1 = temp.transpose() * (mu.replicate(this->len, 1) - this->data) * this->beta.inverse();
	}

	void gradfunvec(const Eigen::Matrix<Type, Eigen::Dynamic, 2> &mu, const Eigen::VectorX<Type> &dens,
		Eigen::VectorX<Type> &ansd0, Eigen::Matrix<Type, Eigen::Dynamic, 2> &ansd1,
		const bool &d0, const bool &d1) const{
		ansd0.resize(mu.rows());
		ansd1.resize(mu.rows(), 2);
		Eigen::Matrix<Type, 1, 2> temp_mu(1, 2), temp_ansd1(1, 2);
		for (int i = 0; i < mu.rows(); ++i){
			temp_mu = mu.row(i);
			this->gradfun(temp_mu, dens, ansd0[i], temp_ansd1, d0, d1);
			ansd1.row(i) = temp_ansd1;
		}
	}

	void checklossfun2(const Eigen::VectorX<Type> &diff, Eigen::VectorX<Type> &pi0, 
		const Eigen::VectorX<Type> &eta, const Eigen::VectorX<Type> &p, 
		const Eigen::VectorX<Type> &dens) const{
		Type llorigin = this->lossfunction(dens);
		Type sigma = 2., alpha = 0.3333;
		Type con = - p.dot(eta);
		Type lhs, rhs;
		Eigen::VectorX<Type> ans(pi0);
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
	void collapse(Eigen::Matrix<Type, Eigen::Dynamic, 2> & mu0, Eigen::VectorX<Type> & pi0, const Type &tol = 1e-6) const{
		Type ll = this->lossfunction(this->mapping(mu0, pi0)), nll;
		Type ntol = std::max(tol * 0.1, ll * 1e-16), prec;
		Eigen::Matrix<Type, Eigen::Dynamic, 2> mu0new = mu0;
		Eigen::VectorX<Type> pi0new = pi0;
		do {
			if (mu0.rows() <= 1) break;
			Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> distmat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>::NullaryExpr(mu0.rows(), mu0.rows(), [&mu0](Eigen::Index i, Eigen::Index j){
				return (mu0.row(i) - mu0.row(j)).norm();
			});
			distmat.diagonal() = Eigen::VectorX<Type>::Constant(mu0.rows(), std::numeric_limits<Type>::infinity());
			prec = 10 * distmat.minCoeff();
			collapsemix2D(mu0new, pi0new, prec);
			nll = this->lossfunction(this->mapping(mu0new, pi0new));
			if (nll <= ll + ntol){
				pi0 = pi0new;
				mu0 = mu0new;
			}else{break;}
		} while(true);
		simplifymix2D(mu0, pi0);
	}

	void computeweights(const Eigen::Matrix<Type, Eigen::Dynamic, 2> &mu0, Eigen::VectorX<Type> &pi0, const Eigen::VectorX<Type> &dens) const{
		Eigen::VectorX<Type> fp = dens + this->precompute;
		Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> sp = dnormNDarray(this->data, mu0, this->beta), tp = sp.array().colwise() / fp.array();
		Eigen::VectorX<Type> nw(pi0.size());
		if (this->len > 1e3){
			// maybe switch to pnnqp?
			nw = pnnqp_(tp.transpose() * tp, tp.transpose() * (this->precompute.cwiseQuotient(fp) - Eigen::VectorX<Type>::Constant(this->len, 2)), 1. - this->pi0fixed.sum());
		}else{
			nw = pnnlssum_(tp, Eigen::VectorX<Type>::Constant(this->len, 2) - this->precompute.cwiseQuotient(fp), 1. - this->pi0fixed.sum());
		}
		this->checklossfun2(sp * nw - dens, pi0, nw - pi0, tp.colwise().sum(), dens);
	}

	Type operator()(const Eigen::VectorX<Type>& x, Eigen::VectorX<Type>& grad) const{
		Type ansd0;
		Eigen::Matrix<Type, 1, 2> grad1 = grad.derived();
		this->gradfun(x, this->dens, ansd0, grad1, true, true);
		grad = grad1.transpose();
		return ansd0;
	}

	Eigen::Matrix<Type, Eigen::Dynamic, 2> solvegrad(const Eigen::VectorX<Type> &dens, const Type &tol = 1e-6) const{
		Eigen::Matrix<Type, Eigen::Dynamic, 2> ans(0, 2); // pointsgrad not referenced.
		Type fval;
		Eigen::VectorX<Type> xval(2, 1), lb(2, 1), ub(2, 1);
		LBFGSpp::LBFGSBParam<Type> param;
		param.epsilon = tol;
		param.max_linesearch = 100;
		param.max_iterations = 100;
    	LBFGSpp::LBFGSBSolver<Type> solver(param);
    	this->setdens(dens);
		for (int i = 0; i < this->gridpoints.rows() - 1; ++i)
			for (int j = 0; j < this->gridpoints.rows() - 1; ++j){
				// check this approximation
				// double gradapprox = (this->gridpoints.maxCoeff() - this->gridpoints.minCoeff()) * this->beta.maxCoeff();
				// Eigen::Matrix<Type, 1, 2> temp(1, 2), dumm(1, 2);
				// double gp0, gp1, gp2, gp3;
				// temp << this->gridpoints(i, 0), this->gridpoints(j, 1);
				// this->gradfun(temp, dens, gp0, dumm, true, false);
				// temp << this->gridpoints(i, 0), this->gridpoints(j + 1, 1);
				// this->gradfun(temp, dens, gp1, dumm, true, false);
				// temp << this->gridpoints(i + 1, 0), this->gridpoints(j, 1);
				// this->gradfun(temp, dens, gp2, dumm, true, false);
				// temp << this->gridpoints(i + 1, 0), this->gridpoints(j + 1, 1);
				// this->gradfun(temp, dens, gp3, dumm, true, false);
				// if (gp0 > gradapprox | gp1 > gradapprox | gp2 > gradapprox | gp3 > gradapprox){continue;}

				lb << this->gridpoints(i, 0), this->gridpoints(j, 1);
				ub << this->gridpoints(i + 1, 0), this->gridpoints(j + 1, 1);
				xval = (lb + ub) * .5;
				// Rcpp::Rcout<<lb.transpose()<<"    "<<ub.transpose()<<"     "<<xval.transpose()<<"     ";
				solver.minimize(*this, xval, fval, lb, ub);
				// Rcpp::Rcout<<xval.transpose()<<" "<<iter<<" "<<fval<<std::endl;
				if (fval < 0){
					ans.conservativeResize(ans.rows() + 1, 2);
					ans.bottomRows(1) = xval.transpose();
				}
			}
		return ans;
	}

	// compute mixing distribution
	void computemixdist(const Type &tol = 1e-6, const int &maxit = 100, const int &verbose = 0) const{
		Eigen::Matrix<Type, Eigen::Dynamic, 2> mu0 = initpt;
		Eigen::VectorX<Type> pi0 = initpr * (1. - pi0fixed.sum());
		// int iter = 0, convergence = 0;
		this->iter = 0;
		Eigen::Matrix<Type, Eigen::Dynamic, 2> newpoints;
		Eigen::VectorX<Type> dens = this->mapping(mu0, pi0);
		Type closs = this->lossfunction(dens), nloss;

		do{
			newpoints = this->solvegrad(dens, tol);

			mu0.conservativeResize(mu0.rows() + newpoints.rows(), 2);
			pi0.conservativeResize(pi0.size() + newpoints.rows());
			mu0.bottomRows(newpoints.rows()) = newpoints;
			pi0.tail(newpoints.rows()) = Eigen::VectorX<Type>::Zero(newpoints.rows());

			if (verbose >= 1){
				Rcpp::Rcout<<"Iteration: "<<iter<<" with loss "<<nloss<<std::endl;
				Rcpp::Rcout<<"support points: "<<std::endl<<mu0.transpose()<<std::endl;
				Rcpp::Rcout<<"probabilities: "<<pi0.transpose()<<std::endl;
			}
			
			if (verbose >= 2){
				Rcpp::Rcout<<"new points: "<<std::endl<<newpoints.transpose()<<std::endl;
				Eigen::VectorX<Type> pointsval;
				Eigen::Matrix<Type, Eigen::Dynamic, 2> pointsgrad;
				this->gradfunvec(newpoints, dens, pointsval, pointsgrad, true, true);
				Rcpp::Rcout<<"gradient: "<<pointsval.transpose()<<std::endl;
				Rcpp::Rcout<<"loss:"<<this->lossfunction(this->mapping(mu0, pi0))<<std::endl;
				
			}

			this->computeweights(mu0, pi0, dens); // return using mu0, pi0? remember to sort
			if (verbose >= 2){
				Rcpp::Rcout<<"After computeweights"<<std::endl;
				Rcpp::Rcout<<"support points: "<<std::endl<<mu0.transpose()<<std::endl;
				Rcpp::Rcout<<"probabilities: "<<pi0.transpose()<<std::endl;
				Rcpp::Rcout<<"loss:"<<this->lossfunction(this->mapping(mu0, pi0))<<std::endl;
			}
			this->collapse(mu0, pi0);
			if (verbose >= 2){
				Rcpp::Rcout<<"After collapse"<<std::endl;
				Rcpp::Rcout<<"support points: "<<std::endl<<mu0.transpose()<<std::endl;
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

		this->resultpt.resize(mu0.rows(), 2);
		this->resultpr.resize(pi0.size());
		this->resultpt = mu0;
		this->resultpr = pi0;
	}

	Rcpp::List get_ans() const{
		Eigen::Matrix<Type, Eigen::Dynamic, 2> mu0new(this->resultpt.rows() + mu0fixed.rows(), 2);
		Eigen::VectorX<Type> pi0new(this->resultpr.size() + pi0fixed.size());
		mu0new.topRows(this->resultpt.rows()) = this->resultpt; mu0new.bottomRows(mu0fixed.rows()) = mu0fixed;
		pi0new.head(this->resultpr.size()) = this->resultpr; pi0new.tail(pi0fixed.size()) = pi0fixed;

		Eigen::VectorX<Type> maxgrad;
		Eigen::Matrix<Type, Eigen::Dynamic, 2> maxgrad2;
		this->gradfunvec(this->resultpt, this->mapping(this->resultpt, this->resultpr), maxgrad, maxgrad2, true, false);

		return Rcpp::List::create(Rcpp::Named("iter") = iter,
			Rcpp::Named("family") = family,
			Rcpp::Named("min.gradient") = maxgrad.minCoeff(),
			Rcpp::Named("beta") = this->beta,
			Rcpp::Named("mix") = Rcpp::List::create(Rcpp::Named("pt") = mu0new, Rcpp::Named("pr") = pi0new),
			Rcpp::Named("ll") = this->lossfunction(this->mapping(this->resultpt, this->resultpr)),
			Rcpp::Named("flag") = this->flag,
			Rcpp::Named("convergence") = this->convergence);
	}

};

// [[Rcpp::export]]
Rcpp::List npnorm2Dll_(const Eigen::MatrixXd &data, const Eigen::MatrixXd &mu0fixed, const Eigen::VectorXd &pi0fixed,
	const Eigen::MatrixXd &beta, const Eigen::MatrixXd &initpt, const Eigen::VectorXd &initpr, const Eigen::MatrixXd &gridpoints,
	const double &tol = 1e-6, const int &maxit = 100, const int &verbose = 0){
	npnorm2Dll<double> f(data, mu0fixed, pi0fixed, beta, initpt, initpr, gridpoints);
	f.computemixdist(tol, maxit, verbose);
	return f.get_ans();
	// return f.solvegrad(f.mapping(f.initpt, f.initpr));
}
