// Implementation of Qi and Sun (2006) in C++ using Eigen library from R's implementation
// Authors can be obtained on https://www.polyu.edu.hk/ama/profile/dfsun/
// Xiangjie Xue translate the R code into C++ using Eigen library.

#include <RcppEigen.h>
#include <unsupported/Eigen/IterativeSolvers>
#include "../inst/include/miscfuns.h"

void MyEigen(const Eigen::MatrixXd &X, Eigen::MatrixXd &eigvec, Eigen::VectorXd &eigval){
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig;
	eig.compute(X);
	eigvec.resize(X.rows(), X.cols());
	eigvec = eig.eigenvectors().rowwise().reverse();
	eigval.resize(X.rows());
	eigval = eig.eigenvalues().reverse();
}

void Corrsub_gradient(const Eigen::VectorXd &y, const Eigen::VectorXd &lambda, const Eigen::MatrixXd &P,
	const Eigen::VectorXd &b, const int & n, double &f, Eigen::VectorXd &Fy){
	int r = (lambda.array() > 0).count();
	if (r > 0){
		Fy = P.leftCols(r).cwiseAbs2() * lambda.head(r);
		f = 0.5 * lambda.head(r).squaredNorm() - b.dot(y);
	}else{
		Fy.resize(n);
		Fy.setZero();
		f = 0.0;
	}
}

void PCA(Eigen::MatrixXd & X, const Eigen::VectorXd &lambda, const Eigen::MatrixXd &P, const Eigen::VectorXd &b, const int &n){
	if (n == 1){
		X.resize(b.size(), 1);
		X.leftCols<1>() = b;
	}else{
		int r = (lambda.array() > 0).count();
		if (r > 1 & r < n){
			if (r <= 2){
				X = P.leftCols(r) * lambda.head(r).asDiagonal() * P.leftCols(r).transpose();
			}else{
				X += P.rightCols(n - r) * lambda.tail(n - r).cwiseAbs().asDiagonal() * P.rightCols(n - r).transpose();
			}
		}else{
			if (r == 0){
				X.resize(n, n);
				X.setZero();
			}else{
				if (r == 1)
					X = lambda[0] * P.leftCols<1>() * P.leftCols<1>().transpose();
			}
		}
		Eigen::VectorXd d = X.diagonal().cwiseMax(b);
		X.diagonal() = d;
		d = b.cwiseQuotient(d).cwiseSqrt();
		X = X.cwiseProduct(d * d.transpose());
	}
}

void omega_mat(const Eigen::VectorXd &lambda, const int & n, Eigen::MatrixXd & Omega12){
	int r = (lambda.array() > 0).count();
	if (r > 0){
		if (r < n){
			Omega12.resize(r, n - r);
			Omega12 = lambda.head(r).replicate(1, n - r);
			Omega12 = Omega12.cwiseQuotient(Omega12.rowwise() - lambda.tail(n - r).transpose());
		}else{
			Omega12.resize(n, n);
			Omega12.setOnes();
		}
	}else{
		Omega12.resize(0, 0);
	}
}

void Jacobian_matrix(const Eigen::VectorXd &d, const Eigen::MatrixXd &Omega12, const Eigen::MatrixXd &P, const int &n, Eigen::VectorXd &Vd){
	int r = Omega12.rows();
	Vd.resize(n);
	Vd.setZero();
	if (r > 0){
		if (r < n){
			Eigen::VectorXd hh = (P.leftCols(r) * Omega12.cwiseProduct(P.leftCols(r).transpose() * d.asDiagonal() * P.rightCols(n - r))).cwiseProduct(P.rightCols(n - r)).rowwise().sum() * 2.;
			if ((double) r < (double) n / 2.){
				Vd = (P.leftCols(r) * P.leftCols(r).transpose()).cwiseAbs2() * d + hh + 1e-10 * d;
			}else{
				Eigen::MatrixXd PP2 = P.rightCols(n - r) * P.rightCols(n - r).transpose();
				Vd = d + PP2.cwiseAbs2() * d + hh - 2. * d.cwiseProduct(PP2.diagonal()) + 1e-10 * d;
			}
		}else{
			Vd = (1. + 1e-10) * d;
		}
	}
}

void precond_matrix(const Eigen::MatrixXd &Omega12, const Eigen::MatrixXd &P, const int &n, Eigen::VectorXd &c){
	int r = Omega12.rows();
	c.resize(n);
	c.setOnes();

	if (r > 1){
		Eigen::MatrixXd H = P.cwiseAbs2().transpose();
		if ((double) r < (double) n / 2.){
			c = H.topRows(r).colwise().sum().cwiseAbs2().transpose() + 2. * (H.topRows(r).transpose() * Omega12).cwiseProduct(H.bottomRows(n - r).transpose()).rowwise().sum();
		}else{
			if (r < n){
				c = (H.colwise().sum().cwiseAbs2() - H.bottomRows(n - r).colwise().sum().cwiseAbs2() - 2.* H.topRows(r).cwiseProduct((Eigen::MatrixXd::Ones(Omega12.rows(), Omega12.cols()) - Omega12) * H.bottomRows(n - r)).colwise().sum()).transpose();
			}
		}
	}

	c = c.cwiseMax(1e-8);
}

void pre_cg(const Eigen::VectorXd &b, const double &tol, const int & iter_CG, const Eigen::VectorXd & c, 
	const Eigen::MatrixXd & Omega12, const Eigen::MatrixXd &P, const int &n, Eigen::VectorXd &p){
	Eigen::VectorXd r(b);
	double n2b = b.norm(), tolb = tol * n2b;
	p.resize(n);
	p.setZero();
	Eigen::VectorXd z = r.cwiseQuotient(c);
	double rz1 = r.dot(z), rz2 = 1.;
	Eigen::VectorXd d(z);

	Eigen::VectorXd w;
	double denom;

	for (int k = 1; k <= iter_CG; ++k){
		if (k > 1){
			d = z + (rz1 / rz2) * d;
		}
		Jacobian_matrix(d, Omega12, P, n, w);
		denom = d.dot(w);
		if (denom <= 0){
			p = d / d.norm();
			break;
		}else{
			p += (rz1 / denom) * d;
			r -= (rz1 / denom) * w;
		}
		z = r.cwiseQuotient(c);
		if (r.norm() <= tolb){
			break;
		}
		rz2 = rz1; rz1 = r.dot(z);
	}
}

// [[Rcpp::export]]
Eigen::MatrixXd correlationmatrixcpp(const Eigen::MatrixXd &G1, const double tau = 0, const double tol = 1e-6){
	int n = G1.rows();
	Eigen::MatrixXd G(G1);
	Eigen::VectorXd b(n);
	b.setOnes();
	if (tau > 0){
		b.array() -= tau;
		G.diagonal().array() -= tau;
	}

	Eigen::VectorXd b0(b);
	double error_tol = std::max(tol, 1e-12);

	Eigen::VectorXd y(n), Fy(n);
	y.setZero(); Fy.setZero();

	int Iter_whole = 200, Iter_inner = 20, Iter_CG = 200, iter_k = 0;
	double f_eval = 0, tol_CG = 1e-2, G_1 = 1e-4;

	Eigen::VectorXd x0(y), c(n), d(n);
	c.setOnes(), d.setZero();
	double val_G = G.squaredNorm() * 0.5;

	Eigen::MatrixXd X(G);
	X.diagonal() += y;
	X = (X + X.transpose()) / 2.;

	Eigen::MatrixXd P;
	Eigen::VectorXd lambda;
	MyEigen(X, P, lambda);

	double f0;
	Corrsub_gradient(y, lambda, P, b0, n, f0, Fy);
	double val_dual = val_G - f0;
	PCA(X, lambda, P, b0, n);
	double val_obj = (X - G).squaredNorm() * 0.5, gap = (val_obj - val_dual) / (1. + std::fabs(val_dual) + std::fabs(val_obj));
	double f = f0;
	f_eval++;
	b = b0 - Fy;

	double norm_b = b.norm(), norm_b0 = b0.norm() + 1., norm_b_rel = norm_b / norm_b0;
	Eigen::MatrixXd Omega12;
	omega_mat(lambda, n, Omega12);
	x0 = y;

	double slope;
	int k_inner;
	while (gap > error_tol & norm_b_rel > error_tol & iter_k < Iter_whole){
		precond_matrix(Omega12, P, n, c);
		pre_cg(b, tol_CG, Iter_CG, c, Omega12, P, n, d);
		slope = d.dot(Fy - b0);
		y = x0 + d;
		X = G;
		X.diagonal() += y;
		X = (X + X.transpose()) / 2.;
		MyEigen(X, P, lambda);
		Corrsub_gradient(y, lambda, P, b0, n, f, Fy);
		k_inner = 0;

		while ((k_inner <= Iter_inner) & (f > f0 + G_1 * std::pow(.5, (double) k_inner) * slope + 1e-6)){
			k_inner++;
			y = x0 + std::pow(0.5, (double) k_inner) * d;
			X = G;
			X.diagonal() += y;
			X = (X + X.transpose()) * 0.5;
			MyEigen(X, P, lambda);
			Corrsub_gradient(y, lambda, P, b0, n, f, Fy);
		}

		f_eval += k_inner + 1;
		x0 = y; f0 = f;
		val_dual = val_G - f0;
		PCA(X, lambda, P, b0, n);
		val_obj = (X - G).squaredNorm() * 0.5;
		gap = (val_obj - val_dual) / (1. + std::fabs(val_dual) + std::fabs(val_obj));

		iter_k++;
		b = b0 - Fy;
		norm_b = b.norm();
		norm_b_rel = norm_b / norm_b0;

		omega_mat(lambda, n, Omega12); 
	}

	X.diagonal().array() += tau;

	return X;
}
