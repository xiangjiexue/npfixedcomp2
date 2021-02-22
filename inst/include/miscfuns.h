#ifndef miscfuns_h
#define miscfuns_h
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

extern "C" void pnnls_(double* A, int* MDA, int* M, int* N, double* B, double* X, double* RNORM, double* W, double* ZZ, int* INDEX, int* MODE, int* K);

// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>
#include <unsupported/Eigen/SpecialFunctions>

// diff function
// inline Eigen::VectorXd diff_(const Eigen::VectorXd & x){
// 	return x.tail(x.size() - 1) - x.head(x.size() - 1);
// }

namespace Eigen{

// begin diff

template<class ArgType>
struct diff_struct {
  typedef Matrix<typename ArgType::Scalar,
                 (ArgType::SizeAtCompileTime > 0) ? ArgType::SizeAtCompileTime - 1 : -1,
                 1,
                 ColMajor,
                 (ArgType::MaxSizeAtCompileTime > 0) ? ArgType::MaxSizeAtCompileTime - 1 : -1,
                 1> MatrixType;
};

template <class ArgType>
inline typename diff_struct<ArgType>::MatrixType
diff_(const MatrixBase<ArgType>& x)
{
	const int L = x.size() - 1;
	return x.tail(L) - x.head(L);
}

// end diff

// begin log1mexp

template<class ArgType>
struct log1mexp_struct {
  typedef Matrix<typename ScalarBinaryOpTraits<typename ArgType::Scalar, double>::ReturnType,
                 ArgType::RowsAtCompileTime,
                 ArgType::ColsAtCompileTime,
                 ColMajor,
                 ArgType::MaxRowsAtCompileTime,
                 ArgType::MaxColsAtCompileTime> MatrixType;
};

template<class ArgType>
class log1mexp_functor {
	const ArgType &x;
public:
	log1mexp_functor(const ArgType& x_) : x(x_) {}

	const double operator() (Index row, Index col) const {
		return (x.coeff(row, col) < -M_LN2) ? std::log1p(std::exp(x.coeff(row, col)) * -1.) : std::log(-1. * std::expm1(x.coeff(row, col)));
	}
};


template <class ArgType>
CwiseNullaryOp<log1mexp_functor<ArgType>, typename log1mexp_struct<ArgType>::MatrixType>
log1mexp(const MatrixBase<ArgType> &x)
{
	typedef typename log1mexp_struct<ArgType>::MatrixType MatrixType;
	return MatrixType::NullaryExpr(x.rows(), x.cols(), log1mexp_functor<ArgType>(x.derived()));
}

// end log1mexp

// begin logspaceadd/sub

template<class ArgType, class ArgType2>
struct logspace_struct {
  typedef Matrix<typename ScalarBinaryOpTraits<typename ScalarBinaryOpTraits<typename ArgType::Scalar, typename ArgType2::Scalar>::ReturnType, double>::ReturnType,
                 ArgType::SizeAtCompileTime,
                 ArgType::SizeAtCompileTime,
                 ColMajor,
                 ArgType::MaxSizeAtCompileTime,
                 ArgType::MaxSizeAtCompileTime> MatrixType;
};

template<class ArgType, class ArgType2>
class logspaceadd_functor {
	const ArgType &lx;
	const ArgType2 &ly;
public:
	logspaceadd_functor(const ArgType& lx_, const ArgType2 &ly_) : lx(lx_), ly(ly_) {}

	const double operator() (Index row, Index col) const {
		return R::logspace_add(lx.coeff(row, col), ly.coeff(row, col));
	}
};


template <class ArgType, class ArgType2>
CwiseNullaryOp<logspaceadd_functor<ArgType, ArgType2>, typename logspace_struct<ArgType, ArgType2>::MatrixType>
logspaceadd(const MatrixBase<ArgType> &lx, const MatrixBase<ArgType2> &ly)
{
	EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(ArgType, ArgType2);
	typedef typename logspace_struct<ArgType, ArgType2>::MatrixType MatrixType;
	return MatrixType::NullaryExpr(lx.rows(), lx.cols(), logspaceadd_functor<ArgType, ArgType2>(lx.derived(), ly.derived()));
}

template<class ArgType, class ArgType2>
class logspacesub_functor {
	const ArgType &lx;
	const ArgType2 &ly;
public:
	logspacesub_functor(const ArgType& lx_, const ArgType2 &ly_) : lx(lx_), ly(ly_) {}

	const double operator() (Index row, Index col) const {
		return R::logspace_sub(lx.coeff(row, col), ly.coeff(row, col));
	}
};


template <class ArgType, class ArgType2>
CwiseNullaryOp<logspacesub_functor<ArgType, ArgType2>, typename logspace_struct<ArgType, ArgType2>::MatrixType>
logspacesub(const MatrixBase<ArgType> &lx, const MatrixBase<ArgType2> &ly)
{
	EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(ArgType, ArgType2);
	typedef typename logspace_struct<ArgType, ArgType2>::MatrixType MatrixType;
	return MatrixType::NullaryExpr(lx.rows(), lx.cols(), logspacesub_functor<ArgType, ArgType2>(lx.derived(), ly.derived()));
}

// end logspaceadd/sub

// begin densities

template<class ArgType>
struct densityarrayscale {
  typedef Matrix<typename ScalarBinaryOpTraits<typename ArgType::Scalar, double>::ReturnType,
                 ArgType::RowsAtCompileTime,
                 1,
                 ColMajor,
                 ArgType::MaxRowsAtCompileTime,
                 1> MatrixType;
};

template<class ArgType, class ArgType2>
struct densityarray {
  typedef Matrix<typename ScalarBinaryOpTraits<typename ScalarBinaryOpTraits<typename ArgType::Scalar, typename ArgType2::Scalar>::ReturnType, double>::ReturnType,
                 ArgType::SizeAtCompileTime,
                 ArgType2::SizeAtCompileTime,
                 ColMajor,
                 ArgType::MaxSizeAtCompileTime,
                 ArgType2::MaxSizeAtCompileTime> MatrixType;
};

// begin normal density
template<class ArgType, class ArgType2>
class dnorm_functor {
	const ArgType &x;
	const ArgType2 &mu;
	const double stdev;
	const bool lg;
public:
	dnorm_functor(const ArgType& x_, const ArgType2 &mu_, const double &stdev_, const bool &lg_) : x(x_), mu(mu_), stdev(stdev_), lg(lg_) {}

	const double operator() (Index row, Index col) const {
		return R::dnorm4(x.coeff(row), mu.coeff(col), stdev, lg);
	}
};

template <class ArgType, class ArgType2>
inline CwiseNullaryOp<dnorm_functor<ArgType, ArgType2>, typename densityarray<ArgType, ArgType2>::MatrixType>
dnormarray(const MatrixBase<ArgType>& x, const MatrixBase<ArgType2> & mu0, const double &stdev, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType);
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType2);
	typedef typename densityarray<ArgType, ArgType2>::MatrixType MatrixType;
	return MatrixType::NullaryExpr(x.size(), mu0.size(), dnorm_functor<ArgType, ArgType2>(x.derived(), mu0.derived(), stdev, lg));
}

template <class ArgType>
inline typename densityarrayscale<ArgType>::MatrixType
dnormarray(const MatrixBase<ArgType>& x, const double & mu0, const double &stdev, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType);
	return dnormarray(x, Matrix<double, 1, 1>::Constant(1, mu0), stdev, lg);
}

template <class ArgType>
inline typename densityarrayscale<ArgType>::MatrixType
dnpnorm_(const MatrixBase<ArgType> &x, const double &mu0, const double &pi0, const double& stdev, const bool& lg = false){
	typename densityarrayscale<ArgType>::MatrixType ans(x.size(), 1);
	if (lg){
		ans = dnormarray(x, mu0, stdev, true).array() + ((pi0 > 0) ? std::log(pi0) : -1e100);
	}else{
		ans = dnormarray(x, mu0, stdev) * pi0;
	}
	return ans;
}

template <class ArgType, class ArgType2, class ArgType3>
inline typename densityarrayscale<ArgType>::MatrixType
dnpnorm_(const MatrixBase<ArgType> &x, const MatrixBase<ArgType2> &mu0, const MatrixBase<ArgType3> &pi0, const double& stdev, const bool& lg = false){
	EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(ArgType2, ArgType3);
	typename densityarrayscale<ArgType>::MatrixType ans(x.size(), 1);
	if (mu0.size() == 1){
		ans = dnpnorm_(x, mu0[0], pi0[0], stdev, lg);
	}else{
		if (lg){
			ans = dnpnorm_(x, mu0[0], pi0[0], stdev, true); 
			for (auto i = 1; i < mu0.size(); ++i){
				ans = logspaceadd(ans, dnpnorm_(x, mu0[i], pi0[i], stdev, true));
			}
		}else{
			ans = dnormarray(x, mu0, stdev) * pi0;
		}		
	}
	
	return ans;
}

// end normal density

// begin normal cdf

template<class ArgType, class ArgType2>
class pnorm_functor {
	const ArgType &x;
	const ArgType2 &mu;
	const double stdev;
	const bool lt, lg;
public:
	pnorm_functor(const ArgType& x_, const ArgType2 &mu_, const double &stdev_, const bool &lt_, const bool &lg_) : x(x_), mu(mu_), stdev(stdev_), lt(lt_), lg(lg_) {}

	const double operator() (Index row, Index col) const {
		return R::pnorm5(x.coeff(row), mu.coeff(col), stdev, lt, lg);
	}
};


template <class ArgType, class ArgType2>
inline CwiseNullaryOp<pnorm_functor<ArgType, ArgType2>, typename densityarray<ArgType, ArgType2>::MatrixType>
pnormarray(const MatrixBase<ArgType>& x, const MatrixBase<ArgType2> &mu0, const double &stdev, const bool &lt = true, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType);
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType2);
	typedef typename densityarray<ArgType, ArgType2>::MatrixType MatrixType;
	return MatrixType::NullaryExpr(x.size(), mu0.size(), pnorm_functor<ArgType, ArgType2>(x.derived(), mu0.derived(), stdev, lt, lg));
}

template <class ArgType>
inline typename densityarrayscale<ArgType>::MatrixType
pnormarray(const MatrixBase<ArgType>& x, const double &mu0, const double &stdev, const bool &lt = true, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType);
 	return pnormarray(x, Matrix<double, 1, 1>::Constant(1, mu0), stdev, lt, lg);
}

template <class ArgType>
inline typename densityarrayscale<ArgType>::MatrixType
pnpnorm_(const MatrixBase<ArgType> &x, const double &mu0, const double &pi0, const double& stdev, const bool &lt = true, const bool& lg = false){
	typename densityarrayscale<ArgType>::MatrixType ans(x.size(), 1);
	
	if (lg){
		ans = pnormarray(x, mu0, stdev, lt, true).array() + ((pi0 > 0) ? std::log(pi0) : -1e100);
	}else{
		ans = pnormarray(x, mu0, stdev, lt, false) * pi0;
	}

	return ans;
}

template <class ArgType, class ArgType2, class ArgType3>
inline typename densityarrayscale<ArgType>::MatrixType
pnpnorm_(const MatrixBase<ArgType> &x, const MatrixBase<ArgType2> &mu0, const MatrixBase<ArgType3> &pi0, const double& stdev, const bool &lt = true, const bool& lg = false){
	EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(ArgType2, ArgType3);
	typename densityarrayscale<ArgType>::MatrixType ans(x.size(), 1);
	if (mu0.size() == 1){
		ans = pnpnorm_(x, mu0.coeff(0), pi0.coeff(0), stdev, lt, lg);
	}else{
		if (lg){
			ans = pnpnorm_(x, mu0[0], pi0[0], stdev, lt, true);
			for (auto i = 1; i < mu0.size(); ++i){
				ans = logspaceadd(ans, pnpnorm_(x, mu0[i], pi0[i], stdev, lt, true));
			}
		}else{
			ans = pnormarray(x, mu0, stdev, lt, false) * pi0;
		}	
	}

	return ans;
}

// end normal cdf

// begin t density

template<class ArgType, class ArgType2>
class dnt_functor {
	const ArgType &x;
	const ArgType2 &mu;
	const double n;
	const bool lg;
public:
	dnt_functor(const ArgType& x_, const ArgType2 &mu_, const double &n_, const bool &lg_) : x(x_), mu(mu_), n(n_), lg(lg_) {}

	const double operator() (Index row, Index col) const {
		return R::dnt(x.coeff(row), n, mu.coeff(col), lg);
	}
};

template <class ArgType, class ArgType2>
inline CwiseNullaryOp<dnt_functor<ArgType, ArgType2>, typename densityarray<ArgType, ArgType2>::MatrixType>
dtarray(const MatrixBase<ArgType>& x, const MatrixBase<ArgType2> &mu0, const double &n, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType);
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType2);
	typedef typename densityarray<ArgType, ArgType2>::MatrixType MatrixType;
	return MatrixType::NullaryExpr(x.size(), mu0.size(), dnt_functor<ArgType, ArgType2>(x.derived(), mu0.derived(), n, lg));
}

template <class ArgType>
inline typename densityarrayscale<ArgType>::MatrixType
dtarray(const MatrixBase<ArgType>& x, const double &mu0, const double &n, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType);
 	return dtarray(x, Matrix<double, 1, 1>::Constant(1, mu0), n, lg);
}

template <class ArgType>
inline typename densityarrayscale<ArgType>::MatrixType
dnpt_(const MatrixBase<ArgType> &x, const double &mu0, const double &pi0, const double& n, const bool& lg = false){
	typename densityarrayscale<ArgType>::MatrixType ans(x.size(), 1);

	if (lg){
		ans = dtarray(x, mu0, n, true).array() + ((pi0 > 0) ? std::log(pi0) : -1e100);
	}else{
		ans = dtarray(x, mu0, n, false) * pi0;
	}
	return ans;	
}

template <class ArgType, class ArgType2, class ArgType3>
inline typename densityarrayscale<ArgType>::MatrixType
dnpt_(const MatrixBase<ArgType> &x, const MatrixBase<ArgType2> &mu0, const MatrixBase<ArgType3> &pi0, const double& n, const bool& lg = false){
	EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(ArgType2, ArgType3);
	typename densityarrayscale<ArgType>::MatrixType ans(x.size(), 1);
	if (mu0.size() == 1){
		ans = dnpt_(x, mu0.coeff(0), pi0.coeff(0), n, lg);
	}else{
		if (lg){
			ans = dnpt_(x, mu0[0], pi0[0], n, true);
			for (auto i = 1; i < mu0.size(); ++i){
				ans = logspaceadd(ans, dnpt_(x, mu0[i], pi0[i], n, true));
			}
		}else{
			ans = dtarray(x, mu0, n, false) * pi0;
		}	
	}

	return ans;
}

// end t density

// begin one-parameter normal density
template<class ArgType, class ArgType2>
class dnormc_functor {
	const ArgType &x;
	const ArgType2 &mu;
	const double n;
	const bool lg;
public:
	dnormc_functor(const ArgType& x_, const ArgType2 &mu_, const double &n_, const bool &lg_) : x(x_), mu(mu_), n(n_), lg(lg_) {}

	const double operator() (Index row, Index col) const {
		return R::dnorm4(x.coeff(row), mu.coeff(col), (1. - mu.coeff(col) * mu.coeff(col)) / std::sqrt(n), lg);
	}
};

template <class ArgType, class ArgType2>
inline typename densityarray<ArgType, ArgType2>::MatrixType
dnormcarray(const MatrixBase<ArgType>& x, const MatrixBase<ArgType2> & mu0, const double &n, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType);
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType2);
	typedef typename densityarray<ArgType, ArgType2>::MatrixType MatrixType;
	return MatrixType::NullaryExpr(x.size(), mu0.size(), dnormc_functor<ArgType, ArgType2>(x.derived(), mu0.derived(), n, lg));
}

template <class ArgType>
inline typename densityarrayscale<ArgType>::MatrixType
dnormcarray(const MatrixBase<ArgType>& x, const double & mu0, const double &n, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType);
	return dnormcarray(x, Matrix<double, 1, 1>::Constant(1, mu0), n, lg);
}

template <class ArgType>
inline typename densityarrayscale<ArgType>::MatrixType
dnpnormc_(const MatrixBase<ArgType> &x, const double &mu0, const double &pi0, const double& n, const bool& lg = false){
	typename densityarrayscale<ArgType>::MatrixType ans(x.size(), 1);
	if (lg){
		ans = dnormcarray(x, mu0, n, true).array() + ((pi0 > 0) ? std::log(pi0) : -1e100);
	}else{
		ans = dnormcarray(x, mu0, n) * pi0;
	}

	return ans;
}

template <class ArgType, class ArgType2, class ArgType3>
inline typename densityarrayscale<ArgType>::MatrixType
dnpnormc_(const MatrixBase<ArgType> &x, const MatrixBase<ArgType2> &mu0, const MatrixBase<ArgType3> &pi0, const double& n, const bool& lg = false){
	EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(ArgType2, ArgType3);
	typename densityarrayscale<ArgType>::MatrixType ans(x.size(), 1);
	if (mu0.size() == 1){
		ans = dnpnormc_(x, mu0.coeff(0), pi0.coeff(0), n, lg);
	}else{
		if (lg){
			ans = dnpnormc_(x, mu0[0], pi0[0], n, true);
			for (auto i = 1; i < mu0.size(); ++i){
				ans = logspaceadd(ans, dnpnormc_(x, mu0[i], pi0[i], n, true));
			}
		}else{
			ans = dnormcarray(x, mu0, n, false) * pi0;
		}	
	}

	return ans;
}

// end one-parameter normal density

// begin discrete normal cdf

template <class ArgType>
inline typename densityarrayscale<ArgType>::MatrixType
pdiscnormarray(const MatrixBase<ArgType> &x, const double &mu0, const double &stdev, const double &h, const bool &lt = true, const bool &lg = false){
	return pnormarray(x, mu0 - h, stdev, lt, lg);
}

template <class ArgType, class ArgType2>
inline typename densityarray<ArgType, ArgType2>::MatrixType
pdiscnormarray(const MatrixBase<ArgType> &x, const MatrixBase<ArgType2> &mu0, const double &stdev, const double &h, const bool &lt = true, const bool &lg = false){
	if (mu0.size() == 1){
		return pdiscnormarray(x, mu0.coeff(0), stdev, h, lt, lg);
	}else{
		if (x.size() > mu0.size()){
			return pnormarray(x, mu0 - Eigen::VectorXd::Constant(mu0.size(), h), stdev, lt, lg);
		}else{
			return pnormarray(x + Eigen::VectorXd::Constant(x.size(), h), mu0, stdev, lt, lg);
		}
	}
}

template <class ArgType>
inline typename densityarrayscale<ArgType>::MatrixType
pnpdiscnorm_(const MatrixBase<ArgType> &x, const double &mu0, const double &pi0, const double &stdev, const double& h, const double &lt = true, const bool& lg = false){
	typename densityarrayscale<ArgType>::MatrixType ans(x.size(), 1);
	
	if (lg){
		ans = pdiscnormarray(x, mu0, stdev, h, lt, true).array() + ((pi0 > 0) ? std::log(pi0) : -1e100);
	}else{
		ans = pdiscnormarray(x, mu0, stdev, h, lt, false) * pi0;
	}

	return ans;
}

template <class ArgType, class ArgType2, class ArgType3>
inline typename densityarrayscale<ArgType>::MatrixType
pnpdiscnorm_(const MatrixBase<ArgType> &x, const MatrixBase<ArgType2> &mu0, const MatrixBase<ArgType3> &pi0, const double& stdev, const double &h, const bool &lt = true, const bool& lg = false){
	EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(ArgType2, ArgType3);
	typename densityarrayscale<ArgType>::MatrixType ans(x.size(), 1);
	if (mu0.size() == 1){
		ans = pnpdiscnorm_(x, mu0.coeff(0), pi0.coeff(0), stdev, h, lt, lg);
	}else{
		if (lg){
			ans = pnpdiscnorm_(x, mu0[0], pi0[0], stdev, h, lt, true);
			for (auto i = 1; i < mu0.size(); ++i){
				ans = logspaceadd(ans, pnpdiscnorm_(x, mu0[i], pi0[i], stdev, h, lt, true));
			}
		}else{
			ans = pdiscnormarray(x, mu0, stdev, h, lt, false) * pi0;
		}	
	}

	return ans;
}

// end discrete normal cdf

// begin discrete normal density

template <class ArgType>
inline typename densityarrayscale<ArgType>::MatrixType
ddiscnormarray(const MatrixBase<ArgType> &x, const double &mu0, const double &stdev, const double &h, const bool &lg = false){
	// Trapezoid rule. Relative error <= 1e-6 / 12;
	typedef typename densityarrayscale<ArgType>::MatrixType MatrixType;
	MatrixType ans(x.size(), 1);
	if (lg){
		ans = logspacesub(pnormarray(x, mu0 - h, stdev, true, true), pnormarray(x, mu0, stdev, true, true));
	}else{
		int N = std::max(std::ceil((x.array() - mu0).abs().maxCoeff() * 1e3 * std::pow(h, 1.5)), 5.);
		double delta = h / N;
		ans = (dnormarray(x, mu0, stdev) + dnormarray(x, mu0 - h, stdev)) * 0.5;
		for (int i = 1; i < N; i++){
			ans.noalias() += dnormarray(x, mu0 - delta * i, stdev);
		}
		ans = ans * delta;
	}

	return ans;
}

template <class ArgType, class ArgType2>
inline typename densityarray<ArgType, ArgType2>::MatrixType
ddiscnormarray(const MatrixBase<ArgType> &x, const MatrixBase<ArgType2> &mu0, const double &stdev, const double &h, const bool &lg = false){
	typedef typename densityarray<ArgType, ArgType2>::MatrixType MatrixType;
	typedef typename densityarrayscale<ArgType2>::MatrixType MatrixType2;
	MatrixType ans(x.size(), mu0.size());
	if (mu0.size() == 1){
		ans = ddiscnormarray(x, mu0[0], stdev, h, false);
	}else{
		// Trapezoid rule. Relative error <= 1e-6 / 12;
		if (lg){
			ans = logspacesub(pnormarray(x, mu0 - MatrixType2::Constant(mu0.size(), h), stdev, true, true), pnormarray(x, mu0, stdev, true, true));
		}else{
			int N = std::max(std::ceil(std::max(x.maxCoeff() - mu0.minCoeff(), mu0.maxCoeff() - x.minCoeff()) * 1e3 * std::pow(h, 1.5)), 5.);
			double delta = h / N;
			ans = (dnormarray(x, mu0, stdev) + dnormarray(x, mu0 - Eigen::VectorXd::Constant(mu0.size(), h), stdev)) * 0.5;
			for (int i = 1; i < N; i++){
				ans.noalias() += dnormarray(x, mu0 - Eigen::VectorXd::Constant(mu0.size(), delta * i), stdev);
			}
			ans = ans * delta;
		}
	}
	return ans;
}

template <class ArgType>
inline typename densityarrayscale<ArgType>::MatrixType
dnpdiscnorm_(const MatrixBase<ArgType> &x, const double &mu0, const double &pi0, const double &stdev, const double& h, const bool& lg = false){
	typename densityarrayscale<ArgType>::MatrixType ans(x.size(), 1);
	if (lg){
		ans = ddiscnormarray(x, mu0, stdev, h, true).array() + ((pi0 > 0) ? std::log(pi0) : -1e100);
	}else{
		ans = ddiscnormarray(x, mu0, stdev, h, false) * pi0;
	}
	return ans;
}

template <class ArgType, class ArgType2, class ArgType3>
inline typename densityarrayscale<ArgType>::MatrixType
dnpdiscnorm_(const MatrixBase<ArgType> &x, const MatrixBase<ArgType2> &mu0, const MatrixBase<ArgType3> &pi0, const double& stdev, const double &h, const bool& lg = false){
	EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(ArgType2, ArgType3);
	typename densityarrayscale<ArgType>::MatrixType ans(x.size(), 1);
	if (mu0.size() == 1){
		ans = dnpdiscnorm_(x, mu0.coeff(0), pi0.coeff(0), stdev, h, lg);
	}else{
		if (lg){
			ans = dnpdiscnorm_(x, mu0[0], pi0[0], stdev, h, true);
			for (auto i = 1; i < mu0.size(); ++i){
				ans = logspaceadd(ans, dnpdiscnorm_(x, mu0[i], pi0[i], stdev, h, true));
			}
		}else{
			ans = ddiscnormarray(x, mu0, stdev, h, false) * pi0;
		}	
	}
	return ans;
}

// end discrete normal density

// begin indexing
	
template<class ArgType, class RowIndexType, class ColIndexType>
class indexing_functor {
	const ArgType &m_arg;
	const RowIndexType &m_rowIndices;
	const ColIndexType &m_colIndices;
public:
	typedef Matrix<typename ArgType::Scalar,
	             RowIndexType::SizeAtCompileTime,
	             ColIndexType::SizeAtCompileTime,
	             ArgType::Flags&RowMajorBit?RowMajor:ColMajor,
	             RowIndexType::MaxSizeAtCompileTime,
	             ColIndexType::MaxSizeAtCompileTime> MatrixType;

	indexing_functor(const ArgType& arg, const RowIndexType& row_indices, const ColIndexType& col_indices)
	: m_arg(arg), m_rowIndices(row_indices), m_colIndices(col_indices)
	{}

	const typename ArgType::Scalar& operator() (Index row, Index col) const {
		return m_arg(m_rowIndices[row], m_colIndices[col]);
	}

	const typename ArgType::Scalar& operator() (Index row) const {
		return m_arg.coef(m_rowIndices[row]);
	}
};


template <class ArgType, class RowIndexType, class ColIndexType>
CwiseNullaryOp<indexing_functor<ArgType,RowIndexType,ColIndexType>, typename indexing_functor<ArgType,RowIndexType,ColIndexType>::MatrixType>
indexing(const MatrixBase<ArgType>& arg, const RowIndexType& row_indices, const ColIndexType& col_indices)
{
	typedef indexing_functor<ArgType,RowIndexType,ColIndexType> Func;
	typedef typename Func::MatrixType MatrixType;
	return MatrixType::NullaryExpr(row_indices.size(), col_indices.size(), Func(arg.derived(), row_indices, col_indices));
}

template <class ArgType, class RowIndexType>
CwiseNullaryOp<indexing_functor<ArgType,RowIndexType, Matrix<int, 1, 1> >, typename indexing_functor<ArgType,RowIndexType, Matrix<int, 1, 1> >::MatrixType>
indexing(const MatrixBase<ArgType>& arg, const RowIndexType& row_indices)
{
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType);	
	typedef indexing_functor<ArgType,RowIndexType, Matrix<int, 1, 1> > Func;
	typedef typename Func::MatrixType MatrixType;
	return MatrixType::NullaryExpr(row_indices.size(), Func(arg.derived(), row_indices, Eigen::VectorXi::Zero(1)));
}

// end indexing

// begin pnnls
template<class ArgType, class ArgType2>
struct pnnls_struct {
  typedef Matrix<typename ScalarBinaryOpTraits<typename ScalarBinaryOpTraits<typename ArgType::Scalar, typename ArgType2::Scalar>::ReturnType, double>::ReturnType,
                 ArgType::ColsAtCompileTime,
                 1,
                 ColMajor,
                 ArgType::MaxColsAtCompileTime,
                 1> MatrixType;
};

template<class ArgType, class ArgType2>
inline typename pnnls_struct<ArgType, ArgType2>::MatrixType
pnnlssum_(const MatrixBase<ArgType> &A, const MatrixBase<ArgType2> &b, const double &sum){
	int m = A.rows() + 1, n = A.cols();
	MatrixXd AA = ((A * sum).colwise() - b).colwise().homogeneous();
	VectorXd bb = Eigen::VectorXd::Zero(m - 1).homogeneous(), zz(m);
	VectorXd x(n), w(n);
	double rnorm;
	VectorXi index(n);
	int mode, k = 0;
	pnnls_(AA.data(), &m, &m, &n, bb.data(), x.data(), &rnorm, w.data(), zz.data(), index.data(), &mode, &k);
	x = x / x.sum() * sum;
	return x;
}

// The program pnnqp using LLT
template<class ArgType, class ArgType2>
inline typename pnnls_struct<ArgType, ArgType2>::MatrixType
pnnqp_(const MatrixBase<ArgType> &q, const MatrixBase<ArgType2> &p, const double &sum){
	SelfAdjointEigenSolver<MatrixXd> eig(q.rows());
	eig.compute(q);
	Matrix<typename ArgType::Scalar, ArgType::RowsAtCompileTime, 1> eigvec = eig.eigenvalues();
	Matrix<typename ArgType::Scalar, ArgType::RowsAtCompileTime, ArgType::RowsAtCompileTime> eigmat = eig.eigenvectors();
	int index = (eigvec.array() > eigvec.template tail<1>()[0] * 1e-15).count();
	return pnnlssum_((eigmat.rightCols(index) * eigvec.tail(index).cwiseSqrt().asDiagonal()).transpose(), 
		(eigmat.rightCols(index).transpose() * -p).cwiseQuotient(eigvec.tail(index).cwiseSqrt()), sum);
}


}


class comparemu0
{
private:
	Eigen::VectorXd mu0;
public:
	comparemu0(const Eigen::VectorXd &mu0_) : mu0(mu0_) {};

	const bool operator()(const int & x, const int & y) const {return mu0[x] < mu0[y];};
};


// The program NNLS
inline Eigen::VectorXi index2num(const Eigen::VectorXi &index){
	// This function should only be used before Eigen 3.4.
	// In Eigen 3.4 there is a built-in function for slicing.
	Eigen::VectorXi ans(index.sum());
	int *ansptr = ans.data();
	for (int i = 0; i < index.size(); ++i){
		if (index[i]){
			*ansptr = i;
			ansptr++;
		}
	}
	return ans;
}

#endif