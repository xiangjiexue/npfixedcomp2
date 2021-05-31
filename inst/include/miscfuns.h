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
  typedef Matrix<typename ArgType::Scalar,
                 ArgType::RowsAtCompileTime,
                 ArgType::ColsAtCompileTime,
                 ArgType::Flags&RowMajorBit?RowMajor:ColMajor,
                 ArgType::MaxRowsAtCompileTime,
                 ArgType::MaxColsAtCompileTime> MatrixType;
};

template<class ArgType>
class log1mexp_functor {
	const ArgType &x;
public:
	log1mexp_functor(const ArgType& x_) : x(x_) {}

	const typename ArgType::Scalar operator() (Index row, Index col) const {
		#ifdef log1mexp
			return Rf_log1mexp((double) x.coeff(row, col)); // R 4.1 provides the implementation.
		#else
			return (x.coeff(row, col) < -M_LN2) ? std::log1p(std::exp(x.coeff(row, col)) * -1.) : std::log(-1. * std::expm1(x.coeff(row, col)));
		#endif
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
  typedef Matrix<typename ScalarBinaryOpTraits<typename ArgType::Scalar, typename ArgType2::Scalar>::ReturnType,
                 ArgType::SizeAtCompileTime,
                 ArgType::SizeAtCompileTime,
                 ArgType::Flags&RowMajorBit?RowMajor:ColMajor,
                 ArgType::MaxSizeAtCompileTime,
                 ArgType::MaxSizeAtCompileTime> MatrixType;
};

template<class ArgType, class ArgType2>
class logspaceadd_functor {
	const ArgType &lx;
	const ArgType2 &ly;
public:
	logspaceadd_functor(const ArgType& lx_, const ArgType2 &ly_) : lx(lx_), ly(ly_) {}

	const typename ScalarBinaryOpTraits<typename ArgType::Scalar, typename ArgType2::Scalar>::ReturnType operator() (Index row, Index col) const {
		return Rf_logspace_add((double) lx.coeff(row, col), (double) ly.coeff(row, col));
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

	const typename ScalarBinaryOpTraits<typename ArgType::Scalar, typename ArgType2::Scalar>::ReturnType operator() (Index row, Index col) const {
		return Rf_logspace_sub((double) lx.coeff(row, col), (double) ly.coeff(row, col));
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
struct densityscale {
  typedef Matrix<typename ArgType::Scalar,
                 ArgType::RowsAtCompileTime,
                 1,
                 ColMajor,
                 ArgType::MaxRowsAtCompileTime,
                 1> MatrixType;	
};

template<class ArgType, class ArgType2>
struct densityvec {
  typedef Matrix<typename ScalarBinaryOpTraits<typename ArgType::Scalar, typename ArgType2::Scalar>::ReturnType,
                 ArgType::SizeAtCompileTime,
                 ArgType2::SizeAtCompileTime,
                 ColMajor,
                 ArgType::MaxSizeAtCompileTime,
                 ArgType2::MaxSizeAtCompileTime> MatrixType;
};

template<class ArgType, class ArgType2, class ArgType3>
struct densitynp {
  typedef Matrix<typename ScalarBinaryOpTraits<typename ScalarBinaryOpTraits<typename ArgType::Scalar, typename ArgType2::Scalar>::ReturnType, typename ArgType3::Scalar>::ReturnType,
                 ArgType::SizeAtCompileTime,
                 1,
                 ColMajor,
                 ArgType::MaxSizeAtCompileTime,
                 1> MatrixType;
};

// begin normal density

// Eigen implementation. Might have weird behaviour under some circumstance.
template <class ArgType>
inline typename densityscale<ArgType>::MatrixType
dnormarray(const MatrixBase<ArgType>& x, const typename MatrixBase<ArgType>::Scalar & mu0, const typename MatrixBase<ArgType>::Scalar &stdev, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType);
	typedef typename densityscale<ArgType>::MatrixType MatrixType;
	MatrixType ans(x.size(), 1);
	ans = (x.array() - mu0).square() / (-2 * stdev * stdev) - (M_LN_SQRT_2PI + std::log(stdev));
	if (lg){
		return ans;
	}else{
		return ans.array().exp();
	}
}

template <class ArgType, class ArgType2>
inline typename densityvec<ArgType, ArgType2>::MatrixType
dnormarray(const MatrixBase<ArgType>& x, const MatrixBase<ArgType2> & mu0, const typename MatrixBase<ArgType>::Scalar &stdev, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType);
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType2);
	typedef typename densityvec<ArgType, ArgType2>::MatrixType MatrixType;
	MatrixType ans(x.size(), mu0.size());
	if (mu0.size() == 1){
		ans = dnormarray(x, mu0.coeff(0), stdev, true);
	}else{
		ans = (mu0.transpose().colwise().replicate(x.size()).colwise() - x).array().square() / (-2 * stdev * stdev) - (M_LN_SQRT_2PI + std::log(stdev));
	}
	if (lg){
		return ans;
	}else{
		return ans.array().exp();
	}
}

template <class ArgType>
inline typename densityscale<ArgType>::MatrixType
dnpnorm_(const MatrixBase<ArgType> &x, const typename MatrixBase<ArgType>::Scalar &mu0, 
	const typename MatrixBase<ArgType>::Scalar &pi0, const typename MatrixBase<ArgType>::Scalar & stdev, const bool& lg = false){
	typename densityscale<ArgType>::MatrixType ans(x.size(), 1);
	if (lg){
		ans = dnormarray(x, mu0, stdev, true).array() + ((pi0 > 0) ? std::log(pi0) : -1e100);
	}else{
		ans = dnormarray(x, mu0, stdev) * pi0;
	}
	return ans;
}

template <class ArgType, class ArgType2, class ArgType3>
inline typename densitynp<ArgType, ArgType2, ArgType3>::MatrixType
dnpnorm_(const MatrixBase<ArgType> &x, const MatrixBase<ArgType2> &mu0, const MatrixBase<ArgType3> &pi0, const typename MatrixBase<ArgType>::Scalar& stdev, const bool& lg = false){
	EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(ArgType2, ArgType3);
	typename densitynp<ArgType, ArgType2, ArgType3>::MatrixType ans(x.size(), 1);
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
	const typename MatrixBase<ArgType>::Scalar stdev;
	const bool lt, lg;
public:
	pnorm_functor(const ArgType& x_, const ArgType2 &mu_, const typename MatrixBase<ArgType>::Scalar &stdev_, const bool &lt_, const bool &lg_) : x(x_), mu(mu_), stdev(stdev_), lt(lt_), lg(lg_) {}

	const typename ScalarBinaryOpTraits<typename ArgType::Scalar, typename ArgType2::Scalar>::ReturnType operator() (Index row, Index col) const {
		return Rf_pnorm5((double) x.coeff(row), (double) mu.coeff(col), (double) stdev, lt, lg);
	}
};


template <class ArgType, class ArgType2>
inline CwiseNullaryOp<pnorm_functor<ArgType, ArgType2>, typename densityvec<ArgType, ArgType2>::MatrixType>
pnormarray(const MatrixBase<ArgType>& x, const MatrixBase<ArgType2> &mu0, const typename MatrixBase<ArgType>::Scalar &stdev, const bool &lt = true, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType);
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType2);
	typedef typename densityvec<ArgType, ArgType2>::MatrixType MatrixType;
	return MatrixType::NullaryExpr(x.size(), mu0.size(), pnorm_functor<ArgType, ArgType2>(x.derived(), mu0.derived(), stdev, lt, lg));
}

template <class ArgType>
inline typename densityscale<ArgType>::MatrixType
pnormarray(const MatrixBase<ArgType>& x, const typename MatrixBase<ArgType>::Scalar &mu0, const typename MatrixBase<ArgType>::Scalar &stdev, const bool &lt = true, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType);
 	return pnormarray(x, Matrix<typename MatrixBase<ArgType>::Scalar, 1, 1>::Constant(1, mu0), stdev, lt, lg);
}

template <class ArgType>
inline typename densityscale<ArgType>::MatrixType
pnpnorm_(const MatrixBase<ArgType> &x, const typename MatrixBase<ArgType>::Scalar &mu0, 
	const typename MatrixBase<ArgType>::Scalar &pi0, const typename MatrixBase<ArgType>::Scalar& stdev, const bool &lt = true, const bool& lg = false){
	typename densityscale<ArgType>::MatrixType ans(x.size(), 1);
	
	if (lg){
		ans = pnormarray(x, mu0, stdev, lt, true).array() + ((pi0 > 0) ? std::log(pi0) : -1e100);
	}else{
		ans = pnormarray(x, mu0, stdev, lt, false) * pi0;
	}

	return ans;
}

template <class ArgType, class ArgType2, class ArgType3>
inline typename densitynp<ArgType, ArgType2, ArgType3>::MatrixType
pnpnorm_(const MatrixBase<ArgType> &x, const MatrixBase<ArgType2> &mu0, const MatrixBase<ArgType3> &pi0, const typename MatrixBase<ArgType>::Scalar& stdev, const bool &lt = true, const bool& lg = false){
	EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(ArgType2, ArgType3);
	typename densitynp<ArgType, ArgType2, ArgType3>::MatrixType ans(x.size(), 1);
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
	const typename MatrixBase<ArgType>::Scalar n;
	const bool lg;
public:
	dnt_functor(const ArgType& x_, const ArgType2 &mu_, const typename MatrixBase<ArgType>::Scalar &n_, const bool &lg_) : x(x_), mu(mu_), n(n_), lg(lg_) {}

	const typename ScalarBinaryOpTraits<typename ArgType::Scalar, typename ArgType2::Scalar>::ReturnType operator() (Index row, Index col) const {
		return Rf_dnt((double) x.coeff(row), (double) n, (double) mu.coeff(col), lg);
	}
};

template <class ArgType, class ArgType2>
inline CwiseNullaryOp<dnt_functor<ArgType, ArgType2>, typename densityvec<ArgType, ArgType2>::MatrixType>
dtarray(const MatrixBase<ArgType>& x, const MatrixBase<ArgType2> &mu0, const typename MatrixBase<ArgType>::Scalar &n, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType);
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType2);
	typedef typename densityvec<ArgType, ArgType2>::MatrixType MatrixType;
	return MatrixType::NullaryExpr(x.size(), mu0.size(), dnt_functor<ArgType, ArgType2>(x.derived(), mu0.derived(), n, lg));
}

template <class ArgType>
inline typename densityscale<ArgType>::MatrixType
dtarray(const MatrixBase<ArgType>& x, const typename MatrixBase<ArgType>::Scalar &mu0, const typename MatrixBase<ArgType>::Scalar &n, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType);
 	return dtarray(x, Matrix<typename MatrixBase<ArgType>::Scalar, 1, 1>::Constant(1, mu0), n, lg);
}

template <class ArgType>
inline typename densityscale<ArgType>::MatrixType
dnpt_(const MatrixBase<ArgType> &x, const typename MatrixBase<ArgType>::Scalar &mu0, 
	const typename MatrixBase<ArgType>::Scalar &pi0, const typename MatrixBase<ArgType>::Scalar& n, const bool& lg = false){
	typename densityscale<ArgType>::MatrixType ans(x.size(), 1);

	if (lg){
		ans = dtarray(x, mu0, n, true).array() + ((pi0 > 0) ? std::log(pi0) : -1e100);
	}else{
		ans = dtarray(x, mu0, n, false) * pi0;
	}
	return ans;	
}

template <class ArgType, class ArgType2, class ArgType3>
inline typename densitynp<ArgType, ArgType2, ArgType3>::MatrixType
dnpt_(const MatrixBase<ArgType> &x, const MatrixBase<ArgType2> &mu0, const MatrixBase<ArgType3> &pi0, const typename MatrixBase<ArgType>::Scalar& n, const bool& lg = false){
	EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(ArgType2, ArgType3);
	typename densitynp<ArgType, ArgType2, ArgType3>::MatrixType ans(x.size(), 1);
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

template <class ArgType>
inline typename densityscale<ArgType>::MatrixType
dnormcarray(const MatrixBase<ArgType>& x, const typename MatrixBase<ArgType>::Scalar & mu0, const typename MatrixBase<ArgType>::Scalar &n, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType);
	typedef typename densityscale<ArgType>::MatrixType MatrixType;
	MatrixType ans(x.size(), 1);
	ans = ((x.array() - mu0) / (1. - mu0 * mu0)).square() * (n / -2.) - (M_LN_SQRT_2PI + std::log1p(- mu0 * mu0) - std::log(std::sqrt(n)));
	if (lg){
		return ans;
	}else{
		return ans.array().exp();
	}
}

template <class ArgType, class ArgType2>
inline typename densityvec<ArgType, ArgType2>::MatrixType
dnormcarray(const MatrixBase<ArgType>& x, const MatrixBase<ArgType2> & mu0, const typename MatrixBase<ArgType>::Scalar &n, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType);
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType2);
	typedef typename densityvec<ArgType, ArgType2>::MatrixType MatrixType;
	MatrixType ans(x.size(), mu0.size());
	if (mu0.size() == 1){
		ans = dnormcarray(x, mu0.coeff(0), n, true);
	}else{
		MatrixType stdevarray = ((1. - mu0.array().square()) / std::sqrt(n)).replicate(1, x.size()).transpose();
		ans = (mu0.transpose().colwise().replicate(x.size()).colwise() - x).array().square() / (-2 * stdevarray.array().square()) - (M_LN_SQRT_2PI + stdevarray.array().log());
	}
	if (lg){
		return ans;
	}else{
		return ans.array().exp();
	}
}

template <class ArgType>
inline typename densityscale<ArgType>::MatrixType
dnpnormc_(const MatrixBase<ArgType> &x, const typename MatrixBase<ArgType>::Scalar &mu0, 
	const typename MatrixBase<ArgType>::Scalar &pi0, const typename MatrixBase<ArgType>::Scalar& n, const bool& lg = false){
	typename densityscale<ArgType>::MatrixType ans(x.size(), 1);
	if (lg){
		ans = dnormcarray(x, mu0, n, true).array() + ((pi0 > 0) ? std::log(pi0) : -1e100);
	}else{
		ans = dnormcarray(x, mu0, n) * pi0;
	}

	return ans;
}

template <class ArgType, class ArgType2, class ArgType3>
inline typename densitynp<ArgType, ArgType2, ArgType3>::MatrixType
dnpnormc_(const MatrixBase<ArgType> &x, const MatrixBase<ArgType2> &mu0, const MatrixBase<ArgType3> &pi0, 
	const typename MatrixBase<ArgType>::Scalar& n, const bool& lg = false){
	EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(ArgType2, ArgType3);
	typename densitynp<ArgType, ArgType2, ArgType3>::MatrixType ans(x.size(), 1);
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
inline typename densityscale<ArgType>::MatrixType
pdiscnormarray(const MatrixBase<ArgType> &x, const typename MatrixBase<ArgType>::Scalar &mu0, 
	const typename MatrixBase<ArgType>::Scalar &stdev, const typename MatrixBase<ArgType>::Scalar &h, const bool &lt = true, const bool &lg = false){
	return pnormarray(x, mu0 - h, stdev, lt, lg);
}

template <class ArgType, class ArgType2>
inline typename densityvec<ArgType, ArgType2>::MatrixType
pdiscnormarray(const MatrixBase<ArgType> &x, const MatrixBase<ArgType2> &mu0, 
	const typename MatrixBase<ArgType>::Scalar &stdev, const typename MatrixBase<ArgType>::Scalar &h, const bool &lt = true, const bool &lg = false){
	if (mu0.size() == 1){
		return pdiscnormarray(x, mu0.coeff(0), stdev, h, lt, lg);
	}else{
		if (x.size() > mu0.size()){
			return pnormarray(x, mu0 - MatrixBase<ArgType2>::Constant(mu0.size(), h), stdev, lt, lg);
		}else{
			return pnormarray(x + MatrixBase<ArgType>::Constant(x.size(), h), mu0, stdev, lt, lg);
		}
	}
}

template <class ArgType>
inline typename densityscale<ArgType>::MatrixType
pnpdiscnorm_(const MatrixBase<ArgType> &x, const typename MatrixBase<ArgType>::Scalar &mu0, 
	const typename MatrixBase<ArgType>::Scalar &pi0, const typename MatrixBase<ArgType>::Scalar &stdev, 
	const typename MatrixBase<ArgType>::Scalar & h, const bool &lt = true, const bool& lg = false){
	typename densityscale<ArgType>::MatrixType ans(x.size(), 1);
	
	if (lg){
		ans = pdiscnormarray(x, mu0, stdev, h, lt, true).array() + ((pi0 > 0) ? std::log(pi0) : -1e100);
	}else{
		ans = pdiscnormarray(x, mu0, stdev, h, lt, false) * pi0;
	}

	return ans;
}

template <class ArgType, class ArgType2, class ArgType3>
inline typename densitynp<ArgType, ArgType2, ArgType3>::MatrixType
pnpdiscnorm_(const MatrixBase<ArgType> &x, const MatrixBase<ArgType2> &mu0, const MatrixBase<ArgType3> &pi0, 
	const typename MatrixBase<ArgType>::Scalar& stdev, const typename MatrixBase<ArgType>::Scalar &h, const bool &lt = true, const bool& lg = false){
	EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(ArgType2, ArgType3);
	typename densitynp<ArgType, ArgType2, ArgType3>::MatrixType ans(x.size(), 1);
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
inline typename densityscale<ArgType>::MatrixType
ddiscnormarray(const MatrixBase<ArgType> &x, const typename MatrixBase<ArgType>::Scalar &mu0,
 const typename MatrixBase<ArgType>::Scalar &stdev, const typename MatrixBase<ArgType>::Scalar &h, const bool &lg = false){
	// Trapezoid rule. Relative error <= 1e-6 / 12;
	typedef typename densityscale<ArgType>::MatrixType MatrixType;
	MatrixType ans(x.size(), 1);
	if (lg){
		ans = logspacesub(pnormarray(x, mu0 - h, stdev, true, true), pnormarray(x, mu0, stdev, true, true));
	}else{
		int N = std::max(std::ceil((x.array() - mu0).abs().maxCoeff() * 1e3 * std::pow(h, 1.5)), 5.);
		const typename MatrixBase<ArgType>::Scalar delta = h / N;
		ans = (dnormarray(x, mu0, stdev) + dnormarray(x, mu0 - h, stdev)) * 0.5;
		for (int i = 1; i < N; i++){
			ans.noalias() += dnormarray(x, mu0 - delta * i, stdev);
		}
		ans = ans * delta;
	}

	return ans;
}

template <class ArgType, class ArgType2>
inline typename densityvec<ArgType, ArgType2>::MatrixType
ddiscnormarray(const MatrixBase<ArgType> &x, const MatrixBase<ArgType2> &mu0, 
	const typename MatrixBase<ArgType>::Scalar &stdev, const typename MatrixBase<ArgType>::Scalar &h, const bool &lg = false){
	typedef typename densityvec<ArgType, ArgType2>::MatrixType MatrixType;
	MatrixType ans(x.size(), mu0.size());
	if (mu0.size() == 1){
		ans = ddiscnormarray(x, mu0[0], stdev, h, false);
	}else{
		// Trapezoid rule. Relative error <= 1e-6 / 12;
		if (lg){
			ans = logspacesub(pnormarray(x, mu0 - MatrixBase<ArgType2>::Constant(mu0.size(), h), stdev, true, true), pnormarray(x, mu0, stdev, true, true));
		}else{
			int N = std::max(std::ceil(std::max(x.maxCoeff() - mu0.minCoeff(), mu0.maxCoeff() - x.minCoeff()) * 1e3 * std::pow(h, 1.5)), 5.);
			const typename MatrixBase<ArgType>::Scalar delta = h / N;
			ans = (dnormarray(x, mu0, stdev) + dnormarray(x, mu0 - MatrixBase<ArgType2>::Constant(mu0.size(), h), stdev)) * 0.5;
			for (int i = 1; i < N; i++){
				ans.noalias() += dnormarray(x, mu0 - MatrixBase<ArgType2>::Constant(mu0.size(), delta * i), stdev);
			}
			ans = ans * delta;
		}
	}
	return ans;
}

template <class ArgType>
inline typename densityscale<ArgType>::MatrixType
dnpdiscnorm_(const MatrixBase<ArgType> &x, const typename MatrixBase<ArgType>::Scalar &mu0, 
	const typename MatrixBase<ArgType>::Scalar &pi0, const typename MatrixBase<ArgType>::Scalar &stdev, 
	const typename MatrixBase<ArgType>::Scalar & h, const bool& lg = false){
	typename densityscale<ArgType>::MatrixType ans(x.size(), 1);
	if (lg){
		ans = ddiscnormarray(x, mu0, stdev, h, true).array() + ((pi0 > 0) ? std::log(pi0) : -1e100);
	}else{
		ans = ddiscnormarray(x, mu0, stdev, h, false) * pi0;
	}
	return ans;
}

template <class ArgType, class ArgType2, class ArgType3>
inline typename densitynp<ArgType, ArgType2, ArgType3>::MatrixType
dnpdiscnorm_(const MatrixBase<ArgType> &x, const MatrixBase<ArgType2> &mu0, const MatrixBase<ArgType3> &pi0, 
	const typename MatrixBase<ArgType>::Scalar& stdev, const typename MatrixBase<ArgType>::Scalar &h, const bool& lg = false){
	EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(ArgType2, ArgType3);
	typename densitynp<ArgType, ArgType2, ArgType3>::MatrixType ans(x.size(), 1);
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

// begin poisson density
// Eigen implementation. Might have weird behaviour under some circumstance.
// return type conversion is not working for <int, double>. the vector x needs to be of double type.
// structure parameter stdev not referenced.
template <class ArgType>
inline typename densityscale<ArgType>::MatrixType
dpoisarray(const MatrixBase<ArgType>& x, const typename MatrixBase<ArgType>::Scalar & mu0, const typename MatrixBase<ArgType>::Scalar &stdev, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType);
	typedef typename densityscale<ArgType>::MatrixType MatrixType;
	MatrixType ans(x.size(), 1);
	ans = (x.array() + mu0 > 0).select(x.array() * std::log(mu0) - mu0 - (x.array() + 1.).lgamma(), MatrixType::Zero(x.size(), 1));
	if (lg){
		return ans;
	}else{
		return ans.array().exp();
	}
}

template <class ArgType, class ArgType2>
inline typename densityvec<ArgType, ArgType2>::MatrixType
dpoisarray(const MatrixBase<ArgType>& x, const MatrixBase<ArgType2> & mu0, const typename MatrixBase<ArgType>::Scalar &stdev, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType);
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType2);
	typedef typename densityvec<ArgType, ArgType2>::MatrixType MatrixType;
	MatrixType ans(x.size(), mu0.size());
	if (mu0.size() == 1){
		ans = dpoisarray(x, mu0.coeff(0), stdev, true);
	}else{
		MatrixType mu0array = mu0.replicate(1, x.size()).transpose();
		ans = (mu0array.array().colwise() + x.array() > 0).select(mu0array.array().log().colwise() * x.array() - mu0array.array() - (x.array() + 1.).lgamma().replicate(1, mu0.size()), MatrixType::Zero(x.size(), mu0.size()));
	}
	if (lg){
		return ans;
	}else{
		return ans.array().exp();
	}
}

template <class ArgType>
inline typename densityscale<ArgType>::MatrixType
dnppois_(const MatrixBase<ArgType> &x, const typename MatrixBase<ArgType>::Scalar &mu0, 
	const typename MatrixBase<ArgType>::Scalar &pi0, const typename MatrixBase<ArgType>::Scalar& stdev, const bool& lg = false){
	typename densityscale<ArgType>::MatrixType ans(x.size(), 1);
	if (lg){
		ans = dpoisarray(x, mu0, stdev, true).array() + ((pi0 > 0) ? std::log(pi0) : -1e100);
	}else{
		ans = dpoisarray(x, mu0, stdev) * pi0;
	}
	return ans;
}

template <class ArgType, class ArgType2, class ArgType3>
inline typename densitynp<ArgType, ArgType2, ArgType3>::MatrixType
dnppois_(const MatrixBase<ArgType> &x, const MatrixBase<ArgType2> &mu0, const MatrixBase<ArgType3> &pi0, 
	const typename MatrixBase<ArgType>::Scalar& stdev, const bool& lg = false){
	EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(ArgType2, ArgType3);
	typename densitynp<ArgType, ArgType2, ArgType3>::MatrixType ans(x.size(), 1);
	if (mu0.size() == 1){
		ans = dnppois_(x, mu0[0], pi0[0], stdev, lg);
	}else{
		if (lg){
			ans = dnppois_(x, mu0[0], pi0[0], stdev, true); 
			for (auto i = 1; i < mu0.size(); ++i){
				ans = logspaceadd(ans, dnpnorm_(x, mu0[i], pi0[i], stdev, true));
			}
		}else{
			ans = dpoisarray(x, mu0, stdev) * pi0;
		}		
	}
	
	return ans;
}

// end poisson density

// begin poisson CDF
template<class ArgType, class ArgType2>
class ppois_functor {
	const ArgType &x;
	const ArgType2 &mu;
	const bool lt;
	const bool lg;
public:
	ppois_functor(const ArgType& x_, const ArgType2 &mu_, const bool &lt_, const bool &lg_) : x(x_), mu(mu_), lt(lt_), lg(lg_) {}

	const typename ScalarBinaryOpTraits<typename ArgType::Scalar, typename ArgType2::Scalar>::ReturnType operator() (Index row, Index col) const {
		return Rf_ppois((double) x.coeff(row), (double) mu.coeff(col), lt, lg);
	}
};

template <class ArgType, class ArgType2>
inline typename densityvec<ArgType, ArgType2>::MatrixType
ppoisarray(const MatrixBase<ArgType>& x, const MatrixBase<ArgType2> & mu0, const typename MatrixBase<ArgType>::Scalar &stdev, const bool &lt = true, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType);
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType2);
	typedef typename densityvec<ArgType, ArgType2>::MatrixType MatrixType;
	return MatrixType::NullaryExpr(x.size(), mu0.size(), ppois_functor<ArgType, ArgType2>(x.derived(), mu0.derived(), lt, lg));
}

template <class ArgType>
inline typename densityscale<ArgType>::MatrixType
ppoisarray(const MatrixBase<ArgType>& x, const typename MatrixBase<ArgType>::Scalar & mu0, const typename MatrixBase<ArgType>::Scalar &stdev, const bool &lt = true, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType);
	return ppoisarray(x, Matrix<typename MatrixBase<ArgType>::Scalar, 1, 1>::Constant(1, mu0), stdev, lt, lg);
}

template <class ArgType>
inline typename densityscale<ArgType>::MatrixType
pnppois_(const MatrixBase<ArgType> &x, const typename MatrixBase<ArgType>::Scalar &mu0, 
	const typename MatrixBase<ArgType>::Scalar &pi0, const typename MatrixBase<ArgType>::Scalar& stdev, const bool &lt = true, const bool& lg = false){
	typename densityscale<ArgType>::MatrixType ans(x.size(), 1);
	if (lg){
		ans = ppoisarray(x, mu0, stdev, lt, true).array() + ((pi0 > 0) ? std::log(pi0) : -1e100);
	}else{
		ans = ppoisarray(x, mu0, stdev, lt) * pi0;
	}

	return ans;
}

template <class ArgType, class ArgType2, class ArgType3>
inline typename densitynp<ArgType, ArgType2, ArgType3>::MatrixType
pnppois_(const MatrixBase<ArgType> &x, const MatrixBase<ArgType2> &mu0, const MatrixBase<ArgType3> &pi0, 
	const typename MatrixBase<ArgType>::Scalar& stdev, const bool &lt = true, const bool& lg = false){
	EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(ArgType2, ArgType3);
	typename densitynp<ArgType, ArgType2, ArgType3>::MatrixType ans(x.size(), 1);
	if (mu0.size() == 1){
		ans = pnppois_(x, mu0.coeff(0), pi0.coeff(0), stdev, lt, lg);
	}else{
		if (lg){
			ans = pnppois_(x, mu0[0], pi0[0], stdev, lt, true);
			for (auto i = 1; i < mu0.size(); ++i){
				ans = logspaceadd(ans, pnppois_(x, mu0[i], pi0[i], stdev, lt, true));
			}
		}else{
			ans = ppoisarray(x, mu0, stdev, lt, false) * pi0;
		}	
	}

	return ans;
}

// end poisson CDF

// begin t CDF

template<class ArgType, class ArgType2>
class pnt_functor {
	const ArgType &x;
	const ArgType2 &mu;
	const typename MatrixBase<ArgType>::Scalar n;
	const bool lt, lg;
public:
	pnt_functor(const ArgType& x_, const ArgType2 &mu_, const typename MatrixBase<ArgType>::Scalar &n_, const bool &lt_, 
		const bool &lg_) : x(x_), mu(mu_), n(n_), lt(lt_), lg(lg_) {}

	const typename ScalarBinaryOpTraits<typename ArgType::Scalar, typename ArgType2::Scalar>::ReturnType operator() (Index row, Index col) const {
		return Rf_pnt((double) x.coeff(row), (double) n, (double) mu.coeff(col), lt, lg);
	}
};

template <class ArgType, class ArgType2>
inline CwiseNullaryOp<pnt_functor<ArgType, ArgType2>, typename densityvec<ArgType, ArgType2>::MatrixType>
ptarray(const MatrixBase<ArgType>& x, const MatrixBase<ArgType2> &mu0, const typename MatrixBase<ArgType>::Scalar &n, const bool &lt = true, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType);
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType2);
	typedef typename densityvec<ArgType, ArgType2>::MatrixType MatrixType;
	return MatrixType::NullaryExpr(x.size(), mu0.size(), pnt_functor<ArgType, ArgType2>(x.derived(), mu0.derived(), n, lt, lg));
}

template <class ArgType>
inline typename densityscale<ArgType>::MatrixType
ptarray(const MatrixBase<ArgType>& x, const typename MatrixBase<ArgType>::Scalar &mu0, 
	const typename MatrixBase<ArgType>::Scalar &n, const bool &lt = true, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArgType);
 	return ptarray(x, Matrix<typename MatrixBase<ArgType>::Scalar, 1, 1>::Constant(1, mu0), n, lt, lg);
}

template <class ArgType>
inline typename densityscale<ArgType>::MatrixType
pnpt_(const MatrixBase<ArgType> &x, const typename MatrixBase<ArgType>::Scalar &mu0,
	const typename MatrixBase<ArgType>::Scalar &pi0, const typename MatrixBase<ArgType>::Scalar& n, const bool &lt = true, const bool& lg = false){
	typename densityscale<ArgType>::MatrixType ans(x.size(), 1);
	
	if (lg){
		ans = ptarray(x, mu0, n, lt, true).array() + ((pi0 > 0) ? std::log(pi0) : -1e100);
	}else{
		ans = ptarray(x, mu0, n, lt, false) * pi0;
	}

	return ans;
}

template <class ArgType, class ArgType2, class ArgType3>
inline typename densitynp<ArgType, ArgType2, ArgType3>::MatrixType
pnpt_(const MatrixBase<ArgType> &x, const MatrixBase<ArgType2> &mu0, const MatrixBase<ArgType3> &pi0, 
	const typename MatrixBase<ArgType>::Scalar& n, const bool &lt = true, const bool& lg = false){
	EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(ArgType2, ArgType3);
	typename densitynp<ArgType, ArgType2, ArgType3>::MatrixType ans(x.size(), 1);
	if (mu0.size() == 1){
		ans = pnpt_(x, mu0.coeff(0), pi0.coeff(0), n, lt, lg);
	}else{
		if (lg){
			ans = pnpt_(x, mu0[0], pi0[0], n, lt, true);
			for (auto i = 1; i < mu0.size(); ++i){
				ans = logspaceadd(ans, pnpt_(x, mu0[i], pi0[i], n, lt, true));
			}
		}else{
			ans = pnormarray(x, mu0, n, lt, false) * pi0;
		}	
	}

	return ans;
}

// end t CDF

// begin discrete t density
// The density of dnt is hard to compute and dnt is computed throught pnt, there is no point to use numerical integration with dnt
// the x + h is not the same as mu - h !!!

template <class ArgType>
inline typename densityscale<ArgType>::MatrixType
ddisctarray(const MatrixBase<ArgType> &x, const typename MatrixBase<ArgType>::Scalar &mu0, 
	const typename MatrixBase<ArgType>::Scalar &n, const typename MatrixBase<ArgType>::Scalar &h, const bool &lg = false){
	// Trapezoid rule. Relative error <= 1e-6 / 12;
	typedef typename densityscale<ArgType>::MatrixType MatrixType;
	MatrixType ans(x.size(), 1);
	ans = logspacesub(ptarray(x + MatrixBase<ArgType>::Constant(x.size(), h), mu0, n, true, true), ptarray(x, mu0, n, true, true));
	// R implementation suffers numerical stability in extreme circumstance, so manually setting a large value
	// use of -500. keep finiteness in likelihood.
	ans = ans.array().isNaN().select(MatrixType::Constant(x.size(), -100.), ans);

	if (lg)
		return ans;
	else
		return ans.array().exp();
}

template <class ArgType, class ArgType2>
inline typename densityvec<ArgType, ArgType2>::MatrixType
ddisctarray(const MatrixBase<ArgType> &x, const MatrixBase<ArgType2> &mu0, 
	const typename MatrixBase<ArgType>::Scalar &n, const typename MatrixBase<ArgType>::Scalar &h, const bool &lg = false){
	typedef typename densityvec<ArgType, ArgType2>::MatrixType MatrixType;
	MatrixType ans(x.size(), mu0.size());
	ans = logspacesub(ptarray(x + MatrixBase<ArgType>::Constant(x.size(), h), mu0, n, true, true), ptarray(x, mu0, n, true, true));
		// R implementation suffers numerical stability in extreme circumstance, so manually setting a large value
	ans = ans.array().isNaN().select(MatrixType::Constant(x.size(), mu0.size(), -100.), ans);

	if (lg)
		return ans;
	else
		return ans.array().exp();
}

template <class ArgType>
inline typename densityscale<ArgType>::MatrixType
dnpdisct_(const MatrixBase<ArgType> &x, const typename MatrixBase<ArgType>::Scalar &mu0, 
	const typename MatrixBase<ArgType>::Scalar &pi0, const typename MatrixBase<ArgType>::Scalar &n, const typename MatrixBase<ArgType>::Scalar& h, const bool& lg = false){
	typename densityscale<ArgType>::MatrixType ans(x.size(), 1);
	if (lg){
		ans = ddisctarray(x, mu0, n, h, true).array() + ((pi0 > 0) ? std::log(pi0) : -1e100);
	}else{
		ans = ddisctarray(x, mu0, n, h, false) * pi0;
	}
	return ans;
}

template <class ArgType, class ArgType2, class ArgType3>
inline typename densitynp<ArgType, ArgType2, ArgType3>::MatrixType
dnpdisct_(const MatrixBase<ArgType> &x, const MatrixBase<ArgType2> &mu0, const MatrixBase<ArgType3> &pi0, 
	const typename MatrixBase<ArgType>::Scalar& n, const typename MatrixBase<ArgType>::Scalar &h, const bool& lg = false){
	EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(ArgType2, ArgType3);
	typename densitynp<ArgType, ArgType2, ArgType3>::MatrixType ans(x.size(), 1);
	if (mu0.size() == 1){
		ans = dnpdisct_(x, mu0.coeff(0), pi0.coeff(0), n, h, lg);
	}else{
		if (lg){
			ans = dnpdisct_(x, mu0[0], pi0[0], n, h, true);
			for (auto i = 1; i < mu0.size(); ++i){
				ans = logspaceadd(ans, dnpdisct_(x, mu0[i], pi0[i], n, h, true));
			}
		}else{
			ans = ddisctarray(x, mu0, n, h, false) * pi0;
		}	
	}
	return ans;
}

// end discrete t density

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
  typedef Matrix<typename ScalarBinaryOpTraits<typename ArgType::Scalar, typename ArgType2::Scalar>::ReturnType,
                 ArgType::ColsAtCompileTime,
                 1,
                 ColMajor,
                 ArgType::MaxColsAtCompileTime,
                 1> MatrixType;
};

template<class ArgType, class ArgType2>
inline typename pnnls_struct<ArgType, ArgType2>::MatrixType
pnnlssum_(const MatrixBase<ArgType> &A, const MatrixBase<ArgType2> &b, const typename MatrixBase<ArgType>::Scalar &sum){
	int m = A.rows() + 1, n = A.cols();
	MatrixXd AA = ((A * sum).colwise() - b).colwise().homogeneous().template cast<double>();
	VectorXd bb = Eigen::VectorXd::Zero(m - 1).homogeneous(), zz(m);
	VectorXd w(n), x(n);
	double rnorm;
	VectorXi index(n);
	int mode, k = 0;
	pnnls_(AA.data(), &m, &m, &n, bb.data(), x.data(), &rnorm, w.data(), zz.data(), index.data(), &mode, &k);
	typename pnnls_struct<ArgType, ArgType2>::MatrixType ans = x.template cast<typename pnnls_struct<ArgType, ArgType2>::MatrixType::Scalar>();
	ans = ans / ans.sum() * sum;
	return ans;
}

// The program pnnqp using eigen
template<class ArgType, class ArgType2>
inline typename pnnls_struct<ArgType, ArgType2>::MatrixType
pnnqp_(const MatrixBase<ArgType> &q, const MatrixBase<ArgType2> &p, const typename MatrixBase<ArgType>::Scalar &sum){
	SelfAdjointEigenSolver<Matrix<typename ArgType::Scalar, ArgType::RowsAtCompileTime, ArgType::RowsAtCompileTime> > eig(q);
	Matrix<typename ArgType::Scalar, ArgType::RowsAtCompileTime, 1> eigvec = eig.eigenvalues();
	Matrix<typename ArgType::Scalar, ArgType::RowsAtCompileTime, ArgType::RowsAtCompileTime> eigmat = eig.eigenvectors();
	int index = (eigvec.array() > eigvec.template tail<1>()[0] * 1e-15).count();
	return pnnlssum_((eigmat.rightCols(index) * eigvec.tail(index).cwiseSqrt().asDiagonal()).transpose(), 
		(eigmat.rightCols(index).transpose() * -p).cwiseQuotient(eigvec.tail(index).cwiseSqrt()), sum);
}

// end pnnls and pnnqp.


}

template<class Type>
class comparemu0
{
private:
	const Eigen::Ref<const Eigen::Matrix<Type, Eigen::Dynamic, 1> > mu0;
public:
	comparemu0(const Eigen::Matrix<Type, Eigen::Dynamic, 1> &mu0_) : mu0(mu0_) {};

	const bool operator()(int x, int y) const {return mu0[x] < mu0[y];};
};


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