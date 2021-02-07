#ifndef miscfuns_h
#define miscfuns_h
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>
#include <unsupported/Eigen/SpecialFunctions>

// diff function
inline Eigen::VectorXd diff_(const Eigen::VectorXd & x){
	return x.tail(x.size() - 1) - x.head(x.size() - 1);
}

namespace Eigen{

template<class ArgType>
struct densityarrayscale {
  typedef Matrix<typename ArgType::Scalar,
                 ArgType::SizeAtCompileTime,
                 1,
                 ColMajor,
                 ArgType::MaxSizeAtCompileTime,
                 1> MatrixType;
};

template<class ArgType, class ArgType2>
struct densityarray {
  typedef Matrix<typename ScalarBinaryOpTraits<typename ArgType::Scalar, typename ArgType2::Scalar>::ReturnType,
                 ArgType::SizeAtCompileTime,
                 ArgType2::SizeAtCompileTime,
                 ColMajor,
                 ArgType::MaxSizeAtCompileTime,
                 ArgType2::MaxSizeAtCompileTime> MatrixType;
};


template <class ArgType>
inline typename densityarrayscale<ArgType>::MatrixType
dnormarray(const Eigen::MatrixBase<ArgType>& x, const double & mu0, const double &stdev, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(ArgType, -1, 1);
	typedef typename densityarrayscale<ArgType>::MatrixType MatrixType;
	MatrixType ans(x.size(), 1);
	ans = (x.array() - mu0).square() / (-2 * stdev * stdev) - (M_LN_SQRT_2PI + std::log(stdev));
	if (lg){
		return ans;
	}else{
		return ans.array().exp();
	}
}

template <class ArgType, class ArgType2>
inline typename densityarray<ArgType, ArgType2>::MatrixType
dnormarray(const Eigen::MatrixBase<ArgType>& x, const Eigen::MatrixBase<ArgType2> & mu0, const double &stdev, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(ArgType, -1, 1);
	EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(ArgType2, -1, 1);
	typedef typename densityarray<ArgType, ArgType2>::MatrixType MatrixType;
	MatrixType ans(x.size(), mu0.size());
	if (mu0.size() == 1){
		ans = dnormarray<ArgType>(x, mu0.coeff(0), stdev, true);
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
inline typename densityarrayscale<ArgType>::MatrixType
dnpnorm_(const Eigen::MatrixBase<ArgType> &x, const double &mu0, const double &pi0, const double& stdev, const bool& lg = false){
	typename densityarrayscale<ArgType>::MatrixType ans(x.size(), 1);
	ans = dnormarray(x, mu0, stdev) * pi0;
	if (lg){
		return ans.array().log();
	}else{
		return ans;
	}
}

template <class ArgType, class ArgType2, class ArgType3>
inline typename densityarrayscale<ArgType>::MatrixType
dnpnorm_(const Eigen::MatrixBase<ArgType> &x, const Eigen::MatrixBase<ArgType2> &mu0, const Eigen::MatrixBase<ArgType3> &pi0, const double& stdev, const bool& lg = false){
	EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(ArgType2, ArgType3);
	typename densityarrayscale<ArgType>::MatrixType ans(x.size(), 1);
	if (mu0.size() == 1){
		ans = dnormarray(x, mu0.coeff(0), stdev) * pi0.coeff(0);
	}else{
		ans = dnormarray(x, mu0, stdev) * pi0;
	}
	if (lg){
		return ans.array().log();
	}else{
		return ans;
	}
}



template<class ArgType, class ArgType2>
class pnormlog_functor {
	const ArgType &x;
	const ArgType2 &mu;
	const double stdev;
	const bool lt;
public:
	pnormlog_functor(const ArgType& x_, const ArgType2 &mu_, const double &stdev_, const bool &lt_) : x(x_), mu(mu_), stdev(stdev_), lt(lt_) {}

	const double operator() (Index row, Index col) const {
		return R::pnorm5(x.coeff(row), mu.coeff(col), stdev, lt, true);
	}
};


template <class ArgType, class ArgType2>
CwiseNullaryOp<pnormlog_functor<ArgType, ArgType2>, typename densityarray<ArgType, ArgType2>::MatrixType>
pnormarraylog(const Eigen::MatrixBase<ArgType>& x, const Eigen::MatrixBase<ArgType2> &mu0, const double &stdev, const bool &lt = true)
{
	typedef typename densityarray<ArgType, ArgType2>::MatrixType MatrixType;
	return MatrixType::NullaryExpr(x.size(), mu0.size(), pnormlog_functor<ArgType, ArgType2>(x.derived(), mu0.derived(), stdev, lt));
}

template <class ArgType>
inline typename densityarrayscale<ArgType>::MatrixType
pnormarraylog(const Eigen::MatrixBase<ArgType>& x, const double &mu0, const double &stdev, const bool &lt = true)
{
 	typedef typename densityarrayscale<ArgType>::MatrixType MatrixType;
 	return pnormarraylog(x, MatrixType::Constant(1, mu0), stdev, lt);
}

template <class ArgType>
inline typename densityarrayscale<ArgType>::MatrixType
pnormarray(const Eigen::MatrixBase<ArgType>& x, const double & mu0, const double &stdev, const bool &lt = true, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(ArgType, -1, 1);
	typedef typename densityarrayscale<ArgType>::MatrixType MatrixType;
	MatrixType ans(x.size(), 1);
	if (lg){
		ans = pnormarraylog(x, mu0, stdev, lt);
	}else{
		if (lt){
			ans = 0.5 * (1 + ((x.array() - mu0) / (stdev * std::sqrt(2))).erf());
		}else{
			ans = 0.5 * ((x.array() - mu0) / (stdev * std::sqrt(2))).erfc();
		}
	}

	return ans;
}

template <class ArgType, class ArgType2>
inline typename densityarray<ArgType, ArgType2>::MatrixType
pnormarray(const Eigen::MatrixBase<ArgType>& x, const Eigen::MatrixBase<ArgType2> & mu0, const double &stdev, const bool &lt = true, const bool &lg = false)
{
	EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(ArgType, -1, 1);
	EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(ArgType2, -1, 1);
	typedef typename densityarray<ArgType, ArgType2>::MatrixType MatrixType;
	MatrixType ans(x.size(), mu0.size());
	if (lg){
		ans = pnormarraylog(x, mu0, stdev, lt);
	}else{
		if (mu0.size() == 1){
			ans = pnormarray(x, mu0.coeff(0), stdev, lt, false);
		}else{
			if (lt){
				ans = 0.5 * (1 + ((mu0.transpose().replicate(x.size(), 1).colwise() - x) / (stdev * -std::sqrt(2))).array().erf());
			}else{
				ans = 0.5 * ((mu0.transpose().replicate(x.size(), 1).colwise() - x) / (stdev * -std::sqrt(2))).array().erfc();
			}		
		}
	
	}

	return ans;
}

template <class ArgType>
inline typename densityarrayscale<ArgType>::MatrixType
pnpnorm_(const Eigen::MatrixBase<ArgType> &x, const double &mu0, const double &pi0, const double& stdev, const bool &lt = true, const bool& lg = false){
	typename densityarrayscale<ArgType>::MatrixType ans(x.size(), 1);
	ans = pnormarray(x, mu0, stdev, lt, false) * pi0;
	if (lg){
		return ans.array().log();
	}else{
		return ans;
	}
}

template <class ArgType, class ArgType2, class ArgType3>
inline typename densityarrayscale<ArgType>::MatrixType
pnpnorm_(const Eigen::MatrixBase<ArgType> &x, const Eigen::MatrixBase<ArgType2> &mu0, const Eigen::MatrixBase<ArgType3> &pi0, const double& stdev, const bool &lt = true, const bool& lg = false){
	EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(ArgType2, ArgType3);
	typename densityarrayscale<ArgType>::MatrixType ans(x.size(), 1);
	if (mu0.size() == 1){
		ans = pnormarray(x, mu0.coeff(0), stdev) * pi0.coeff(0);
	}else{
		ans = pnormarray(x, mu0, stdev) * pi0;
	}
	if (lg){
		return ans.array().log();
	}else{
		return ans;
	}
}

}


// Simple implementaion of density of normal mixture.
// if mu0 a scalar
// inline Eigen::VectorXd dnormarraylog(const Eigen::VectorXd &x, const double &mu0, const double &stdev){
// 	return (x.array() - mu0).square() / (-2 * stdev * stdev) - (M_LN_SQRT_2PI + std::log(stdev));
// }

// inline Eigen::MatrixXd dnormarraylog(const Eigen::VectorXd &x, const Eigen::VectorXd &mu0, const double& stdev){
// 	if (mu0.size() == 1){
// 		return dnormarraylog(x, mu0[0], stdev);
// 	}else{
// 		return (mu0.transpose().replicate(x.size(), 1).colwise() - x).array().square() / (-2 * stdev * stdev) - (M_LN_SQRT_2PI + std::log(stdev));
// 	}
// }

// template<typename T>
// inline Eigen::MatrixXd dnormarray(const Eigen::VectorXd &x, const T &mu0, const double& stdev, const bool& lg = false){
// 	if (lg){
// 		return dnormarraylog(x, mu0, stdev);
// 	}else{
// 		return dnormarraylog(x, mu0, stdev).array().exp();
// 	}
// }

// template<typename T>
// inline Eigen::VectorXd dnpnorm_(const Eigen::VectorXd &x, const T &mu0, const T &pi0, const double& stdev, const bool& lg = false){
// 	if (lg){
// 		return (dnormarray(x, mu0, stdev) * pi0).array().log();
// 	}else{
// 		return dnormarray(x, mu0, stdev) * pi0;
// 	}
// }

// Simple implementaion of CDF of normal mixture.
// struct pnormptr{
// 	pnormptr(const double &mu_, const double &stdev_, const bool& lt_, const bool& lg_ = false) : mu(mu_), stdev(stdev_), lt(lt_), lg(lg_) {};

// 	const double operator()(const double & x) const {
// 		return R::pnorm5(x, mu, stdev, lt, lg);
// 	}

// 	double mu, stdev;
// 	bool lt, lg;
// };

// inline Eigen::VectorXd pnormarray(const Eigen::VectorXd &x, const double &mu0, const double& stdev, const bool &lt = true, const bool &lg = false){
// 	if (lg){
// 		return x.unaryExpr(pnormptr(mu0, stdev, lt, lg));
// 	}else{
// 		if (lt){
// 			return 0.5 * (1 + ((x.array() - mu0) / (stdev * std::sqrt(2))).erf());
// 		}else{
// 			return 0.5 * ((x.array() - mu0) / (stdev * std::sqrt(2))).erfc();
// 		}
// 	}
// }

// inline Eigen::MatrixXd pnormarray(const Eigen::VectorXd &x, const Eigen::VectorXd &mu0, const double& stdev, const bool &lt = true, const bool &lg = false){
// 	if (mu0.size() == 1){
// 		return pnormarray(x, mu0[0], stdev, lt, lg);
// 	}else{
// 		Eigen::MatrixXd ans(x.size(), mu0.size());
// 		if (lg){
// 			const double *mu0ptr = mu0.data();
// 			for (auto i = 0; i < mu0.size(); i++, mu0ptr++){
// 				ans.col(i) = pnormarray(x, *mu0ptr, stdev, lt, lg);
// 			}
// 		}else{
// 			if (lt){
// 				ans = 0.5 * (1 + ((mu0.transpose().replicate(x.size(), 1).colwise() - x) / (stdev * -std::sqrt(2))).array().erf());
// 			}else{
// 				ans = 0.5 * ((mu0.transpose().replicate(x.size(), 1).colwise() - x) / (stdev * -std::sqrt(2))).array().erfc();
// 			}
// 		}
// 		return ans;
// 	}
// }

// template<typename T>
// inline Eigen::VectorXd pnpnorm_(const Eigen::VectorXd &x, const T &mu0, const T &pi0, const double& stdev, const bool &lt = true, const bool& lg = false){
// 	if (lg){
// 		return (pnormarray(x, mu0, stdev, lt, false) * pi0).array().log();
// 	}else{
// 		return pnormarray(x, mu0, stdev, lt, false) * pi0;
// 	}
// }

// Simple implementaion of density of one-parameter normal mixture.
inline Eigen::VectorXd dnormcarraylog(const Eigen::VectorXd &x, const double &mu0, const double &n){
	return ((x.array() - mu0) * std::sqrt(n) / (1 - mu0 * mu0)).square() * -.5 - M_LN_SQRT_2PI - std::log((1 - mu0 * mu0) / std::sqrt(n));
}

inline Eigen::MatrixXd dnormcarraylog(const Eigen::VectorXd &x, const Eigen::VectorXd &mu0, const double &n){
	if (mu0.size() == 1){
		return dnormcarraylog(x, mu0[0], n);
	}else{
		Eigen::MatrixXd temp = mu0.transpose().replicate(x.size(), 1);
		Eigen::MatrixXd stdev = (1 - temp.array().square()) / std::sqrt(n);
		return ((temp.colwise() - x).array() / stdev.array()).square() * -.5 - M_LN_SQRT_2PI - stdev.array().log();
	}
}

template<typename T>
inline Eigen::MatrixXd dnormcarray(const Eigen::VectorXd &x, const T &mu0, const double& n, const bool& lg = false){
	if (lg){
		return dnormcarraylog(x, mu0, n);
	}else{
		return dnormcarraylog(x, mu0, n).array().exp();
	}
}

template<typename T>
inline Eigen::VectorXd dnpnormc_(const Eigen::VectorXd &x, const T &mu0, const T &pi0, const double& n, const bool& lg = false){
	if (lg){
		return (dnormcarray(x, mu0, n) * pi0).array().log();
	}else{
		return dnormcarray(x, mu0, n) * pi0;
	}
}

// Simple implementaion of density of discete normal.
inline Eigen::VectorXd ddiscnormarray(const Eigen::VectorXd &x, const double &mu0, const double &stdev, const double &h, const bool &lg = false){
	// Trapezoid rule. Relative error <= 1e-6 / 12;
	int N = std::max(std::ceil((x.array() - mu0).abs().maxCoeff() * 1e3 * std::pow(h, 1.5)), 5.);
	double delta = h / N;
	Eigen::VectorXd ans(x.size());
	ans.noalias() = (dnormarray(x, mu0, stdev) + dnormarray(x, mu0 - h, stdev)) * 0.5;
	for (int i = 1; i < N; i++){
		ans.noalias() += dnormarray(x, mu0 - delta * i, stdev);
	}
	if (lg){
		return (ans * delta).array().log();
	}else{
		return ans * delta;
	}
}

inline Eigen::MatrixXd ddiscnormarray(const Eigen::VectorXd &x, const Eigen::VectorXd &mu0,
	const double &stdev, const double &h, const bool &lg = false){
	if (mu0.size() == 1){
		return ddiscnormarray(x, mu0[0], stdev, h, lg);
	}else{
		// Trapezoid rule. Relative error <= 1e-6 / 12;
		int N = std::max(std::ceil(std::max(x.maxCoeff() - mu0.minCoeff(), mu0.maxCoeff() - x.minCoeff()) * 1e3 * std::pow(h, 1.5)), 5.);
		double delta = h / N;
		Eigen::MatrixXd ans(x.size(), mu0.size());
		ans.noalias() = (dnormarray(x, mu0, stdev) + dnormarray(x, mu0 - Eigen::VectorXd::Constant(mu0.size(), h), stdev)) * 0.5;
		for (int i = 1; i < N; i++){
			ans.noalias() += dnormarray(x, mu0 - Eigen::VectorXd::Constant(mu0.size(), delta * i), stdev);
		}
		if (lg){
			return (ans * delta).array().log();
		}else{
			return ans * delta;
		}
	}
}

template<typename T>
inline Eigen::VectorXd dnpdiscnorm_(const Eigen::VectorXd &x, const T &mu0, const T &pi0, const double& stdev, const double &h, const bool& lg = false){
	if (lg){
		return (ddiscnormarray(x, mu0, stdev, h) * pi0).array().log();
	}else{
		return ddiscnormarray(x, mu0, stdev, h) * pi0;
	}
}

// Simple implementaion of CDF of discete normal.
inline Eigen::VectorXd pdiscnormarray(const Eigen::VectorXd &x,
	const double &mu0, const double &stdev, const double &h, const bool &lt = true, const bool &lg = false){
	return pnormarray(x, mu0 - h, stdev, lt, lg);
}

inline Eigen::MatrixXd pdiscnormarray(const Eigen::VectorXd &x,
	const Eigen::VectorXd &mu0, const double &stdev, const double &h, const bool &lt = true, const bool &lg = false){
	if (mu0.size() == 1){
		return pdiscnormarray(x, mu0[0], stdev, h, lt, lg);
	}else{
		if (x.size() > mu0.size()){
			return pnormarray(x, mu0 - Eigen::VectorXd::Constant(mu0.size(), h), stdev, lt, lg);
		}else{
			return pnormarray(x + Eigen::VectorXd::Constant(x.size(), h), mu0, stdev, lt, lg);
		}
	}
}

template<typename T>
inline Eigen::VectorXd pnpdiscnorm_(const Eigen::VectorXd &x, const T &mu0, const T &pi0,
	const double& stdev, const double &h, const bool &lt = true, const bool& lg = false){
	if (lg){
		return (pdiscnormarray(x, mu0, stdev, h, lt, false) * pi0).array().log();
	}else{
		return pdiscnormarray(x, mu0, stdev, h, lt, false) * pi0;
	}

}

// Simple implementaion of density of t mixture.
struct dtptr{
	dtptr(const double &mu_, const double &n_) : mu(mu_), n(n_) {};

	const double operator()(const double & x) const {
		return R::dnt(x, n, mu, false);
	}

	double mu, n;
};

inline Eigen::VectorXd dtarray(const Eigen::VectorXd &x, const double &mu0, const double &n, const bool& lg = false){
	if (lg) {
		return x.unaryExpr(dtptr(mu0, n)).array().log();
	}else{
		return x.unaryExpr(dtptr(mu0, n));
	}
}

inline Eigen::MatrixXd dtarray(const Eigen::VectorXd &x, const Eigen::VectorXd &mu0, const double& n, const bool& lg = false){
	if (mu0.size() == 1){
		return dtarray(x, mu0[0], n, lg);
	}else{
		const double *mu0ptr = mu0.data();
		Eigen::MatrixXd ans(x.size(), mu0.size());
		for (auto i = 0; i < mu0.size(); i++, mu0ptr++){
			ans.col(i) = dtarray(x, *mu0ptr, n, lg);
		}
		return ans;
	}
}

template<typename T>
inline Eigen::VectorXd dnpt_(const Eigen::VectorXd &x, const T &mu0, const T &pi0, const double &n, const bool& lg = false){
	if (lg){
		return (dtarray(x, mu0, n) * pi0).array().log();
	}else{
		return dtarray(x, mu0, n) * pi0;
	}		
}

// class for comparison
// sort mixing distribution
namespace Eigen {
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
	};


	template <class ArgType, class RowIndexType, class ColIndexType>
	CwiseNullaryOp<indexing_functor<ArgType,RowIndexType,ColIndexType>, typename indexing_functor<ArgType,RowIndexType,ColIndexType>::MatrixType>
	indexing(const Eigen::MatrixBase<ArgType>& arg, const RowIndexType& row_indices, const ColIndexType& col_indices)
	{
	  typedef indexing_functor<ArgType,RowIndexType,ColIndexType> Func;
	  typedef typename Func::MatrixType MatrixType;
	  return MatrixType::NullaryExpr(row_indices.size(), col_indices.size(), Func(arg.derived(), row_indices, col_indices));
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

inline void sort1(Eigen::VectorXd &x){
	std::sort(x.data(), x.data() + x.size(), std::less<double>());
}

inline void sort2(Eigen::VectorXd &x){
	std::sort(x.data(), x.data() + x.size(), std::greater<double>());
}

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


inline void vecsubassign(Eigen::VectorXd &x, const Eigen::VectorXd &y, const Eigen::VectorXi &index){
	// y is of size index.sum();
	// This function should only be used before Eigen 3.4.
	// In Eigen 3.4 there is a built-in function for slicing.
	int j = 0;
	for (int i = 0; i < x.size(); i++){
		if (index[i] > 0){
			x[i] = y[j];
			j++;
		}
	}
}

inline Eigen::VectorXd nnls(const Eigen::MatrixXd &A, const Eigen::VectorXd &b){
	// fnnls by Bros and Jong (1997)
	int n = A.cols();
	Eigen::VectorXi p = Eigen::VectorXi::Zero(n), index;
	Eigen::VectorXd x = Eigen::VectorXd::Zero(n), ZX = A.transpose() * b, s(n), one = Eigen::VectorXd::Ones(n);
	Eigen::MatrixXd ZZ = A.transpose() * A;
	Eigen::VectorXd w = ZX - ZZ * x;
	int maxind, iter = 0;
	double alpha, tol = std::max(ZX.array().abs().maxCoeff(), ZZ.array().abs().maxCoeff()) * 10 * std::numeric_limits<double>::epsilon();
	while ((p.array() == 0).count() > 0 & (p.array() == 0).select(w, -1 * one).maxCoeff(&maxind) > tol & iter < 3 * n){
		p[maxind] = 1;
		index.lazyAssign(index2num(p));
		s.setZero();
		vecsubassign(s, indexing(ZZ, index, index).llt().solve(indexing(ZX, index, Eigen::VectorXi::Zero(1))), p);
		while ((p.array() > 0).select(s, one).minCoeff() <= 0){
			alpha = indexing(x.cwiseQuotient(x - s).eval(), index2num(((p.array() > 0).select(s, one).array() <= 0).cast<int>()), Eigen::VectorXi::Zero(1)).minCoeff();
			x.noalias() += alpha * (s - x);
			p.array() *= (x.array() > 0).cast<int>();
			index.lazyAssign(index2num(p));
			s.setZero();
			vecsubassign(s, indexing(ZZ, index, index).llt().solve(indexing(ZX, index, Eigen::VectorXi::Zero(1))), p);
		}
		x = s;
		w = ZX - ZZ * x;
		iter++;
	}
	return x;
}

inline Eigen::VectorXd pnnlssum_(const Eigen::MatrixXd &A, const Eigen::VectorXd &b, const double &sum){
	int m = A.rows(), n = A.cols();
	Eigen::MatrixXd AA = ((A * sum).colwise() - b).colwise().homogeneous();
	Eigen::VectorXd x(n), bb = Eigen::VectorXd::Zero(m).homogeneous();
	x = nnls(AA, bb);
	x = x / x.sum() * sum;
	return x;
}

// The program pnnqp using LLT
inline Eigen::VectorXd pnnqp_(const Eigen::MatrixXd &q, const Eigen::VectorXd &p, const double &sum){
	Eigen::LLT<Eigen::MatrixXd> llt;
	llt.compute(q);
	return pnnlssum_(llt.matrixU(), llt.matrixL().solve(-p), sum);
}

#endif