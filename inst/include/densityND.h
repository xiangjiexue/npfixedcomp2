#ifndef density2D_h
#define density2D_h
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>
#include <unsupported/Eigen/SpecialFunctions>

// diff function
// inline Eigen::VectorXd diff_(const Eigen::VectorXd & x){
// 	return x.tail(x.size() - 1) - x.head(x.size() - 1);
// }

namespace Eigen{

template<class ArgType>
struct identitynull {
  typedef Matrix<typename ArgType::Scalar,
                 ArgType::RowsAtCompileTime,
                 ArgType::ColsAtCompileTime,
                 ColMajor,
                 ArgType::MaxRowsAtCompileTime,
                 ArgType::MaxColsAtCompileTime> MatrixType;
};

template<class ArgType, class ArgType2, class ArgType3>
struct densityNDvec {
  typedef Matrix<typename ScalarBinaryOpTraits<typename ArgType::Scalar, typename ScalarBinaryOpTraits<typename ArgType2::Scalar, typename ArgType3::Scalar>::ReturnType>::ReturnType,
                 ArgType::RowsAtCompileTime,
                 ArgType2::RowsAtCompileTime,
                 ColMajor,
                 ArgType::MaxRowsAtCompileTime,
                 ArgType2::MaxRowsAtCompileTime> MatrixType;
};

template<class ArgType, class ArgType2, class ArgType3, class ArgType4>
struct densityNDnp {
  typedef Matrix<typename ScalarBinaryOpTraits<typename ScalarBinaryOpTraits<typename ArgType::Scalar, typename ArgType2::Scalar>::ReturnType, typename ScalarBinaryOpTraits<typename ArgType3::Scalar, typename ArgType4::Scalar>::ReturnType>::ReturnType,
                 ArgType::RowsAtCompileTime,
                 1,
                 ColMajor,
                 ArgType::MaxRowsAtCompileTime,
                 1> MatrixType;
};

template <class ArgType, class ArgType2, class ArgType3>
inline typename densityNDvec<ArgType, ArgType2, ArgType3>::MatrixType
dnormNDarray(const MatrixBase<ArgType>& x, const MatrixBase<ArgType2> & mu0, const MatrixBase<ArgType3> &stdev, const bool &lg = false)
{
	typedef typename densityNDvec<ArgType, ArgType2, ArgType3>::MatrixType MatrixType;
	MatrixType ans(x.rows(), mu0.rows());
	LLT<typename identitynull<ArgType3>::MatrixType> stdevdecomp(stdev);
	ans = MatrixType::NullaryExpr(x.rows(), mu0.rows(), [&x, &mu0, &stdevdecomp](Eigen::Index i, Eigen::Index j){
		return stdevdecomp.solve((x.row(i) - mu0.row(j)).transpose()).dot(x.row(i) - mu0.row(j));
	}).array() * -.5 - (M_LN_SQRT_2PI * x.cols() + .5 * std::log(stdev.determinant()));
	if (lg){
		return ans;
	}else{
		return ans.array().exp();
	}
}


template <class ArgType, class ArgType2, class ArgType3, class ArgType4>
inline typename densityNDnp<ArgType, ArgType2, ArgType3, ArgType4>::MatrixType
dnpnormND_(const MatrixBase<ArgType> &x, const MatrixBase<ArgType2> &mu0, const MatrixBase<ArgType3> &pi0, const MatrixBase<ArgType4> &stdev, const bool& lg = false){
	typename densityNDnp<ArgType, ArgType2, ArgType3, ArgType4>::MatrixType ans(x.rows(), 1);
	ans = dnormNDarray(x, mu0, stdev, false) * pi0;
	if (lg){
		return ans.array().log();
	}else{
		return ans;
	}
}

}


#endif