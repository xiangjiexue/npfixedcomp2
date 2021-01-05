// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// pnnlssum
Eigen::VectorXd pnnlssum(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const double& sum);
RcppExport SEXP _npfixedcomp2_pnnlssum(SEXP ASEXP, SEXP bSEXP, SEXP sumSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type b(bSEXP);
    Rcpp::traits::input_parameter< const double& >::type sum(sumSEXP);
    rcpp_result_gen = Rcpp::wrap(pnnlssum(A, b, sum));
    return rcpp_result_gen;
END_RCPP
}
// pnnqp
Eigen::VectorXd pnnqp(const Eigen::MatrixXd& q, const Eigen::VectorXd& p, const double& sum);
RcppExport SEXP _npfixedcomp2_pnnqp(SEXP qSEXP, SEXP pSEXP, SEXP sumSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type q(qSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type p(pSEXP);
    Rcpp::traits::input_parameter< const double& >::type sum(sumSEXP);
    rcpp_result_gen = Rcpp::wrap(pnnqp(q, p, sum));
    return rcpp_result_gen;
END_RCPP
}
// diff
Eigen::VectorXd diff(const Eigen::VectorXd& x);
RcppExport SEXP _npfixedcomp2_diff(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(diff(x));
    return rcpp_result_gen;
END_RCPP
}
// dnpnorm
Eigen::VectorXd dnpnorm(const Eigen::VectorXd& x, const Eigen::VectorXd& mu0, const Eigen::VectorXd& pi0, const double& stdev, const bool& lg);
RcppExport SEXP _npfixedcomp2_dnpnorm(SEXP xSEXP, SEXP mu0SEXP, SEXP pi0SEXP, SEXP stdevSEXP, SEXP lgSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type mu0(mu0SEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type pi0(pi0SEXP);
    Rcpp::traits::input_parameter< const double& >::type stdev(stdevSEXP);
    Rcpp::traits::input_parameter< const bool& >::type lg(lgSEXP);
    rcpp_result_gen = Rcpp::wrap(dnpnorm(x, mu0, pi0, stdev, lg));
    return rcpp_result_gen;
END_RCPP
}
// npnormll_
Rcpp::List npnormll_(const Eigen::VectorXd& data, const Eigen::VectorXd& mu0fixed, const Eigen::VectorXd& pi0fixed, const double& beta, const Eigen::VectorXd& initpt, const Eigen::VectorXd& initpr, const Eigen::VectorXd& gridpoints, const double& tol, const int& maxit, const bool& verbose);
RcppExport SEXP _npfixedcomp2_npnormll_(SEXP dataSEXP, SEXP mu0fixedSEXP, SEXP pi0fixedSEXP, SEXP betaSEXP, SEXP initptSEXP, SEXP initprSEXP, SEXP gridpointsSEXP, SEXP tolSEXP, SEXP maxitSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type mu0fixed(mu0fixedSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type pi0fixed(pi0fixedSEXP);
    Rcpp::traits::input_parameter< const double& >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type initpt(initptSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type initpr(initprSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type gridpoints(gridpointsSEXP);
    Rcpp::traits::input_parameter< const double& >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< const int& >::type maxit(maxitSEXP);
    Rcpp::traits::input_parameter< const bool& >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(npnormll_(data, mu0fixed, pi0fixed, beta, initpt, initpr, gridpoints, tol, maxit, verbose));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_npfixedcomp2_pnnlssum", (DL_FUNC) &_npfixedcomp2_pnnlssum, 3},
    {"_npfixedcomp2_pnnqp", (DL_FUNC) &_npfixedcomp2_pnnqp, 3},
    {"_npfixedcomp2_diff", (DL_FUNC) &_npfixedcomp2_diff, 1},
    {"_npfixedcomp2_dnpnorm", (DL_FUNC) &_npfixedcomp2_dnpnorm, 5},
    {"_npfixedcomp2_npnormll_", (DL_FUNC) &_npfixedcomp2_npnormll_, 10},
    {NULL, NULL, 0}
};

RcppExport void R_init_npfixedcomp2(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
