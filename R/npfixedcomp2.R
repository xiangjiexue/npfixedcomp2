#' npfixedcomp2
#' 
#' npfixedcomp2
#' 
#' @docType package
#' @author Xiangjie Xue
#' @import Rcpp RcppEigen nspmix
#' @importFrom utils getFromNamespace
#' @importFrom stats pnorm pt qnorm qt
#' @useDynLib npfixedcomp2
#' @name npfixedcomp2
NULL

initial.npnorm = utils::getFromNamespace("initial.npnorm", "nspmix")
gridpoints.npnorm = utils::getFromNamespace("gridpoints.npnorm", "nspmix")

#' Computing non-parametric mixing distribution
#' 
#' These functions are used to make the object for computing the non-paramtric mixing
#' distribution or estimating the proportion of zero using non-parametric methods.
#'
#' This is a generic function for making the object for computing the non-parametric
#' mixing distribution or estimating the proportion of zero.
#'
#' current implemented families are:
#'
#' - npnormll : normal density using maximum likelihood (Chapter 3). The default beta is 1.
#'
#' - npnormllw : Binned version of normal density using maximum likelihood (Chapter 6).
#' The default beta is 1 and the default order is -3.
#'
#' - npnormcvm : normal density using the Cramer-von Mises distance (Chapter 5). The default beta is 1.
#'
#' - npnormcvmw : Binned version of normal density using the Cramer-von Mises distance (Chapter 6).
#' The default beta is 1 and the default order is -3.
#'
#' - npnormad : normal density using the Anderson-Darling distance (Chapter 5). The default beta is 1.
#'
#' - npnormadw : Binned version of normal density using the Anderson-Darling distance (Chapter 6)
#' The default beta is 1 and the default order is -3.
#'
#' - nptll : t-density using maximum likelihood (Chapter 3). The default beta is infinity (normal distribution).
#'
#' - npnormcll : the one-parameter normal distribution used for approximating the sample
#' correlation coefficients using maximum likelihood. This does not have a
#' corresponding estimation of zero due to incompleted theory (Chapter 8).
#' There is no default beta. The structure beta is the number of observations.
#'
#' The default method used is npnormll.
#'
#' @title Computing non-parametric mixing distribution
#' @param v a vector of observations
#' @param mu0 A vector of support points
#' @param pi0 A vector of weights corresponding to the support points
#' @param beta structual parameter.
#' @param order the parameter for the binned version.
#' @param method An implemented family; see details
#' @param ... parameters above passed to the specific method
#' @examples
#' data = rnorm(500, c(0, 2))
#' pi0 = 0.5
#' computemixdist(data, pi0 = pi0, method = "npnormll")
#' computemixdist(data, pi0 = pi0, method = "npnormcvm")
#' computemixdist(data, pi0 = pi0, method = "nptll")
#' @export
computemixdist = function(v, method = "npnormll", ...){
  f = match.fun(paste0("computemixdist.", method))
  f(v = v, ...)
}

#' @rdname computemixdist
#' @export
computemixdist.npnormll = function(v, mu0, pi0, beta, order, ...){
  if (missing(mu0)) {mu0 = 0}
  if (missing(pi0)) {pi0 = 0}
  if (missing(beta)) {beta = 1}
  v1 = npnorm(v)
  init = initial.npnorm(v1)
  k = npnormll_(v, mu0, pi0, beta, init$mix$pt, init$mix$pr, gridpoints.npnorm(v1, beta = beta), ...)
  attr(k, "class") = "nspmix"
  
  k
}

#' @rdname computemixdist
#' @export
computemixdist.npnormcvm = function(v, mu0, pi0, beta, order, ...){
  if (missing(mu0)) {mu0 = 0}
  if (missing(pi0)) {pi0 = 0}
  if (missing(beta)) {beta = 1}
  v1 = npnorm(sort(v))
  init = initial.npnorm(v1)
  k = npnormcvm_(v1$v, mu0, pi0, beta, init$mix$pt, init$mix$pr, gridpoints.npnorm(v1, beta = beta), ...)
  attr(k, "class") = "nspmix"
  
  k
}

#' @rdname computemixdist
#' @export
computemixdist.npnormad = function(v, mu0, pi0, beta, order, ...){
  if (missing(mu0)) {mu0 = 0}
  if (missing(pi0)) {pi0 = 0}
  if (missing(beta)) {beta = 1}
  v1 = npnorm(sort(v))
  init = initial.npnorm(v1)
  k = npnormad_(v1$v, mu0, pi0, beta, init$mix$pt, init$mix$pr, gridpoints.npnorm(v1, beta = beta), ...)
  attr(k, "class") = "nspmix"
  
  k
}

#' @rdname computemixdist
#' @export
computemixdist.npnormcll = function(v, mu0, pi0, beta, order, ...){
  if (missing(mu0)) {mu0 = 0}
  if (missing(pi0)) {pi0 = 0}
  v1 = npnorm(atanh(v))
  init = initial.npnorm(v1, beta = 1 / sqrt(beta - 3))
  k = npnormcll_(v, mu0, pi0, beta, tanh(init$mix$pt), init$mix$pr, 
                tanh(gridpoints.npnorm(v1, beta = 1 / sqrt(beta - 3))), ...)
  attr(k, "class") = "nspmix"
  
  k
}

#' @rdname computemixdist
#' @export
computemixdist.nptll = function(v, mu0, pi0, beta, order, ...){
  if (missing(mu0)) {mu0 = 0}
  if (missing(pi0)) {pi0 = 0}
  if (missing(beta)) {beta = Inf}
  v1 = npnorm(qnorm(pt(v, df = beta)))
  init = initial.npnorm(v1)
  k = nptll_(v, mu0, pi0, beta, qt(pnorm(init$mix$pt), df = beta), init$mix$pr, 
                 qt(pnorm(gridpoints.npnorm(v1, beta = 1)), df = beta), ...)
  attr(k, "class") = "nspmix"
  
  k
}