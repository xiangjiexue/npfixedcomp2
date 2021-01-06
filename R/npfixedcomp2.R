#' npfixedcomp2
#' 
#' npfixedcomp2
#' 
#' @docType package
#' @author Xiangjie Xue
#' @import Rcpp RcppEigen nspmix
#' @importFrom utils getFromNamespace
#' @useDynLib npfixedcomp2
#' @name npfixedcomp2
NULL

initial.npnorm = utils::getFromNamespace("initial.npnorm", "nspmix")
gridpoints.npnorm = utils::getFromNamespace("gridpoints.npnorm", "nspmix")

#' @export
computemixdist = function(v, method = "npnormll", ...){
  f = match.fun(paste0("computemixdist.", method))
  f(v = v, ...)
}

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