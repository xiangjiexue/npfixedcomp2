#' The package npfixedcomp2
#' 
#' This package is for computing the estimates of a mixing distribution non-parametrically
#' for some implemented family functions using various loss functions. It extends
#' the estimation methods proposed by Wang (2007) in following ways:
#' 
#' - The mixing distribution can be estimated with component fixed.
#' 
#' - The mixing distribution can be estimated simultaneously with estimating the
#' proportion of zero.
#' 
#' - The mixing distribution can be estimated using other loss function (in
#' addition to maximum likelihood).
#' 
#' Solvers for finding new support points:
#' 
#' For the gradient function with derivative supplied, it uses the improved
#' Brent's method by Zhang (2011), while the derivative-free solver is 
#' Successive Parabolic Interpolation.
#' 
#' @references 
#' Wang, Yong. "On Fast Computation of the Non-Parametric Maximum Likelihood Estimate of a Mixing Distribution." Journal of the Royal Statistical Society. Series B (Statistical Methodology) 69, no. 2 (2007): 185-98. \url{http://www.jstor.org/stable/4623262}.
#' @docType package
#' @author Xiangjie Xue
#' @import Rcpp RcppEigen nspmix
#' @importFrom utils getFromNamespace
#' @importFrom stats pnorm pt qnorm qt cov cov2cor
#' @importFrom grDevices rainbow
#' @importFrom graphics abline lines points
#' @useDynLib npfixedcomp2
#' @name npfixedcomp2
NULL

initial.npnorm = utils::getFromNamespace("initial.npnorm", "nspmix")
gridpoints.npnorm = utils::getFromNamespace("gridpoints.npnorm", "nspmix")
initial.nppois = utils::getFromNamespace("initial.nppois", "nspmix")
gridpoints.nppois = utils::getFromNamespace("gridpoints.nppois", "nspmix")

#' Bin the continuous data.
#'
#' This function bins the continuous data using the equal-width bin in order to
#' speed up some functions in this package.
#'
#' h is taken as 10^(-order)
#'
#' the observations are rounded down to the bins ..., -h, 0, h, ...
#'
#' To further speed up the process, omit the bin that has 0 count.
#'
#' @title Bin the continuous data.
#' @param data the observation to be binned.
#' @param order see the details
#' @return a list with v be the representative value of each bin and w be the count in the corresponding bin.
#' @export
bin = function(data, order = -2){
  h = 10^order
  data = floor(data / h)
  rg = range(data)
  t = tabulate(data - rg[1] + 1)
  index = t != 0
  list(v = (h * (rg[1] : rg[2]))[index], w = t[index])
}

#' Computing non-parametric mixing distribution
#' 
#' These functions are used to compute the estimates of a mixing distribution
#' non-parametrically using various methods, with possibly fixed components.
#'
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
#' - nptllw : Binned version of non-centrol t-distribution using maximum likelihood. 
#' The default beta is infinity (normal distribution) and the default order is -3.
#'
#' - npnormcll : the one-parameter normal distribution used for approximating the sample
#' correlation coefficients using maximum likelihood. This does not have a
#' corresponding estimation of zero due to incompleted theory (Chapter 8).
#' There is no default beta. The structure beta is the number of observations.
#' 
#' - nppoisll : poisson mixture using maximum likelihood. There is no structural parameter, and hence
#' the template will pass beta but this beta will never be referenced.
#' 
#' - npnorm2Dll : bi-variate normal mixture using maximum likelihood. The 
#' structural parameter is the covariance matrix. Currently this implementation
#' is experimental, and it could be very slow. Finding the new support points
#' uses L-BFGS-B method provided by https://github.com/yixuan/LBFGSpp/.
#'
#' The default method used is npnormll.
#'
#' @title Computing non-parametric mixing distribution
#' @param v a vector of observations
#' @param mu0 A vector of support points
#' @param pi0 A vector of weights corresponding to the support points
#' @param beta structual parameter.
#' @param order the parameter for the binned version.
#' @param mix initial mixing distribution.
#' @param gridpoints a vector of gridpoints to evaluate new support points
#' @param method An implemented family; see details
#' @param ... parameters above passed to the specific method
#' @examples
#' set.seed(123)
#' pi0 = 0
#' data = rnorm(1000, mean = c(0, 2))
#' system.time(r <- computemixdist(data, pi0 = pi0, method = "npnormll"))
#' r
#' system.time(r <- computemixdist(data, pi0 = pi0, method = "npnormcvm"))
#' r
#' system.time(r <- computemixdist(data, pi0 = pi0, method = "nptll"))
#' r
#' system.time(r <- computemixdist(data, pi0 = pi0, method = "npnormad"))
#' r
#' system.time(r <- computemixdist(tanh(data), pi0 = pi0, method = "npnormcll", beta = 4))
#' r
#' system.time(r <- computemixdist(data, pi0 = pi0, method = "npnormllw", order = -2))
#' r
#' system.time(r <- computemixdist(data, pi0 = pi0, method = "npnormcvmw", order = -2))
#' r
#' system.time(r <- computemixdist(data, pi0 = pi0, method = "npnormadw", order = -2))
#' r
#' system.time(r <- computemixdist(data, pi0 = pi0, method = "nptllw", order = -2))
#' r
#' data2 = rpois(1000, c(0, 2))
#' system.time(r <- computemixdist(data2, pi0 = pi0, method = "nppoisll"))
#' r
#' data3 = matrix(rnorm(1000), 500, 2)
#' system.time(r <- computemixdist(data3, method = "npnorm2Dll"))
#' r
#' @export
computemixdist = function(v, method = "npnormll", ...){
  f = match.fun(paste0("computemixdist.", method))
  f(v = v, ...)
}

#' computing non-parametric mixing distribution with estimated proportion at 0
#'
#' These function are for computing non-parametric mixing
#' distribution with estimated proportion at 0. Different families will
#' have different threshold values.
#'
#' The parameters are listed as follows:
#'
#' - tol: tolerance to stop the code.
#'
#' - verbose: logical; Whether to print the intermediate results.
#'
#' It is not shown in the parameter section since various method have different
#' default threshold values and this function essentially calls the class method
#' in the object.
#'
#' The full list of implemented families is in \code{\link{computemixdist}}.
#' 
#' IMPORTANT: although estpi0 is implemented for nptll and npnormcll, the theory
#' is not yet done, and hence it is recommand to use with precaution.
#'
#' @title Computing non-parametric mixing distribution with estimated proportion at 0
#' @param v observations
#' @param beta structural parameter
#' @param val thresholding function
#' @param order the parameter for the binned version.
#' @param mix initial mixing distribution.
#' @param gridpoints a vector of gridpoints to evaluate new support points
#' @param method An implemented family; see details
#' @param ... parameters above passed to the specific method.
#' @examples
#' data = rnorm(500, c(0, 2))
#' system.time(r <- estpi0(data, method = "npnormll", val = 2, verbose = TRUE))
#' r
#' system.time(r <- estpi0(data, method = "npnormllw", val = 2, verbose = TRUE, order = -2))
#' r
#' system.time(r <- estpi0(data, method = "npnormcvm", val = 0.1, verbose = TRUE))
#' r
#' system.time(r <- estpi0(data, method = "npnormcvmw", val = 0.1, verbose = TRUE, order = -2))
#' r
#' system.time(r <- estpi0(data, method = "npnormad", val = 1, verbose = TRUE))
#' r
#' system.time(r <- estpi0(data, method = "npnormadw", val = 1, verbose = TRUE, order = -2))
#' r
#' system.time(r <- estpi0(data, method = "nptll", val = 2, verbose = TRUE))
#' r
#' system.time(r <- estpi0(tanh(data), method = "npnormcll", val = 2, beta = 4, verbose = TRUE))
#' r
#' system.time(r <- estpi0(data, method = "nptllw", val = 2, verbose = TRUE))
#' r
#' data2 = rpois(1000, c(0, 2))
#' system.time(r <- estpi0(data2, method = "nppoisll", val = 2))
#' r
#' @export
estpi0 = function(v, method = "npnormll", ...){
  f = match.fun(paste0("estpi0.", method))
  f(v = v, ...)
}

#' @rdname computemixdist
#' @export
computemixdist.npnormll = function(v, mu0, pi0, beta, order, mix = NULL, gridpoints = NULL, ...){
  if (missing(mu0)) {mu0 = 0}
  if (missing(pi0)) {pi0 = 0}
  if (missing(beta)) {beta = 1}
  v1 = npnorm(v)
  if (is.null(gridpoints)) {gridpoints = gridpoints.npnorm(v1, beta = beta)}
  init = initial.npnorm(v1, beta = beta, mix = mix)
  k = npnormll_(v, mu0, pi0, beta, init$mix$pt, init$mix$pr, sort(gridpoints), ...)
  attr(k, "class") = "nspmix"
  
  k
}

#' @rdname estpi0
#' @export
estpi0.npnormll = function(v, beta, val, order, mix = NULL, gridpoints = NULL, ...){
  if (missing(beta)) {beta = 1}
  v1 = npnorm(v)
  if (is.null(gridpoints)) {gridpoints = gridpoints.npnorm(v1, beta = beta)}
  init = initial.npnorm(v1, beta = beta, mix = mix)
  k = estpi0npnormll_(v, beta, val, init$mix$pt, init$mix$pr, sort(gridpoints), ...)
  attr(k, "class") = "nspmix"
  
  k
}

#' @rdname computemixdist
#' @export
computemixdist.npnormcvm = function(v, mu0, pi0, beta, order, mix = NULL, gridpoints = NULL, ...){
  if (missing(mu0)) {mu0 = 0}
  if (missing(pi0)) {pi0 = 0}
  if (missing(beta)) {beta = 1}
  v1 = npnorm(sort(v))
  if (is.null(gridpoints)) {gridpoints = c(gridpoints.npnorm(v1, beta = beta), v1$v[1] - 3 * beta, v1$v[length(v1$v)] + 3 * beta)}
  init = initial.npnorm(v1, beta = beta, mix = mix)
  k = npnormcvm_(v1$v, mu0, pi0, beta, init$mix$pt, init$mix$pr, sort(gridpoints), ...)
  attr(k, "class") = "nspmix"
  
  k
}

#' @rdname estpi0
#' @export
estpi0.npnormcvm = function(v, beta, val, order, mix = NULL, gridpoints = NULL, ...){
  if (missing(beta)) {beta = 1}
  v1 = npnorm(sort(v))
  if (is.null(gridpoints)) {gridpoints = c(gridpoints.npnorm(v1, beta = beta), v1$v[1] - 3 * beta, v1$v[length(v1$v)] + 3 * beta)}
  init = initial.npnorm(v1, beta = beta, mix = mix)
  k = estpi0npnormcvm_(v1$v, beta, val, init$mix$pt, init$mix$pr, sort(gridpoints), ...)
  attr(k, "class") = "nspmix"
  
  k
}

#' @rdname computemixdist
#' @export
computemixdist.npnormad = function(v, mu0, pi0, beta, order, mix = NULL, gridpoints = NULL, ...){
  if (missing(mu0)) {mu0 = 0}
  if (missing(pi0)) {pi0 = 0}
  if (missing(beta)) {beta = 1}
  v1 = npnorm(sort(v))
  if (is.null(gridpoints)) {gridpoints = c(gridpoints.npnorm(v1, beta = beta), v1$v[1] - 3 * beta, v1$v[length(v1$v)] + 3 * beta)}
  init = initial.npnorm(v1, beta = beta, mix = mix)
  k = npnormad_(v1$v, mu0, pi0, beta, init$mix$pt, init$mix$pr, sort(gridpoints), ...)
  attr(k, "class") = "nspmix"
  
  k
}

#' @rdname estpi0
#' @export
estpi0.npnormad = function(v, beta, val, order, mix = NULL, gridpoints = NULL, ...){
  if (missing(beta)) {beta = 1}
  v1 = npnorm(sort(v))
  init = initial.npnorm(v1, beta = beta, mix = mix)
  if (is.null(gridpoints)) {gridpoints = c(gridpoints.npnorm(v1, beta = beta), v1$v[1] - 3 * beta, v1$v[length(v1$v)] + 3 * beta)}
  k = estpi0npnormad_(v1$v, beta, val, init$mix$pt, init$mix$pr, sort(gridpoints), ...)
  attr(k, "class") = "nspmix"
  
  k
}

#' @rdname computemixdist
#' @export
computemixdist.npnormcll = function(v, mu0, pi0, beta, order, mix = NULL, gridpoints = NULL, ...){
  if (missing(mu0)) {mu0 = 0}
  if (missing(pi0)) {pi0 = 0}
  v1 = npnorm(atanh(v))
  if (is.null(gridpoints)) {gridpoints = tanh(gridpoints.npnorm(v1, beta = 1 / sqrt(beta - 3)))}
  if (is.null(mix)){
    init = initial.npnorm(v1, beta = 1 / sqrt(beta - 3))
  }else{
    init = initial.npnorm(v1, beta = 1 / sqrt(beta - 3),
                          mix = list(pt = atanh(mix$pt), pr = mix$pr))
  }
  k = npnormcll_(v, mu0, pi0, beta, tanh(init$mix$pt), init$mix$pr, 
                 sort(gridpoints), ...)
  attr(k, "class") = "nspmix"
  
  k
}

#' @rdname estpi0
#' @export
estpi0.npnormcll = function(v, beta, val, order, mix = NULL, gridpoints = NULL, ...){
  v1 = npnorm(atanh(v))
  if (is.null(gridpoints)) {gridpoints = tanh(gridpoints.npnorm(v1, beta = 1 / sqrt(beta - 3)))}
  if (is.null(mix)){
    init = initial.npnorm(v1, beta = 1 / sqrt(beta - 3))
  }else{
    init = initial.npnorm(v1, beta = 1 / sqrt(beta - 3),
                          mix = list(pt = atanh(mix$pt), pr = mix$pr))
  }
  k = estpi0npnormcll_(v, beta, val, tanh(init$mix$pt), init$mix$pr, 
                       sort(gridpoints), ...)
  attr(k, "class") = "nspmix"
  
  k
}

#' @rdname computemixdist
#' @export
computemixdist.nptll = function(v, mu0, pi0, beta, order, mix = NULL, gridpoints = NULL, ...){
  if (missing(mu0)) {mu0 = 0}
  if (missing(pi0)) {pi0 = 0}
  if (missing(beta)) {beta = Inf}
  v1 = npnorm(qnorm(pt(v, df = beta)))
  if (is.null(gridpoints)) {gridpoints = qt(pnorm(gridpoints.npnorm(v1, beta = 1)), df = beta)}
  if (is.null(mix)){
    init = initial.npnorm(v1, beta = 1)
  }else{
    init = initial.npnorm(v1, beta = 1,
                          mix = list(pt = qnorm(pt(mix$pt, df = beta)), pr = mix$pr))   
  }
  k = nptll_(v, mu0, pi0, beta, qt(pnorm(init$mix$pt), df = beta), init$mix$pr, 
             sort(gridpoints), ...)
  attr(k, "class") = "nspmix"
  
  k
}

#' @rdname estpi0
#' @export
estpi0.nptll = function(v, beta, val, order, mix = NULL, gridpoints = NULL, ...){
  if (missing(beta)) {beta = Inf}
  v1 = npnorm(qnorm(pt(v, df = beta)))
  if (is.null(gridpoints)) {gridpoints = qt(pnorm(gridpoints.npnorm(v1, beta = 1)), df = beta)}
  if (is.null(mix)){
    init = initial.npnorm(v1, beta = 1)
  }else{
    init = initial.npnorm(v1, beta = 1,
                          mix = list(pt = qnorm(pt(mix$pt, df = beta)), pr = mix$pr))   
  }
  k = estpi0nptll_(v, beta, val, qt(pnorm(init$mix$pt), df = beta), init$mix$pr, 
                   sort(gridpoints), ...)
  attr(k, "class") = "nspmix"
  
  k
}

#' @rdname computemixdist
#' @export
computemixdist.npnormllw = function(v, mu0, pi0, beta, order = -3, mix = NULL, gridpoints = NULL, ...){
  if (missing(mu0)) {mu0 = 0}
  if (missing(pi0)) {pi0 = 0}
  if (missing(beta)) {beta = 1}
  v1 = bin(v, order)
  v2 = npnorm(v = v1$v, w = v1$w)
  if (is.null(gridpoints)) {gridpoints = gridpoints.npnorm(v2, beta = beta)}
  init = initial.npnorm(v2, beta = beta, mix = mix)
  k = npnormllw_(v1$v, v1$w, mu0, pi0, beta, 10^order, init$mix$pt, init$mix$pr, sort(gridpoints), ...)
  attr(k, "class") = "nspmix"
  
  k
}

#' @rdname estpi0
#' @export
estpi0.npnormllw = function(v, beta, val, order = -3, mix = NULL, gridpoints = NULL, ...){
  if (missing(beta)) {beta = 1}
  v1 = bin(v, order)
  v2 = npnorm(v = v1$v, w = v1$w)
  if (is.null(gridpoints)) {gridpoints = gridpoints.npnorm(v2, beta = beta)}
  init = initial.npnorm(v2, beta = beta, mix = mix)
  k = estpi0npnormllw_(v1$v, v1$w, beta, 10^order, val, init$mix$pt, init$mix$pr, sort(gridpoints), ...)
  attr(k, "class") = "nspmix"
  
  k
}

#' @rdname computemixdist
#' @export
computemixdist.npnormcvmw = function(v, mu0, pi0, beta, order = -3, mix = NULL, gridpoints = NULL, ...){
  if (missing(mu0)) {mu0 = 0}
  if (missing(pi0)) {pi0 = 0}
  if (missing(beta)) {beta = 1}
  v1 = bin(v, order)
  v2 = npnorm(v = v1$v, w = v1$w)
  if (is.null(gridpoints)) {gridpoints = c(gridpoints.npnorm(v2, beta = beta), v2$v[1] - 3 * beta, v2$v[length(v2$v)] + 3 * beta)}
  init = initial.npnorm(v2, beta = beta, mix = mix)
  k = npnormcvmw_(v1$v, v1$w, mu0, pi0, beta, 10^order, init$mix$pt, init$mix$pr, sort(gridpoints), ...)
  attr(k, "class") = "nspmix"
  
  k
}

#' @rdname estpi0
#' @export
estpi0.npnormcvmw = function(v, beta, val, order = -3, mix = NULL, gridpoints = NULL, ...){
  if (missing(beta)) {beta = 1}
  v1 = bin(v, order)
  v2 = npnorm(v = v1$v, w = v1$w)
  if (is.null(gridpoints)) {gridpoints = c(gridpoints.npnorm(v2, beta = beta), v2$v[1] - 3 * beta, v2$v[length(v2$v)] + 3 * beta)}
  init = initial.npnorm(v2, beta = beta, mix = mix)
  k = estpi0npnormcvmw_(v1$v, v1$w, beta, 10^order, val, init$mix$pt, init$mix$pr, sort(gridpoints), ...)
  attr(k, "class") = "nspmix"
  
  k
}

#' @rdname computemixdist
#' @export
computemixdist.npnormadw = function(v, mu0, pi0, beta, order = -3, mix = NULL, gridpoints = NULL, ...){
  if (missing(mu0)) {mu0 = 0}
  if (missing(pi0)) {pi0 = 0}
  if (missing(beta)) {beta = 1}
  v1 = bin(v, order)
  v2 = npnorm(v = v1$v, w = v1$w)
  if (is.null(gridpoints)) {gridpoints = c(gridpoints.npnorm(v2, beta = beta), v2$v[1] - 3 * beta, v2$v[length(v2$v)] + 3 * beta)}
  init = initial.npnorm(v2, beta = beta, mix = mix)
  k = npnormadw_(v1$v, v1$w, mu0, pi0, beta, 10^order, init$mix$pt, init$mix$pr, sort(gridpoints), ...)
  attr(k, "class") = "nspmix"
  
  k
}

#' @rdname estpi0
#' @export
estpi0.npnormadw = function(v, beta, val, order = -3, mix = NULL, gridpoints = NULL, ...){
  if (missing(beta)) {beta = 1}
  v1 = bin(v, order)
  v2 = npnorm(v = v1$v, w = v1$w)
  if (is.null(gridpoints)) {gridpoints = c(gridpoints.npnorm(v2, beta = beta), v2$v[1] - 3 * beta, v2$v[length(v2$v)] + 3 * beta)}
  init = initial.npnorm(v2, beta = beta, mix = mix)
  k = estpi0npnormadw_(v1$v, v1$w, beta, 10^order, val, init$mix$pt, init$mix$pr, sort(gridpoints), ...)
  attr(k, "class") = "nspmix"
  
  k
}

#' @rdname computemixdist
#' @export
computemixdist.nppoisll = function(v, mu0, pi0, beta, order, mix = NULL, gridpoints = NULL, ...){
  if (missing(mu0)) {mu0 = 0}
  if (missing(pi0)) {pi0 = 0}
  if (missing(beta)) {beta = 1}
  v1 = table(v)
  v2 = nppois(v = as.numeric(names(v1)), w = as.numeric(v1))
  if (is.null(gridpoints)) {gridpoints = gridpoints.nppois(v2, beta = beta)}
  init = initial.nppois(v2, beta = beta, mix = mix)
  k = nppoisll_(v2$v, v2$w, mu0, pi0, beta, init$mix$pt, init$mix$pr, sort(gridpoints), ...)
  attr(k, "class") = "nspmix"
  
  k
}

#' @rdname estpi0
#' @export
estpi0.nppoisll = function(v, beta, val, order, mix = NULL, gridpoints = NULL, ...){
  if (missing(beta)) {beta = 1}
  v1 = table(v)
  v2 = nppois(v = as.numeric(names(v1)), w = as.numeric(v1))
  if (is.null(gridpoints)) {gridpoints = gridpoints.nppois(v2, beta = beta)}
  init = initial.nppois(v2, beta = beta, mix = mix)
  k = estpi0nppoisll_(v2$v, v2$w, beta, val, init$mix$pt, init$mix$pr, sort(gridpoints), ...)
  attr(k, "class") = "nspmix"
  
  k
}

#' @rdname computemixdist
#' @export
computemixdist.nptllw = function(v, mu0, pi0, beta, order = -3, mix = NULL, gridpoints = NULL, ...){
  if (missing(mu0)) {mu0 = 0}
  if (missing(pi0)) {pi0 = 0}
  if (missing(beta)) {beta = Inf}
  v1 = bin(v, order)
  v2 = npnorm(v = qnorm(pt(v1$v, df = beta)), w = v1$w)
  if (is.null(gridpoints)) {gridpoints = qt(pnorm(gridpoints.npnorm(v2, beta = 1)), df = beta)}
  if (is.null(mix)){
    init = initial.npnorm(v2, beta = 1)
  }else{
    init = initial.npnorm(v2, beta = 1,
                          mix = list(pt = qnorm(pt(mix$pt, df = beta)), pr = mix$pr))   
  }
  k = nptllw_(v1$v, v1$w, mu0, pi0, beta, 10^order, qt(pnorm(init$mix$pt), df = beta), init$mix$pr, 
             sort(gridpoints), ...)
  attr(k, "class") = "nspmix"
  
  k
}

#' @rdname estpi0
#' @export
estpi0.nptllw = function(v, beta, val, order = -3, mix = NULL, gridpoints = NULL, ...){
  if (missing(beta)) {beta = Inf}
  v1 = bin(v, order)
  v2 = npnorm(v = qnorm(pt(v1$v, df = beta)), w = v1$w)
  if (is.null(gridpoints)) {gridpoints = qt(pnorm(gridpoints.npnorm(v2, beta = 1)), df = beta)}
  if (is.null(mix)){
    init = initial.npnorm(v2, beta = 1)
  }else{
    init = initial.npnorm(v2, beta = 1,
                          mix = list(pt = qnorm(pt(mix$pt, df = beta)), pr = mix$pr))   
  }
  k = estpi0nptllw_(v1$v, v1$w, beta, 10^order, val, qt(pnorm(init$mix$pt), df = beta), init$mix$pr, 
                    sort(gridpoints), ...)
  attr(k, "class") = "nspmix"
  
  k
}

#' @rdname computemixdist
#' @export
computemixdist.npnorm2Dll = function(v, mu0, pi0, beta, order, mix = NULL, gridpoints = NULL, ...){
  if (missing(mu0)) {mu0 = matrix(0, 1, 2)}
  if (missing(pi0)) {pi0 = 0}
  if (missing(beta)) {beta = diag(2)}
  v1 = npnorm(v[, 1])
  v2 = npnorm(v[, 2])
  if (is.null(gridpoints)) {
    g1 = gridpoints.npnorm(v1, beta = beta[1, 1])
    g2 = gridpoints.npnorm(v2, beta = beta[2, 2])
    LLL = max(length(g1), length(g2))
    g1 = sort(rep(g1, length = LLL))
    g2 = sort(rep(g2, length = LLL))
    gridpoints = cbind(g1, g2)
  }
  # think about what to do with mix;
  init1 = initial.npnorm(v1, beta = beta[1, 1], mix = list(pt = mix$pt[, 1], pr = mix$pr))
  init2 = initial.npnorm(v2, beta = beta[2, 2], mix = list(pt = mix$pt[, 2], pr = mix$pr))
  k = npnorm2Dll_(v, mu0, pi0, beta, cbind(rep(init1$mix$pt, length(init2$mix$pt)),
                                         rep(init2$mix$pt, each = length(init1$mix$pt))), 
                as.numeric(outer(init1$mix$pr, init2$mix$pr)), gridpoints, ...)
  attr(k, "class") = "nspmix"
  
  k
}