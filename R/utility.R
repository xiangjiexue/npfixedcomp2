#' Find the posterior mean given the observations and the mixing
#' distribution based on the family in the result
#'
#' @title Find the posterior mean
#' @param x a vector of observations
#' @param result an object of class nspmix
#' @param fun the function to transform the mean. It finds the posterior mean
#' of \code{fun(x)}. The function \code{fun} must be vectorised.
#' @examples
#' data = rnorm(500, c(0, 2))
#' r1 = computemixdist(data, pi0 = 0.5)
#' a1 = posteriormean(data, r1)
#' r2 = computemixdist(data, pi0 = 0.5, method = "nptll") # equivalent to normal
#' a2 = posteriormean(data, r2, fun = function(x) x^2)
#' data = runif(500, min = -0.5, max = 0.5)
#' r3 = computemixdist(data, method = "npnormcll", beta = 100)
#' a3 = posteriormean(data, r3)
#' data = rpois(500, c(0, 2))
#' r4 = computemixdist(data, method = "nppoisll")
#' a4 = posteriormean(data, r4)
#' @export
posteriormean = function(x, result, fun = function(x) x){
  f = match.fun(paste0("posteriormean.", result$family))
  f(x = x, result = result, fun = fun)
}

#' @rdname posteriormean
#' @export
posteriormean.npnorm = function(x, result, fun = function(x) x){
  temp = dnormarray_(x, result$mix$pt, result$beta) *
    rep(result$mix$pr, rep(length(x), length(result$mix$pr)))
  .rowSums(temp * rep(fun(result$mix$pt), rep(length(x), length(result$mix$pt))),
           m = length(x), n = length(result$mix$pt)) /
    .rowSums(temp, m = length(x), n = length(result$mix$pt))
}

#' @rdname posteriormean
#' @export
posteriormean.npt = function(x, result, fun = function(x) x){
  temp = dtarray_(x, result$mix$pt, result$beta) *
    rep(result$mix$pr, rep(length(x), length(result$mix$pr)))
  .rowSums(temp * rep(fun(result$mix$pt), rep(length(x), length(result$mix$pt))),
           m = length(x), n = length(result$mix$pt)) /
    .rowSums(temp, m = length(x), n = length(result$mix$pt))
}

#' @rdname posteriormean
#' @export
posteriormean.npnormc = function(x, result, fun = function(x) x){
  temp = dnormcarray_(x, result$mix$pt, result$beta) *
    rep(result$mix$pr, rep(length(x), length(result$mix$pr)))
  .rowSums(temp * rep(fun(result$mix$pt), rep(length(x), length(result$mix$pt))),
           m = length(x), n = length(result$mix$pt)) /
    .rowSums(temp, m = length(x), n = length(result$mix$pt))
}

#' @rdname posteriormean
#' @export
posteriormean.nppois = function(x, result, fun = function(x) x){
  temp = dpoisarray_(x, result$mix$pt, result$beta) *
    rep(result$mix$pr, rep(length(x), length(result$mix$pr)))
  .rowSums(temp * rep(fun(result$mix$pt), rep(length(x), length(result$mix$pt))),
           m = length(x), n = length(result$mix$pt)) /
    .rowSums(temp, m = length(x), n = length(result$mix$pt))
}

#' Extract or returning the lower triangular part of the matrix
#'
#' The function \code{extractlower} is to extract the strict
#' lower triangular part of a squared matrix and the function
#' \code{returnlower} is to return the vector value into a
#' symmetric matrix with diagonal 1.
#'
#' @param A a matrix to be extracted the lower triangular part
#' @param v a vector to be returned to a symmetric matrix with diagonal 1.
#' @examples
#' a = matrix(1:100, 10, 10)
#' b = extractlower(a)
#' d = returnlower(b)
#' @rdname extractlower
#' @export
extractlower = function(A){
  A[lower.tri(A), drop = TRUE]
}

#' @rdname extractlower
#' @export
returnlower = function(v){
  LLL = (1 + sqrt(1 + 8 * length(v))) / 2
  ans = matrix(0, LLL, LLL)
  ans[lower.tri(ans)] = v
  ans + t(ans) + diag(LLL)
}

#' Estimating covariance matrix using Empirical Bayes
#'
#' The function \code{covestEB} performs covariance matrix estimation using
#' Fisher transformation, while the function \code{covestEB.cor} performs
#' covariance estimation directly on sample correlation coefficients using
#' one-parameter normal approximation.
#'
#' Covariance matrix estimation using Fisher transformation supports estimation
#' sparsity as well as large-scale computation, while estimation on the original
#' scale supports neither and it is for comparison only. It is recommended to
#' perform estimation on Fisher-transformed sample correlation coefficients.
#'
#' @title Estimating Covariance Matrix using Empirical Bayes
#' @param X a matrix of size n * p, where n is the number of observations and
#' p is the number of variables
#' @param estpi0 logical; if TRUE, the NPMLE is estimated based on the
#' estimation of pi0, which in this case can be used to detect sparsity or
#' assume sparsity.
#' @param order the level of binning to use when the number of observations
#' passed to the computation is greater than 5000.
#' @param verbose logical; If TRUE, the intermediate results will be shown.
#' @param force.nonbin logical; If TRUE, overwrites the default setting for
#' large-scale computation.
#' @return a list. a covariance matrix estimate of size p * p is given in mat,
#' whether correction is done is given in correction, and the method for
#' computing the density of sample correlation coefficients is given in method.
#' @rdname covestEB
#' @examples
#' n = 100; p = 50
#' X = matrix(rnorm(n * p), nrow = n, ncol = p)
#' system.time(ans <- covestEB(X))
#' system.time(ans1 <- covestEB.cor(X))
#' @export
covestEB = function(X, estpi0 = FALSE, order = -3, verbose = FALSE, force.nonbin = FALSE){
  p = dim(X)[2]
  n = dim(X)[1]
  covest = cov(X)
  index = diag(covest) > .Machine$double.eps
  fisherdata = atanh(extractlower(cov2cor(covest[index, index])))
  if (estpi0){
    if (length(fisherdata) > 5000& !force.nonbin){
      r = estpi0(fisherdata, method = "npnormllw", order = order, beta = sqrt(1 / (n - 3)), verbose = verbose, val = 1)
    }else{
      r = estpi0(fisherdata, beta = sqrt(1 / (n - 3)), verbose = verbose, val = 1)
    }
  }else{
    if (length(fisherdata) > 5000 & !force.nonbin){
      r = computemixdist(fisherdata, method = "npnormllw", order = order, beta = sqrt(1 / (n - 3)), verbose = verbose)
    }else{
      r = computemixdist(fisherdata, beta = sqrt(1 / (n - 3)), verbose = verbose)
    }
  }
  
  postmean = posteriormean(fisherdata, r, fun = tanh)
  
  ans = diag(nrow = p, ncol = p)
  
  ans[index, index] = returnlower(postmean)
  
  ans1 = correlationmatrixcpp(ans, tol = 1e-3)
  
  varest = sqrt(diag(covest))
  
  list(mat = ans1 * varest * rep(varest, rep(length(varest), length(varest))),
       correction.Fnorm = norm(ans1 - ans, type = "F"),
       mix.dist = r)
}

#' @rdname covestEB
#' @export
covestEB.cor = function(X, verbose = FALSE){
  p = dim(X)[2]
  n = dim(X)[1]
  covest = cov(X)
  index = diag(covest) > .Machine$double.eps
  data = extractlower(cov2cor(covest[index, index]))
  r = computemixdist(data, beta = n, method = "npnormcll", verbose = verbose)
  
  postmean = posteriormean(data, r)
  
  ans = diag(nrow = p, ncol = p)
  
  ans[index, index] = returnlower(postmean)
  
  ans1 = correlationmatrixcpp(ans, tol = 1e-3)
  
  varest = sqrt(diag(covest))
  
  list(mat = ans1 * varest * rep(varest, rep(length(varest), length(varest))),
       correction.Fnorm = norm(ans1 - ans, type = "F"),
       mix.dist = r)
}

#' plot the mapping between the original observations and its posterior mean
#'
#' This function explicity considers that map between the transformed sample
#' correlation coefficients to the posterior mean of tanh(x) under normal cases.
#' \code{result} typically is the mixing distribution of the transformed sample
#' correlations and \code{result2} is the mixing distribution on the sample
#' correlation scale.
#'
#' @title plot the posterier map
#' @param x a vector of observations
#' @param result an object of class nspmix to show in the top
#' @param result2 an object of class nspmix to show in the bottom. If NULL,
#' then the density in the bottom is not drawn.
#' @param ... other parameter passed to \code{plot}
#' @return none
#' @examples
#' n = 100; p = 50
#' X = matrix(rnorm(n * p), nrow = n, ncol = p)
#' r = cor(X)
#' r1 = computemixdist(atanh(extractlower(r)), beta = 1 / sqrt(n - 3))
#' plotposteriormapping(atanh(extractlower(r)), r1)
#' @export
plotposteriormapping = function(x, result, result2 = NULL, ...){
  values = 0;
  plot(values, 0, xlim = range(x), ylim = c(0, 1), type = "n", ylab = "", yaxt = "n", ...)
  ordered.x = sort(x)
  postmean = posteriormean(ordered.x, result, tanh)
  cols = rainbow(length(ordered.x), end = 0.85)
  sapply(1:length(ordered.x), function(ddd){
    points(tanh(ordered.x[ddd]), 1, col = cols[ddd], pch = 16)
    points(ordered.x[ddd], 0.5, col = cols[ddd], pch = 16)
    points(postmean[ddd], 0, col = cols[ddd], pch = 16)
    lines(c(ordered.x[ddd], tanh(ordered.x[ddd])), c(0.5, 1), col = cols[ddd])
    lines(c(ordered.x[ddd], postmean[ddd]), c(0.5, 0), col = cols[ddd])
  })
  points(result$mix$pt, rep(0.5, length(result$mix$pt)))
  sapply(1:length(result$mix$pr), function(ddd){
    lines(rep(result$mix$pt[ddd], 2), 0.5 + c(0, result$mix$pr[ddd] * 0.18), lwd = 3)
  })
  d = seq(min(x), to = max(x), length = 1000)
  dens = dnpnorm(d, mu0 = result$mix$pt, pi0 = result$mix$pr, stdev = result$beta)
  lines(d, dens / max(dens) * 0.2 + 0.5, lwd = 2)
  abline(h = c(0, 0.5, 1), lwd = 0.5)
  if (!is.null(result2)){
    points(result2$mix$pt, rep(0, length(result2$mix$pt)))
    sapply(1:length(result2$mix$pr), function(ddd){
      lines(rep(result2$mix$pt[ddd], 2), c(0, result2$mix$pr[ddd] * 0.18), lwd = 3)
    })
    d2 = pmin(pmax(-0.95, seq(min(x), to = max(x), length = 1000)), 0.95)
    dens2 = dnpnormc(d2, mu0 = result2$mix$pt, pi0 = result2$mix$pr, n = ceiling((1 / result$beta)^2 + 3))
    lines(d2, dens2 / max(dens2) * 0.2, lwd = 2)
  }
  
  NULL
}

#' Implementation of several FDR-controlling procedure in R.
#'
#' The function \code{adaptive.stepdown} is a simple implementation of the
#' adaptive step-down procedure described in Gavrilov et al. (2009)
#'
#' The function \code{BH} is a direct implementation of the procedure
#' described in Benjamini and Hochberg (1995).
#'
#' @title FDR controlling procedures
#' @param pval a vector of p-values (no necessarily sorted)
#' @param alpha given FDR level
#' @return a list with num.rejection, the number of rejections computed by this function,
#' and classifer, a vector of TRUE and FALSE; if TRUE, the corresponding input is regarded
#' as null, and as non-null if otherwise.
#' @examples
#' adaptive.stepdown(pnorm(-abs(rnorm(1000, c(0, 2)))) * 2)
#' BH(pnorm(-abs(rnorm(1000, c(0, 2)))) * 2)
#' @name FDRcontrol
NULL

#' @rdname FDRcontrol
#' @export
adaptive.stepdown = function(pval, alpha = 0.05){
  pval.sorted = sort(pval)
  LLL = length(pval)
  # alpha / (1 - alpha) = i*q/(m + 1 - i)
  # alpha =  i * q / (m + 1 - i) * (1 - alpha)
  # alpha =  (i * q / (m + 1 - i)) / (1 + i * q / (m + 1 - i))
  const = 1:LLL * alpha / LLL:1
  a = const / (1 + const)
  k = match(FALSE, pval.sorted <= a)
  list(num.rejection = k - 1,
       classifier = pval >= pval.sorted[k])
}

#' @rdname FDRcontrol
#' @export
BH = function(pval, alpha = 0.05){
  pval.sorted = sort(pval, decreasing = TRUE)
  LLL = length(pval)
  a = LLL:1 * alpha / LLL
  k = match(TRUE, pval.sorted <= a)
  list(num.rejection = ifelse(is.na(k), 0, LLL + 1 - k),
       classifier = if (is.na(k)) rep(TRUE, LLL) else pval > pval.sorted[k])
}