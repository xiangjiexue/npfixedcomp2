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
#' posteriormean(data, r1)
#' r2 = computemixdist(data, pi0 = 0.5, method = "nptll") # equivalent to normal
#' posteriormean(data, r2, fun = function(x) x^2)
#' data = runif(500, min = -0.5, max = 0.5)
#' r3 = computemixdist(data, method = "npnormcll", beta = 100)
#' posteriormean(data, r3)
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
#' @param order the level of binning to use when the number of observations
#' passed to the computation is greater than 5000.
#' @param verbose logical; If TRUE, the intermediate results will be shown.
#' @param force.nonbin logical; If TRUE, no binning is performce by force.
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
covestEB = function(X, order = -3, verbose = FALSE, force.nonbin = FALSE){
  p = dim(X)[2]
  n = dim(X)[1]
  covest = cov(X)
  index = diag(covest) > .Machine$double.eps
  fisherdata = atanh(extractlower(cov2cor(covest[index, index])))
  if (length(fisherdata) > 5000 & !force.nonbin){
    r = computemixdist(fisherdata, method = "npnormllw", order = order, beta = sqrt(1 / (n - 3)), verbose = verbose)
  }else{
    r = computemixdist(fisherdata, beta = sqrt(1 / (n - 3)), verbose = verbose)
  }
  
  postmean = posteriormean(fisherdata, r, fun = tanh)
  
  ans = diag(nrow = p, ncol = p)
  
  ans[index, index] = returnlower(postmean)
  
  ans1 = CorrelationMatrix(ans, b = rep(1, p), tol = 1e-3)
  ans2 = ans1$CorrMat
  
  varest = sqrt(diag(covest))
  
  list(mat = ans2 * varest * rep(varest, rep(length(varest), length(varest))),
       correction = ifelse(ans1$iterations == 0, FALSE, TRUE),
       correction.Fnorm = norm(ans2 - ans, type = "F"),
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
  
  ans1 = CorrelationMatrix(ans, b = rep(1, p), tol = 1e-3)
  ans2 = ans1$CorrMat
  
  varest = sqrt(diag(covest))
  
  list(mat = ans2 * varest * rep(varest, rep(length(varest), length(varest))),
       correction = ifelse(ans1$iterations == 0, FALSE, TRUE),
       correction.Fnorm = norm(ans2 - ans, type = "F"),
       mix.dist = r)
}