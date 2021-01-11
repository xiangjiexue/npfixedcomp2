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