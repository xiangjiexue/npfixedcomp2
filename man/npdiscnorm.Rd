% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{dnpdiscnorm}
\alias{dnpdiscnorm}
\alias{pnpdiscnorm}
\title{non-parametric discrete normal distribution}
\usage{
dnpdiscnorm(x, mu0, pi0, stdev, h, lg = FALSE)

pnpdiscnorm(x, mu0, pi0, stdev, h, lt = TRUE, lg = FALSE)
}
\arguments{
\item{x}{vector of observations, vector of quantiles}

\item{mu0}{the vector of support points}

\item{pi0}{the vector of weights correponding to the support points}

\item{stdev}{standard deviation}

\item{h}{the bin width}

\item{lg}{logical; if TRUE, the result will be given in log scale.}

\item{lt}{logical; if TRUE, the result will be given in lower tail.}
}
\description{
The density and the distribution function of non-parametric discrete normal distribution
}
\details{
\code{dnpdiscnorm} gives the density and \code{pnpdiscnorm} gives the CDF.

In this implementation, the support space is ..., -h, 0, h, ...
}
