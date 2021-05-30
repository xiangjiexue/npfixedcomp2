---
title: "Non-parametric estimation of mixing distribution and the proportion of zeros"
author: "Xiangjie Xue"
date: "31 May, 2021"
output: 
  html_document: 
    keep_md: yes
---

[![Build Status](https://travis-ci.com/xiangjiexue/npfixedcomp2.svg?branch=main)](https://travis-ci.com/xiangjiexue/npfixedcomp2)

This is a R package for estimating the mixing distribution and the proportion of null effects accompanying the [doctoral thesis](https://hdl.handle.net/2292/54659) by Xiangjie Xue. The main webpage of this package is [here](https://xiangjiexue.github.io/npfixedcomp2).

## Installation

The package can be installed via *devtools* or *remotes*:


```r
devtools::install_github("xiangjiexue/npfixedcomp2")
```

NOTE: This package contains Fortran and C++ codes, make sure gfortran and gcc/clang is available.

## Available family functions

Since this package is written in C++, there are limited number of family functions available. 

| Family | loss | original method | large-scale implementation |
|----|----|----|----|
| normal | maximum likelihood | npnormll | npnormllw |
| one-parameter normal | maximum likelihood | npnormcll | N/A (requires variational bin) |
| (non-central) t | maximum likelihood | nptll | nptllw |
| poisson | maximum likelihood | nppois | N/A (binning by default) |
| normal | Cramer-von Mises | npnormcvm | npnormcvmw |
| normal | Anderson-Darling | npnormad | npnormadw |
