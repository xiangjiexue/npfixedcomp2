---
title: "Non-parametric estimation of mixing distribution and the proportion of zeros"
author: "Xiangjie Xue"
date: "06 三月, 2022"
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

NOTE: This package contains Fortran and C++ codes with C++11 standard, so make sure gfortran and gcc/clang is available.

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
| bivariate normal (experimental) | maximum likelihood | npnorm2Dll | N/A (not yet implemented) |

It is noted that after version 1.1, all the header files are implemented as template.
It is possible to write implementation using other scalar type rather than double
(float, long double, etc.) provided that it is supported by Eigen, 
but please make sure that the precision is what you want.


```cpp
template<class Type>
class npfixedcomp
{
public:
  ...
};

template<class Type>
class npnormll : public npfixedcomp<Type>
{
public:
  ...
};
```
