## Since previous version

Density of multi-dimensional normal mixtures (dnpnormND, dnormNDarray_).

Imported L-BFGS-B from https://github.com/yixuan/LBFGSpp/ and made some
modifications tailored to finding new support points for npnorm2Dll

experimental implementation of npnorm2Dll.

## npfixedcomp2 1.1.0002

Improvement on outer minus in some expressions.

## npfixedcomp2 1.1.0001

improvement on handling results. Added class method get_ans() to process the results.

## npfixedcomp2 1.1.0000

Full ET for all density functions.

Change the implementation for one-parameter normal.

Change to R's C implementation of log1pexp and log1mexp as available since R 4.1.0.

The header file for computing the NPMLE/NPMDE as well as the implemented families
are now written as class templates. It is then easier to change the precision.

## npfixedcomp2 1.0.0010

Added NEWS.md.

Fixed some descriptions in function references.

Improvement to the binning function *bin()* for large-scale data. 