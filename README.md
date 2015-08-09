gencov
=======
[![Build Status][travis-image]][travis-url] [![NPM version][npm-image]][npm-url] [![NPM download][download-image]][npm-url]

Generate random covariance matrices, and draw MVN samples using them.

### Covariance matrix:

The `genS` and `genArray` functions produce random covariance matrices
(as ndarray or javascript array) with a specified variance structure.
The eigenvalues (principal component variances) V for the covariance
matrix may be specified, or may be randomly generated from within a
specified range. A random orthogonal matrix Q is generated and its
columns used as eigenvectors. The covariance matrix is then generated
as S = Q V Q~

### Sampling:

Given a covariance ndarray S, you can generate samples from the
associated multivariate normal distribution using the `mvnrnd`
function (which creates a function that draws samples from N(mean, S))

Samples x ~ N(0, S) are drawn by first drawing z ~ N(0, I) then
transforming x = L z, where S = L L~.

### Example usage:
```
var gencov = require('gencov');

// generate a 3-d correlation matrix with variances between 1 and 10,
// and return it as an ndarray:

var S = gencov.genS(3);

// generate a 5-d correlation matrix with principal components,
// return as a regular array

var S = gencov.genArray([3, 2, 1, 0.5, 0.1]);

// draw 10 3d samples from a N([a,b,c], S) distribution with random S,
// return as an array of 3-vectors.

var X = Array.apply(null, 10).map(mvnrnd([a,b,c], genS(3)))
```

dependencies
-------------
`ndarray`: [https://www.npmjs.com/package/ndarray](https://www.npmjs.com/package/ndarray)
`ndarray-blas-level1`: [https://www.npmjs.com/package/ndarray-blas-level1](https://www.npmjs.com/package/ndarray-blas-level1)
`ndarray-blas-level2`: [https://www.npmjs.com/package/ndarray-blas-level2](https://www.npmjs.com/package/ndarray-blas-level2)
`ndarray-blas-dger`: [https://www.npmjs.com/package/ndarray-blas-dger](https://www.npmjs.com/package/ndarray-blas-dger)
`ndarray-unpack`: [https://www.npmjs.com/package/ndarray-unpack](https://www.npmjs.com/package/ndarray-unpack)
`ndarray-gram-schmidt-qr`: [https://www.npmjs.com/package/ndarray-gram-schmidt-qr](https://www.npmjs.com/package/ndarray-gram-schmidt-qr)
`ndarray-cholesky-factorization`: [https://www.npmjs.com/package/ndarray-cholesky-factorization](https://www.npmjs.com/package/ndarray-cholesky-factorization)

[travis-image]: https://travis-ci.org/welch/tdigest.svg?branch=master
[travis-url]: https://travis-ci.org/welch/gencov
[npm-image]: http://img.shields.io/npm/v/gencov.svg
[download-image]: http://img.shields.io/npm/dm/gencov.svg
[npm-url]: https://www.npmjs.org/package/gencov

