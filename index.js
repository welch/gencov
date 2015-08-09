//
// gencov: generate random covariance matrices, and MVN samples using them.
//
// Covariance matrix:
// The eigenvalues (principal component variances) V for the
// covariance matrix may be specified, or may be randomly generated 
// from within a specified range. A random orthogonal matrix Q is
// generated and its columns used as eigenvectors. The covariance
// matrix is then generated as S = Q V Q~
//
// Sampling:
// Samples x ~ N(0, S) are drawn by first drawing z ~ N(0, I) 
// then transforming x = L z, where S = L L~.
//
// Example usage:
// var gencov = require('gencov');
//
// generate a 3-d correlation matrix with variances between 1 and 10,
// and return it as an ndarray:
//
// var S = gencov.genS(3);
//
// generate a 5-d correlation matrix with principal components,
// return as an ndarray
//
// var S = gencov.genS([3, 2, 1, 0.5, 0.1]);
//
// draw 10 3d samples from a N([a,b,c], S) distribution with random S, return
// as an array of 3-vectors.
//
// var X = Array.apply(null, 10).map(mvnrnd([a,b,c], genS(3)))
//
var ndarray = require('ndarray'),
    scal = require('ndarray-blas-level1').scal,
    axpy = require('ndarray-blas-level1').axpy,
    trmv = require('ndarray-blas-level2').trmv,
    dger = require('ndarray-blas-dger'),
    qr = require('ndarray-gram-schmidt-qr'),
    cholesky = require('ndarray-cholesky-factorization'),
    unpack = require('ndarray-unpack');

function fill(shape, v) {
    // allocate an ndarray of the given shape, filled with v or calls to v() if a function
    //
    var sz = 1;
    for(var i = 0; i < shape.length; ++i) {
        sz *= shape[i];
    }
    var vfill = Array.apply(null, Array(sz)).map(
        (typeof v === 'function') ? v : function() {return v;}
    );
    return ndarray(new Float64Array(vfill), shape);
}

function uniform(min, max) {
    // return a uniform sample from (min...max)
    // 
    min = (min === undefined) ? 0 : min;
    max = (max === undefined) ? 1 : max;
    return Math.random() * (max - min) + min; 
}

var _extra = null;
function normal(mean, sigma) {
    // return a Box-Muller normal
    //
    mean = mean || 0;
    sigma = sigma || 1;
    if (_extra !== null) { 
        var result = mean + sigma * _extra; 
        _extra = null;
        return result;
    } else {
        var u = 2 * Math.random() - 1;
        var v = 2 * Math.random() - 1;
        var r = u*u + v*v;
        if (r === 0 || r > 1) {
            // out of bounds, try again
            return normal(mean, sigma);
        }
        var c = Math.sqrt(-2*Math.log(r)/r);
        _extra = u * c;
        return mean + sigma * v * c;
    }
}

function orthogonal(d) {
    // return a Haar-measure uniform random orthogonal matrix as a
    // (d x d) ndarray. For the theory, see
    // http://www.ams.org/notices/200511/what-is.pdf, and for the
    // trick of adjusting QR decomposition output, see
    // http://arxiv.org/pdf/math-ph/0609050v2.pdf.
    //
    var Z = fill([d, d], function() { return normal(0,1); });
    var R = fill([d, d], 0);
    qr( Z, R );
    // adjust Z = Z * sign(diag(R))
    for (i = 0; i < d; i++) {
        var sign = (R.get(i, i) >= 0) ? 1 : -1;
        var zi = Z.pick(i, null);
        scal(sign, zi);
    }
    return Z;
}

function genS(d, min, max) {
    // generate a random covariance matrix, returned as a d x d ndarray.
    // min and max specify the range from which to draw principal variances,
    // and default to (1..10).
    // If instead S is called with a single array parameter,
    // its dimension is d and its values are the principal variances.
    //
    var V;
    min = (min === undefined) ? 1 : min;
    max = (max === undefined) ? 10 : max;
    if (typeof d === 'number') {
        // uniform random variances in (min...max)
        V = unpack(fill([d], function() { return uniform(min, max); }));
    } else {
        V = d; // assume d is a list of variances
    }
    d = V.length;
    var Q = orthogonal(d);
    var S = fill([d, d], 0);
    // S = Q V Q~
    for (i = 0; i < d; i++) {
        dger(V[i], Q.pick(i, null), Q.pick(i, null), S); 
    }
    return S;
}

function mvnrnd(u, S) {
    // generate a function that returns x ~ N(u, S) where u is the
    // d-dimensional mean vector or 0, and S is a covariance ndarray
    // generated as in genS(len(u), min, max) above.
    // 
    var d = S.shape[0];
    var M = (u === 0) ? fill([d], 0) : ndarray(new Float64Array(u));
    var L = fill([d, d], 0);
    cholesky(S, L);
    return function() {
        var x = fill([d], function() { return normal(0, 1); });
        trmv(L, x, true);
        axpy(1, M, x);
        return unpack(x);
    };
}

module.exports = {
    normal: normal,
    orthogonal: orthogonal,
    genS: genS,
    genArray: function(d, min, max) { return unpack(genS(d, min, max)) },
    mvnrnd: mvnrnd
};
