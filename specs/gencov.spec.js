var orthogonal = require('./../index.js').orthogonal,
    normal = require('./../index.js').normal,
    genS = require('./../index.js').genS,
    genArray = require('./../index.js').genArray,
    mvnrnd = require('./../index.js').mvnrnd,
    ndarray = require('ndarray'),
    scal = require('ndarray-blas-level1').scal,
    dot = require('ndarray-blas-level1').dot,
    axpy = require('ndarray-blas-level1').axpy,
    dger = require('ndarray-blas-dger'),
    gemv = require('ndarray-blas-gemv'),
    cholesky = require('ndarray-cholesky-factorization'),
    pack = require('ndarray-pack'),
    unpack = require('ndarray-unpack'),
    show = require('ndarray-show'),
    assert = require('better-assert');
assert.deepEqual = require('chai').assert.deepEqual;

var EPS = 1e-8;

function epsEqual(x, y, eps) {
    return Math.abs(x - y) < eps;
}

function nddiff(x, y) {
    var resid = 0;
    for (var i = 0; i < x.data.length; i++) {
        resid += Math.abs(x.data[i] - y.data[i]);
    }
    return resid;
}

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

function eye(d) {
    // d x d ndarray identity matrix
    var I = fill([d, d], 0);
    for (i = 0; i < d; i++) {
        I.set(i, i, 1);
    }
    return I;
}
    
describe('orthogonal', function(){
    it('makes an orthogonal matrix', function(){
        var d = 5;
        var Q = orthogonal(d);
        // I = Q Q~
        var QQ = fill([d, d], 0);
        for (i = 0; i < d; i++) {
            dger(1.0, Q.pick(i, null), Q.pick(i, null), QQ); 
        }
        assert(nddiff(eye(d), QQ) < 1e-8);
    });
});

describe('normal', function(){
    it('produces N(0, 1) distributed values, ndarray', function() {
        var N = 10000;
        var X = fill([N], function() { return normal(0, 1); } );
        var ones = fill([N], 1);
        var mean = dot(X, ones) / N;
        var s2 = dot(X, X) / N;
        assert(epsEqual(mean, 0, .05));
        assert(epsEqual(s2,   1, .05));
    });
    it('produces N(0, 1) distributed values, unpacked', function() {
        var N = 10000;
        var X = unpack(fill([N], function() { return normal(0, 1); } ));
        var mean = X.reduce(function(a,b) { return a + b; }) / N;
        var s2 = X.reduce(function(s2,x) { return s2 + x * x / N; }, 0);
        assert(epsEqual(mean, 0, .05));
        assert(epsEqual(s2,   1, .05));
    });
});

describe('genS', function(){
    it('makes an identity matrix given equal variances', function(){
        var d = 5;
        var S = genS([1,1,1,1,1]);
        assert(nddiff(S, eye(d)) < 1e-8);
    });
    it('makes a symmetric matrix', function(){
        var d = 5;
        var S = genS(d);
        var St = S.transpose(1,0);
        assert(nddiff(S, St) < 1e-8);
    });
    it('S from positive variances has a cholesky decompostion', function(){
        var S = genS([1,2,3,4,5]);
        var L = fill([5, 5], 0);
        cholesky(S, L);
        assert(!isNaN(nddiff(L, fill([5, 5], 0))));
    });
    it('S with negative variance cant be factored', function(){
        var S = genS([-1,2,3,4,5]);
        var L = fill([5, 5], 0);
        cholesky(S, L);
        assert(isNaN(nddiff(L, fill([5, 5], 0))));
    });
    it('genArray is an array version of genS', function() {
        var d = 5;
        var S = genArray([1,1,1,1,1]);
        assert(nddiff(pack(S), eye(d)) < 1e-8);
    });
});

describe('mvnrnd', function(){
    it('draws samples from a 3D N(0,I) distribution', function(){
        var N = 10000;
        var d = 3;
        var S = genS(unpack(fill([d], 1)));
        var xfunc = mvnrnd(0, S);
        var X = [];
        for (var i = 0; i < N; i++) {
            X.push(xfunc());
        }
        X = pack(X);
        // test the mean
        var M = fill([d], 0), ones = fill([N], 1);
        gemv(1/N, X, ones, 1, M);
        assert(nddiff(fill([d], 0), M)/d < .05);
        // test S
        var S = fill([d, d], 0);
        for (i = 0; i < N; i++) {
            dger(1/N, X.pick(i,null), X.pick(i,null), S); 
        }
        assert(nddiff(eye(d), S)/d < .05);
    });
});
