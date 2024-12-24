"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
# fixme: what is the float type in python ?
def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Identity function."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> float:
    """If x less than y return 1.0 else 0.0"""
    return 1.0 if x < y else 0.0


def is_close(x: float, y: float) -> float:
    """If |x - y| < 1e-2 return 1.0 else 0.0"""
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def eq(x: float, y: float) -> float:
    """If x equal y return 1.0 else 0.0"""
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def max(x: float, y: float) -> float:
    """Return the maximum of two numbers."""
    return x if x > y else y


def sigmoid(x: float) -> float:
    """Return the sigmoid of x."""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Return the relu of x."""
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Return the log of x."""
    return math.log(x)


def exp(x: float) -> float:
    """Return the exponential of x."""
    return math.exp(x)


def log_back(x: float, y: float) -> float:
    """Return the log back of x."""
    return y / x


def inv(x: float) -> float:
    """Return the inverse of x."""
    return 1.0 / x


def inv_back(x: float, y: float) -> float:
    """Return the inverse back of x."""
    return -1.0 / (x * x) * y


def relu_back(x: float, y: float) -> float:
    """Return the relu back of x."""
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Return a function that applies fn to each element of a list"""
    return lambda xs: [fn(x) for x in xs]


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Return a function that applies fn to each pair of elements from two lists"""
    return lambda xs, ys: [fn(x, y) for x, y in zip(xs, ys)]


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Return a function that reduces a list with fn starting from start"""

    def reduce_fn(xs: Iterable[float]) -> float:
        acc = start
        for x in xs:
            acc = fn(acc, x)
        return acc

    return reduce_fn


def negList(xs: Iterable[float]) -> Iterable[float]:
    """Negate a list."""
    return map(neg)(xs)


def addLists(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
    """Add two lists element-wise."""
    return zipWith(add)(xs, ys)


def sum(xs: Iterable[float]) -> float:
    """Sum a list."""
    return reduce(add, 0.0)(xs)


def prod(xs: Iterable[float]) -> float:
    """Product of a list."""
    return reduce(mul, 1.0)(xs)
