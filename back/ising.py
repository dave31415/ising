import numpy as np
import itertools
import math


def ising_energy(sigmas, b=0.0):
    """
    :param sigmas: list of +/- 1 values
    :param b: self energy factor
    :return:
    """
    n = len(sigmas)
    energy = 0.0
    for i in xrange(n-1):
        energy -= sigmas[i]*sigmas[i+1]

    energy -= b * sum(sigmas)
    return energy


def soft_exp(x):
    """
    :param x: a number
    :return: a softed version of exp that won't overflow
             and behaves reasonable for very large or very
             small numbers
    """
    if x < -100:
        return 0.0
    if x > 100:
        # basically cap it but let it increase slightly
        # just to aid in maximization
        return np.exp(100+np.log(x))
    return np.exp(x)


def prob_unnormalized(sigmas, b=0.0, beta=1.0):
    """
    :param sigmas: list of +/- 1 values
    :param b: self energy factor
    :param beta: pairwise energy factor
    :return:
    """
    energy = ising_energy(sigmas, b=b)
    return soft_exp(-beta*energy)


def all_binary_possibilties(n):
    """
    :param n: a dimension number
    :return: iterable with all binary possibilities
            example list(all_binary_possibilties(2))
            returns
            [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    """
    return itertools.product([1, -1], repeat=n)


def partition_function(sigmas_list, b=0.0, beta=1.0):
    """
    Calculate the partition function for the ising model
    :param n:
    :return:
    """
    norm = 0.0
    for sigmas in sigmas_list:
        prob = prob_unnormalized(sigmas, b=b, beta=beta)
        norm += prob
    return norm


def probability_distribution(n, b=0.0, beta=1.0):
    """
    Calculate the normalized probability distribution for all
    ising model possibilities
    :param n: dimension
    :param b: self energy factor
    :param beta: pairwise energy factor
    :return: list of 2-tuples with levels and probabilities
    Approximate performance, 1 second for n=15, 10 seconds for n=20
    time should scale like 2^n
    """
    assert n >= 1
    sigmas_list = list(all_binary_possibilties(n))
    norm = partition_function(sigmas_list, b=b, beta=beta)
    return [(sigmas, prob_unnormalized(sigmas, b=b, beta=beta)/norm)
            for sigmas in sigmas_list]


def marginals_1d(n, b=0.0, beta=1.0, edge=False):
    """
    Calculate the 1-d marginalized, normalized probability distribution
    for the ising model
    :param n: dimension
    :param b: self energy factor
    :param beta: pairwise energy factor
    :return: dictionary of marginal distribution
    """
    assert n >= 1

    probs = probability_distribution(n, b=b, beta=beta)
    index = 1
    if edge:
        index = 0

    #handle the n=1 edge case, where there is only an edge
    index = min(index, (n-1))

    marg = {-1: 0.0, 1: 0.0}

    for level, prob in probs:
        marg[level[index]] += prob
    return marg


def message_left(index_max, b=0.0, beta=1.0):
    message_first = {-1: 0.5, 1: 0.5}
    message_list = [message_first]
    for i in xrange(index_max):
        message_last = message_list[-1]
        message = {-1: 0.0, 1: 0.0}
        for sigma in [-1, 1]:
            for sigma_prime in [-1, 1]:
                x = beta * (sigma*sigma_prime + b*sigma_prime)
                factor = soft_exp(x)
                message[sigma] += message_last[sigma_prime] * factor
        #now normalize
        norm = message[-1]+message[1]
        for sigma in [-1, 1]:
            message[sigma] /= norm

        message_list.append(message)

    return message_list


def marginals_1d_mp_last(n, b=0.0, beta=1.0):
    last_message_left = message_left(n-1, b=b, beta=beta)[-1]
    for sigma in [-1, 1]:
        last_message_left[sigma] *= soft_exp(b*beta*sigma)
    norm = last_message_left[-1]+last_message_left[1]
    for sigma in [-1, 1]:
        last_message_left[sigma] /= norm
    return last_message_left


def function_u(x, beta=1.0):
    return np.log(x[1]/x[-1])/(2.0*beta)


def u_left(index_max, b=0.0, beta=1.0):
    messages = message_left(index_max, b=b, beta=beta)
    return [function_u(message) for message in messages]


def u_star(b=0.0, beta=1.0, index_max=100):
    return u_left(index_max, b=b, beta=beta)[-1]


def f_function_unused(x):
    y = math.tanh(beta) * math.tanh(beta*x)
    return math.atanh(y)/beta


def average_sigma_edge(n, b=0.0, beta=1.0):
    marginals = marginals_1d_mp_last(n, b=b, beta=beta)
    return marginals[1]-marginals[-1]


def asymptotic_average(b=0.0, beta=1.0, edge=False, index_max=100):
    #check that this works
    ustar = u_star(b=b, beta=beta, index_max=index_max)
    factor = 2.0
    if edge:
        factor = 1.0
    return math.tanh(beta*(factor*ustar+b))

