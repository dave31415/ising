import unittest
from unittest import TestCase
from ising import marginals_1d, marginals_1d_mp_last, \
    asymptotic_average, average_sigma_edge


class TestMessagePassing(TestCase):
    def test_on_edge(self):
        tol = 1e-12
        test_params = [(9, .12, 1.084),
                       (6, .07, 1.27),
                       (5, .4, .5),
                       (1, 0.02, 5.0)]

        for params in test_params:
            n, b, beta = params
            by_brute_force = marginals_1d(n, b=b, beta=beta, edge=True)
            by_message_passing = marginals_1d_mp_last(n, b=b, beta=beta)
            diff_pos = abs(by_message_passing[1] - by_brute_force[1])
            diff_neg = abs(by_message_passing[-1] - by_brute_force[-1])
            self.assertLess(diff_pos, tol)
            self.assertLess(diff_neg, tol)

    def test_asymptotic_average_is_approx_correct(self):
        tol = 1
        index_max = 1000
        n = 5000
        test_params = [(n, 0.0, 1.0),
                       (n, 0.0, 1.1),
                       (n, 0.02, 1.0),
                       (n, .12, 1.184),
                       (n, .333, 1.184)]

        for params in test_params:
            n, b, beta = params
            edge = True
            asymptotic_avg = asymptotic_average(b=b, beta=beta, edge=edge,
                                                index_max=index_max)
            avg = average_sigma_edge(n, b=b, beta=beta)
            diff = abs(asymptotic_avg-avg)
            print asymptotic_avg, avg
            print "n=%s, b=%s, beta=%s" % params
            self.assertLess(diff, tol)

        assert False


if __name__ == "__main__":
    unittest.main()