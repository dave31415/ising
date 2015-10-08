import unittest
from unittest import TestCase
from ising import marginals_1d, marginals_1d_mp_last, ising_energy


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


class TestIsingEnergy(TestCase):
    def test_ising_energy(self):
        tol = 1e-12

        sigmas = [1, 1, 1]
        b = 1.5
        energy = ising_energy(sigmas, b=b)
        expected = -2 - 3*b
        diff = abs(energy-expected)
        self.assertLess(diff, tol)


if __name__ == "__main__":
    unittest.main()