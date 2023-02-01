import unittest
import copy

import torch
import torch.nn
import torch.testing

from fedbox.utils.functional import *


class TestModelOperation(unittest.TestCase):
    def assert_close(self, actual, expected, **kwargs):
        torch.testing.assert_close(actual, expected, **kwargs)

    @staticmethod
    def make_model(w1: float, w2: float):
        class Model(torch.nn.Module):
            def __init__(self, w1: float, w2: float):
                super().__init__()
                self.w1 = torch.nn.Parameter(torch.ones(10, 20) * w1)
                self.w2 = torch.nn.Parameter(torch.ones(10, 20) * w2)

        return Model(w1, w2)

    def test_model_zip(self):
        m1 = self.make_model(0, 1)
        m2 = self.make_model(2, 3)
        result = list(model_zip(m1, m2))
        (m1w1, m2w1), (m1w2, m2w2) = result
        self.assertIs(m1w1, m1.w1)
        self.assertIs(m1w2, m1.w2)
        self.assertIs(m2w1, m2.w1)
        self.assertIs(m2w2, m2.w2)

    def test_model_zip_parameters(self):
        m1 = self.make_model(0, 1)
        m2 = self.make_model(2, 3)
        result = list(model_zip(m1.parameters(), m2.parameters()))
        (m1w1, m2w1), (m1w2, m2w2) = result
        self.assertIs(m1w1, m1.w1)
        self.assertIs(m1w2, m1.w2)
        self.assertIs(m2w1, m2.w1)
        self.assertIs(m2w2, m2.w2)

    def test_model_assign(self):
        m1 = self.make_model(0, 1)
        m2 = self.make_model(2, 3)
        model_assign(m1, m2)
        self.assertTrue((m1.w1 == 2).all())
        self.assertTrue((m1.w2 == 3).all())
        for p in m1.parameters():
            self.assertIsNone(p.grad)
        for p in m2.parameters():
            self.assertIsNone(p.grad)

    def test_model_assign_parameters(self):
        m1 = self.make_model(0, 1)
        m2 = self.make_model(2, 3)
        model_assign(m1.parameters(), m2.parameters())
        self.assertTrue((m1.w1 == 2).all())
        self.assertTrue((m1.w2 == 3).all())
        for p in m1.parameters():
            self.assertIsNone(p.grad)
        for p in m2.parameters():
            self.assertIsNone(p.grad)

    def test_model_assign_helper(self):
        m1 = self.make_model(0, 1)
        m2 = self.make_model(2, 3)
        assign[m1] = m2
        assign[m1.parameters()] = m2.parameters()
        self.assertTrue((m1.w1 == 2).all())
        self.assertTrue((m1.w2 == 3).all())
        for p in m1.parameters():
            self.assertIsNone(p.grad)
        for p in m2.parameters():
            self.assertIsNone(p.grad)

    def test_model_aggregate(self):
        m1 = self.make_model(0, 1)
        m2 = self.make_model(2, 3)
        dest = copy.deepcopy(m1)
        assign[dest] = model_aggregate(sum, [m1, m2])
        self.assert_close(dest.w1, m1.w1 + m2.w1)
        self.assert_close(dest.w2, m1.w2 + m2.w2)
        for p in m1.parameters():
            self.assertIsNone(p.grad)
        for p in m2.parameters():
            self.assertIsNone(p.grad)
        for p in dest.parameters():
            self.assertIsNone(p.grad)

    def test_model_aggregate_unpack(self):
        m1 = self.make_model(0, 1)
        m2 = self.make_model(2, 3)
        dest = copy.deepcopy(m1)
        assign[dest] = model_aggregate(lambda p1, p2: p1 + p2, m1, m2)
        self.assert_close(dest.w1, m1.w1 + m2.w1)
        self.assert_close(dest.w2, m1.w2 + m2.w2)
        for p in m1.parameters():
            self.assertIsNone(p.grad)
        for p in m2.parameters():
            self.assertIsNone(p.grad)
        for p in dest.parameters():
            self.assertIsNone(p.grad)

    def test_model_average(self):
        m1 = self.make_model(0, 1)
        m2 = self.make_model(2, 3)
        dest = copy.deepcopy(m1)
        assign[dest] = model_average([m1, m2], weights=[1, 1])
        self.assert_close(dest.w1, m1.w1 * 0.5 + m2.w1 * 0.5)
        self.assert_close(dest.w2, m1.w2 * 0.5 + m2.w2 * 0.5)
        assign[dest] = model_average([m1, m2], weights=[1, 1], normalize=False)
        self.assert_close(dest.w1, m1.w1 + m2.w1)
        self.assert_close(dest.w2, m1.w2 + m2.w2)
        for p in m1.parameters():
            self.assertIsNone(p.grad)
        for p in m2.parameters():
            self.assertIsNone(p.grad)
        for p in dest.parameters():
            self.assertIsNone(p.grad)


class TestTensorFunction(unittest.TestCase):
    def assert_close(self, actual, expected, **kwargs):
        torch.testing.assert_close(actual, expected, **kwargs)

    def test_weighted_average(self):
        one = torch.ones(10, 20)
        two = torch.ones(10, 20) * 2
        result = weighted_average([one, two], weights=[1, 1])
        self.assert_close(result, one * 0.5 + two * 0.5)
        result = weighted_average([one, two], weights=[1, 1], normalize=False)
        self.assert_close(result, one + two)


if __name__ == '__main__':
    unittest.main()
