import unittest

from fedbox.utils.training import EarlyStopper


class TestEarlyStopper(unittest.TestCase):
    def test_higher_metric(self):
        stopper = EarlyStopper(higher_better=True)
        self.assertTrue(stopper.is_better(1))
        self.assertTrue(stopper.update(1, other=1))
        self.assertFalse(stopper.is_better(0))
        self.assertFalse(stopper.update(0, other=0))
        self.assertTrue(stopper.is_better(2))
        self.assertTrue(stopper.update(2, other=2))
        self.assertEqual(2, stopper.best_metric)
        self.assertEqual(2, stopper.dict['other'])
        self.assertEqual(2, stopper['other'])

    def test_lower_metric(self):
        stopper = EarlyStopper(higher_better=False)
        self.assertTrue(stopper.is_better(1))
        self.assertTrue(stopper.update(1, other=1))
        self.assertFalse(stopper.is_better(2))
        self.assertFalse(stopper.update(2, other=2))
        self.assertTrue(stopper.is_better(0))
        self.assertTrue(stopper.update(0, other=0))
        self.assertEqual(0, stopper.best_metric)
        self.assertEqual(0, stopper.dict['other'])
        self.assertEqual(0, stopper['other'])

    def test_early_stopping(self):
        stopper = EarlyStopper(higher_better=True, patience=2)
        self.assertTrue(stopper.update(50))
        self.assertFalse(stopper.reach_stop())
        self.assertFalse(stopper.update(49))  # worse 1
        self.assertFalse(stopper.reach_stop())
        self.assertFalse(stopper.update(49))  # worse 2
        self.assertTrue(stopper.reach_stop())
        self.assertFalse(stopper.update(49))  # worse 3
        self.assertTrue(stopper.reach_stop())
        self.assertTrue(stopper.update(51))  # better
        self.assertFalse(stopper.reach_stop())


if __name__ == '__main__':
    unittest.main()
