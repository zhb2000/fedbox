import unittest

import numpy as np

from fedbox.utils.data import splitting


class TestSplitting(unittest.TestCase):
    @staticmethod
    def __results_to_pairs(results: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple]:
        data = np.concatenate([x for x, _ in results])
        labels = np.concatenate([y for _, y in results])
        return [(x, y) for x, y in zip(data, labels)]

    def test_split_uniformly(self):
        n = 6
        data = np.arange(n)
        labels = np.arange(n)
        client_num = 3
        results = splitting.split_uniformly(data, labels, client_num)
        self.assertEqual(len(results), client_num)
        self.assertCountEqual(
            [(x, y) for x, y in zip(data, labels)], 
            self.__results_to_pairs(results)
        )
        expected_size = n // client_num
        self.assertEqual(
            [expected_size for _ in range(client_num)],
            [len(x) for x, _ in results]
        )
        self.assertEqual(
            [expected_size for _ in range(client_num)],
            [len(y) for _, y in results]
        )

    def test_split_uniformly_stratified(self):
        data = np.arange(12)
        labels = np.array([1] * 3 + [3] * 3 + [5] * 6)
        client_num = 3
        results = splitting.split_uniformly(data, labels, client_num, stratify=labels)
        self.assertEqual(len(results), client_num)
        self.assertCountEqual(
            [(x, y) for x, y in zip(data, labels)], 
            self.__results_to_pairs(results)
        )
        for _, y in results:
            self.assertCountEqual([1, 3, 5, 5], y.tolist())

    def test_split_dirichlet_quantity(self):
        n = 6
        client_num = 3
        data = np.arange(n)
        labels = np.arange(n)
        results = splitting.split_dirichlet_quantity(data, labels, client_num, alpha=1.0)
        self.assertEqual(len(results), client_num)
        self.assertCountEqual(
            [(x, y) for x, y in zip(data, labels)], 
            self.__results_to_pairs(results)
        )

    def test_split_dirichlet_quantity_stratified(self):
        n = 6
        client_num = 3
        data = np.arange(n)
        labels = np.arange(n)
        results = splitting.split_dirichlet_quantity(data, labels, client_num, alpha=1.0, stratify=labels)
        self.assertEqual(len(results), client_num)
        self.assertCountEqual(
            [(x, y) for x, y in zip(data, labels)], 
            self.__results_to_pairs(results)
        )

    def test_split_dirichlet_label(self):
        n = 6
        client_num = 3
        data = np.arange(n)
        labels = np.array([0, 1, 2, 2, 1, 0])
        results = splitting.split_dirichlet_label(data, labels, client_num, alpha=1.0)
        self.assertEqual(len(results), client_num)
        self.assertCountEqual(
            [(x, y) for x, y in zip(data, labels)], 
            self.__results_to_pairs(results)
        )

    def test_split_dirichlet_label_non_continuous_label(self):
        n = 6
        client_num = 3
        data = np.arange(n)
        labels = np.array([1, 3, 5, 3, 1, 5])
        results = splitting.split_dirichlet_label(data, labels, client_num, alpha=1.0)
        self.assertEqual(len(results), client_num)
        self.assertCountEqual(
            [(x, y) for x, y in zip(data, labels)], 
            self.__results_to_pairs(results)
        )

    def test_split_by_label(self):
        n = 6
        client_num = 3
        class_per_client = 2
        data = np.arange(n)
        labels = np.array([0, 1, 2, 2, 1, 0])
        results = splitting.split_by_label(data, labels, client_num, class_per_client=class_per_client)
        self.assertEqual(len(results), client_num)
        self.assertCountEqual(
            [(x, y) for x, y in zip(data, labels)], 
            self.__results_to_pairs(results)
        )
        self.assertEqual(
            [class_per_client for _ in range(client_num)],
            [len(np.unique(y)) for _, y in results]
        )

    def test_split_by_label_non_continuous_label(self):
        n = 6
        client_num = 3
        class_per_client = 2
        data = np.arange(n)
        labels = np.array([1, 3, 5, 3, 1, 5])
        results = splitting.split_by_label(data, labels, client_num, class_per_client=class_per_client)
        self.assertEqual(len(results), client_num)
        self.assertCountEqual(
            [(x, y) for x, y in zip(data, labels)], 
            self.__results_to_pairs(results)
        )
        self.assertEqual(
            [class_per_client for _ in range(client_num)],
            [len(np.unique(y)) for _, y in results]
        )


class TestSplittingHelperFunc(unittest.TestCase):
    def test_shuffle_together2(self):
        n = 6
        data = np.empty((n, 5, 4, 3))
        labels = np.empty(n)
        shuffled_data, shuffled_labels = splitting._shuffle_together(data, labels)
        self.assertEqual(data.shape, shuffled_data.shape)
        self.assertEqual(labels.shape, shuffled_labels.shape)

    def test_shuffle_together_different_len(self):
        data = np.empty((6, 5, 4, 3))
        labels = np.empty(5)
        with self.assertRaises(ValueError):
            _, _ = splitting._shuffle_together(data, labels)

    def test_ceil_div(self):
        self.assertEqual(1, splitting._ceil_div(1, 1))
        self.assertEqual(3, splitting._ceil_div(3, 1))
        self.assertEqual(1, splitting._ceil_div(1, 3))
        self.assertEqual(3, splitting._ceil_div(7, 3))

    def test_split_by_ratios(self):
        result = splitting._array_split_by_ratios(np.zeros(5), np.array([0.2, 0.3]))
        self.assertEqual([2, 3], [len(x) for x in result])
        result = splitting._array_split_by_ratios(np.zeros(5), np.array([20, 30]))
        self.assertEqual([2, 3], [len(x) for x in result])
        result = splitting._array_split_by_ratios(np.zeros(5), np.array([1, 0.1, 1, 0.1, 1, 1, 1]))
        self.assertEqual([1, 0, 1, 0, 1, 1, 1], [len(x) for x in result])
        result = splitting._array_split_by_ratios(np.zeros(10), np.array([0.26, 0.26, 0.48]))
        self.assertCountEqual([2, 3, 5], [len(x) for x in result])
        result = splitting._array_split_by_ratios(np.zeros(8), np.repeat(1 / 5, 5))
        self.assertCountEqual([2, 2, 2, 1, 1], [len(x) for x in result])
        result = splitting._array_split_by_ratios(np.zeros(160), ratios=np.repeat(1 / 100, 100))
        self.assertCountEqual([1] * 40 + [2] * 60, [len(x) for x in result])

    def test_split_by_client_proportion(self):
        proportion = np.array([
            [1., 0., 0.],  # class 0
            [1., 1., 0.],  # class 1
        ])
        labels = np.array([0, 0, 0, 1, 1])
        results = splitting._split_by_proportion(np.arange(5), labels, proportion)
        clients_labels = [y for _, y in results]
        self.assertEqual(3, len(results))
        self.assertCountEqual([0, 0, 0, 1], clients_labels[0].tolist())
        self.assertCountEqual([1], clients_labels[1].tolist())
        self.assertCountEqual([], clients_labels[2].tolist())

    def test_map_labels_to_continuous(self):
        labels = np.array([2, 4, 6, 7, 9, 11, 6, 2])
        classes = np.unique(labels)
        new_labels = splitting._map_labels_to_continuous(labels, classes)
        self.assertEqual([0, 1, 2, 3, 4, 5, 2, 0], new_labels.tolist())
        self.assertEqual(labels.tolist(), splitting._recover_labels(new_labels, classes).tolist())


if __name__ == '__main__':
    unittest.main()
