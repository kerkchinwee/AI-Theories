import numpy
import unittest

from network_models import Network


class NetworkTest(unittest.TestCase):
    def test_invalid_network_topology(self):
        with self.assertRaises(TypeError):
            Network(3)

    def test_empty_network_topology(self):
        network = Network(numpy.array([]))
        self.assertEqual(len(network.weights), 0)

    def test_single_layer_network_topology(self):
        network = Network(numpy.array([3]))
        self.assertEqual(len(network.weights), 0)

    def test_proper_network_without_hidden_layer_topology(self):
        network = Network(numpy.array([3, 2]))
        self.assertEqual(len(network.weights), 1)

    def test_proper_network_with_hidden_layer_topology(self):
        network = Network(numpy.array([3, 5, 4, 1]))
        self.assertEqual(len(network.weights), 3)

    def test_invalid_run_function_input(self):
        network = Network(numpy.array([3, 5, 1]))
        input = numpy.random.randn(5)
        with self.assertRaises(ValueError):
            network.run(input)

    def test_invalid_test_function_input_label_wrong_input_dimension(self):
        inputs = numpy.random.random((10, 2))
        labels = numpy.random.random((10, 2))
        network = Network(numpy.array([3, 5, 1]))
        with self.assertRaises(ValueError):
            network.test(inputs, labels)

    def test_invalid_test_function_input_label_wrong_label_dimension(self):
        inputs = numpy.random.random((10, 3))
        labels = numpy.random.random((10, 2))
        network = Network(numpy.array([3, 5, 1]))
        with self.assertRaises(ValueError):
            network.test(inputs, labels)


if __name__ == '__main__':
    unittest.main()
