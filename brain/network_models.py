import numpy


class Network:

    """ Basic Neural Network model """

    def __init__(self, network_topology):
        """
        Neural Network model initializer
        :param network_topology: 1D integer array value which indicate the number of nodes in each network layer.
                                 There must be at least 2 integers (first for input layer and last for output layer)
                                 to create the network model properly
        """
        layers = []
        weights = []
        loop = len(network_topology) - 1
        for i in range(0, loop):
            layer = numpy.random.randn(network_topology[i])
            layers.append(layer)
            weight = numpy.random.random((network_topology[i + 1], network_topology[i]))
            weights.append(weight)
        self.layers = numpy.array(layers)
        self.weights = numpy.array(weights)

    def activation_function(self, input):
        """
        Neural network activation function which preserves outputs from any network layer
        before the outputs are being further used
        Override this function to give proper activation function to this network model
        :param input: output of any network layer
        :return: processed output of any network layer
        """
        return input

    def run(self, input):
        """
        run the network model to generate output based on the input given
        :param input: 1D numeric array value which is expected to have a dimension value same as first integer given
                      by network topology
        :return: 1D numeric array value which is expected to have a dimension value same as last integer given
                 by network topology
        """
        output = input.copy()
        i = 0
        for weight in self.weights:
            self.layers[i] = output
            i += 1
            output = numpy.dot(weight, output)
            output = self.activation_function(output)
        return output

    def test(self, inputs, labels):
        """
        evaluate the network model performance by getting the Mean Squared Error (MSE) of its performance
        :param inputs: 2D numeric array value which represents the batch of inputs to the network.
                       The array is expected to have each element with a dimension value same as
                       first integer given by network topology
        :param labels: 2D numeric array value which represents the batch of desired outputs generated
                       from the network.
                       The array is expected to have each element with a dimension value same as
                       last integer given by network topology
        :return: MSE of the performance
        """
        loop = len(inputs)
        mse = numpy.zeros(len(labels[0]))
        for i in range(0, loop):
            output = self.run(inputs[i])
            if len(output) != len(labels[i]):
                raise ValueError()
            error = output - labels[i]
            mse += (error * error)
        mse /= loop
        return mse


if __name__ == '__main__':
    nn = Network(numpy.array([3, 5, 4, 1]))
    output = nn.run(numpy.random.randn(3))
    print(output)
    ip = numpy.random.random((10, 3))
    lb = numpy.random.random((10, 1))
    mse = nn.test(ip, lb)
    print(mse)
