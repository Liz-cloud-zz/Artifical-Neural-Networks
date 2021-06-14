# Simple Perceptron class
# June 2021

import random


class Perceptron:

    def __init__(self, num_inputs: int, bias: float = 0.0, seeded_weights: [float] = None,
                 float_threshold: float = 0.0005):
        """NOTE: if you want to seed your weights, the size of seeded weights must be greater than or equal to
        num_inputs """

        self.num_inputs = num_inputs

        # The line below uses list comprehension to generate the weights list for out perceptron. If seeded_weights
        # is None, the weights will be randomly initialized to be a value between [0,1) else the weights[i] will be
        # set to whatever the value of seeded_weights[i] is.
        self.weights = [random.random() for i in range(num_inputs)] if seeded_weights is None else [seeded_weights[i]
                                                                                                    for i in
                                                                                                    range(num_inputs)]
        self.bias = bias
        self.float_threshold = float_threshold

    def activate(self, inputs: [float]) -> float:

        weighted_sum = sum([self.weights[i] * inputs[i] for i in range(self.num_inputs)])

        return 1.0 if weighted_sum + self.bias > 0 else 0.0

    def predict(self, inputs: [float]) -> bool:
        return self.activate(inputs) == 1.0

    def learn(self, index: int, output: float, target_output: float, x: float, learning_rate: float):
        self.weights[index] += learning_rate * (target_output - output) * x

    def train(self, examples: [[float]], labels: [float], learning_rate: float):
        """NOTE: This function will run the each input example[i] through the activate function and compare its output to the label labels[i].
		if the output does not match, the perceptron learning rule is applied and each weight of the perceptron is modified in accordance with the rule."""

        for i in range(len(examples)):

            predicted_val = self.activate(examples[i])

            for w in range(self.num_inputs):
                self.learn(w, predicted_val, labels[i], examples[i][w], learning_rate)

    def validate(self, examples: [[float]], labels: [float], verbose: bool = False) -> float:
        num_correct = 0.0

        for i in range(len(examples)):

            prediction = self.activate(examples[i])
            if abs(prediction - labels[i]) < self.float_threshold:
                num_correct += 1.0
            elif verbose:
                print('Perceptron failed on example' + str(examples[i]) + '\nPredicted: ' + str(
                    prediction) + ' Correct Output: ' + str(labels[i]))

        return num_correct / len(examples)
