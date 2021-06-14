import random
from Perceptron import Perceptron

if __name__ == '__main__':

    generate_training_set = True
    num_train = 100
    generate_validation_set =True
    num_valid = 100

    training_examples = [[1.0, 1.0],
                         [1.0, 0.0],
                         [0.0, 1.0],
                         [0.0, 0.0]]

    training_labels = [1.0, 1.0, 1.0, 0.0]

    validate_examples = training_examples
    validate_labels = training_labels

    if generate_training_set:

        training_examples = []
        training_labels = []

        for i in range(num_train):
            training_examples.append([random.random(), random.random()])
            # We want our perceptron to be noise tolerant, so we label all examples where x1 and x2 > 0.8 as 1.0
            training_labels.append(1.0 if training_examples[i][0] > 0.8 or training_examples[i][1] > 0.8 else 0.0)

    if generate_validation_set:

        validate_examples = []
        validate_labels = []

        for i in range(num_train):
            validate_examples.append([random.random(), random.random()])
            validate_labels.append(1.0 if validate_examples[i][0] > 0.8 or validate_examples[i][1] > 0.8 else 0.0)


    # Create Perceptron
    OR = Perceptron(2, bias=-1.0)

    print(OR.weights)
    valid_percentage = OR.validate(validate_examples, validate_labels, verbose=True)
    print(valid_percentage)
    i = 0
    while valid_percentage < 0.98:  # We want our Perceptron to have an accuracy of at least 80%

        i += 1

        OR.train(training_examples, training_labels, 0.2)  # Train our Perceptron
        print('------ Iteration ' + str(i) + ' ------')
        print(OR.weights)
        valid_percentage = OR.validate(validate_examples, validate_labels, verbose=True)  # Validate it
        print(valid_percentage)

        # This is just to break the training if it takes over 50 iterations. (For demonstration purposes)
        # You shouldn't need to do this as your networks may require much longer to train.
        if i == 50:
            break
