# Perceptron TRaining Rule(X,epsilon):
# Initialise w(wi<- an initial (small) random value):
# Repeat:
#   for each training instance(x,tx) E X:
#       compute the real output ox=Activation (Summation(w,x)):
#       if(tx not equal ox):
#           for each wi:
#               wi<- wi+ delta(wi)
#               delta(wi)<- epsilon(tx-ox)xi
#           end for
#       end if
#   end for
# until all the training instances in X are correctly classified
# return w


import random
from typing import List

from Perceptron import Perceptron

# Create Perceptrons
NOT = Perceptron(1, bias=0.5)
AND = Perceptron(2, bias=-1.0)
OR = Perceptron(2, bias=-1.0)


def main():
    generate_training_set = True
    num_train = 100
    generate_validation_set = True
    num_valid = 100

    training_examples_all: List[float] = [[1.0, 1.0],
                                          [1.0, 0.0],
                                          [0.0, 1.0],
                                          [0.0, 0.0]]

    training_examples_NOT: List[float] = [[1.0], [0.0]]

    training_labels_and: List[float] = [1.0, 0.0, 0.0, 0.0]

    training_labels_or: List[float] = [1.0, 1.0, 1.0, 0.0]

    training_labels_not: List[float] = [0.0, 1.0]

    # temporary variables
    validate_examples = []
    validate_labels = []
    training_examples = []
    training_labels = []

    # train AND gate
    if generate_training_set:
        training_examples, training_labels = generateTrainingSetAnd(num_train)
    else:
        training_examples, training_labels = training_examples_all, training_labels_and
    if generate_validation_set:
        validate_examples, validate_labels = generateValidationSetAnd(num_train)
    else:
        validate_examples, validate_labels = training_examples_all, training_labels_and

    train(AND, validate_examples, validate_labels, training_examples, training_labels)

    # Train the OR Gates
    if generate_training_set:
        training_examples, training_labels = generateTrainingSetOr(num_train)
    else:
        training_examples, training_labels = training_examples_all, training_labels_or
    if generate_validation_set:
        validate_examples, validate_labels = generateValidationSetOr(num_train)
    else:
        validate_examples, validate_labels = training_examples_all, training_labels_or

    train(OR, validate_examples, validate_labels, training_examples, training_labels)

    # Train the not gate
    if generate_training_set:
        training_examples, training_labels = generateTrainingSetNot(num_train)
    else:
        training_examples, training_labels = training_examples_NOT, training_labels_not
    if generate_validation_set:
        validate_examples, validate_labels = generateValidationSetNot(num_train)
    else:
        validate_examples, validate_labels = training_examples_NOT, training_labels_not
    train(NOT, validate_examples, validate_labels, training_examples, training_labels)

    # Test the Gates if they work
    print("____Testing_Logic_Gates_____")
    print("AND GATE:")
    Gate(AND)
    print("OR GATE:")
    Gate(OR)
    print("NOT GATE:")
    Gate(NOT)

    # Prompt user to enter input then implement the XOR gate to it to see if it works
    print("Enter X1 & X2 for testing XOR GATE: ")
    X1: float = float(input("Enter X1: "))
    X2: float = float(input("Enter X2: "))

    OUTPUT: float = XOR(X1, X2)
    print("XOR GATE Output: %.1f" % OUTPUT)


# Check if the gates where properly trained
def Gate(GATE: Perceptron):
    # for AND & OR gates
    if GATE.num_inputs == 2:
        print(GATE.activate([0.0, 0.0]))
        print(GATE.activate([0.0, 1.0]))
        print(GATE.activate([1.0, 0.0]))
        print(GATE.activate([1.0, 1.0]))
    # for NOT GATE
    elif GATE.num_inputs == 1:
        print(GATE.activate([0.0]))
        print(GATE.activate([1.0]))


# XOR GATE= (OR) AND (NOT(AND))
# Y1= X1 AND X2
# Y2= NOT(Y1)
# Y3= X1 (OR) X2
# Y4 = Y3(AND) Y2 -----XOR GATE SOLUTION
def XOR(X1: float, X2: float):
    # Y1=X1 AND X2:
    Y1: float = AND.activate([X1, X2])
    # Y2= NOT(Y1)
    Y2: float = NOT.activate([Y1])
    # Y3= X1 (OR) X2
    Y3: float = OR.activate([X1, X2])
    # Y4 = Y3(AND) Y2 -----XOR GATE SOLUTION
    Y4: float = AND.activate([Y3, Y2])

    return Y4


# Generate Validation Set for AND GATE
def generateValidationSetAnd(num_train):
    validate_examples = []
    validate_labels = []

    for i in range(num_train):
        zero: float = random.uniform(-0.2, 0.2)
        one: float = random.uniform(0.2, 1.2)
        validate_examples.append([random.choice([zero, one]), random.choice([zero, one])])
        validate_labels.append(1.0 if validate_examples[i][0] > 0.75 and validate_examples[i][1] > 0.75 else 0.0)

    return validate_examples, validate_labels


# Generate Training Set for AND GATE
def generateTrainingSetAnd(num_train):
    training_examples = []
    training_labels = []

    for i in range(num_train):
        zero: float = random.uniform(-0.2, 0.2)
        one: float = random.uniform(0.2, 1.2)
        training_examples.append([random.choice([zero, one]), random.choice([zero, one])])
        # We want our perceptron to be noise tolerant, so we label all examples where x1 and x2 > 0.75 as 1.0
        training_labels.append(1.0 if training_examples[i][0] > 0.75 and training_examples[i][1] > 0.75 else 0.0)

    return training_examples, training_labels


# Generate Validation Set for OR GATE
def generateValidationSetOr(num_train):
    validate_examples: float = []
    validate_labels: float = []

    for i in range(num_train):
        zero: float = random.uniform(-0.2, 0.2)
        one: float = random.uniform(0.2, 1.2)
        validate_examples.append([random.choice([zero, one]), random.choice([zero, one])])
        # validate_labels.append(1.0 if (validate_examples[i][0] > 0.75 or validate_examples[i][1] > 0.75) else 0.0)
        if validate_examples[i][0] > 0.75:
            validate_labels.append(1.0)
        elif validate_examples[i][1] > 0.75:
            validate_labels.append(1.0)
        else:
            validate_labels.append(0.0)

    return validate_examples, validate_labels


# Generate Training Set for OR GATE
def generateTrainingSetOr(num_train):
    training_examples: float = []
    training_labels: float = []
    for i in range(num_train):
        zero: float = random.uniform(-0.2, 0.2)
        one: float = random.uniform(0.8, 1.2)
        training_examples.append([random.choice([zero, one]), random.choice([zero, one])])
        # We want our perceptron to be noise tolerant, so we label all examples where x1 and x2 > 0.75 as 1.0
        # training_labels.append(1.0 if training_examples[i][0] > 0.75 or training_examples[i][1] > 0.75 else 0.0)
        if training_examples[i][0] > 0.75:
            training_labels.append(1.0)
        elif training_examples[i][1] > 0.75: \
                training_labels.append(1.0)
        else:
            training_labels.append(0.0)
    return training_examples, training_labels


# Generate Validation Set for NOT GATE
def generateValidationSetNot(num_train):
    validate_examples: float = []
    validate_labels: float = []

    for i in range(num_train):
        zero: float = random.uniform(-0.2, 0.2)
        one: float = random.uniform(0.8, 1.2)
        validate_examples.append([random.random()])
        validate_labels.append(0.0 if validate_examples[i][0] > 0.75 else 1.0)

    return validate_examples, validate_labels


# Generate Training Set for NOT GATE
def generateTrainingSetNot(num_train):
    training_examples: float = []
    training_labels: float = []

    for i in range(num_train):
        zero: float = random.uniform(-0.2, 0.2)
        one: float = random.uniform(0.8, 1.2)
        training_examples.append([random.random()])
        # We want our perceptron to be noise tolerant, so we label all examples where x1 and x2 > 0.75 as 1.0
        training_labels.append(0.0 if training_examples[i][0] > 0.75 else 1.0)

    return training_examples, training_labels


# Train the logic gates
def train(LogicGate: Perceptron, validate_examples, validate_labels, training_examples,
          training_labels):
    print(LogicGate.weights)
    valid_percentage = LogicGate.validate(validate_examples, validate_labels, verbose=True)
    print(valid_percentage)
    i = 0
    while valid_percentage < 0.98:  # We want our Perceptron to have an accuracy of at least 80%
        i += 1
        LogicGate.train(training_examples, training_labels, 0.2)  # Train our Perceptron
        print('------ Iteration ' + str(i) + ' ------')
        print(LogicGate.weights)
        valid_percentage = LogicGate.validate(validate_examples, validate_labels, verbose=True)  # Validate it
        print(valid_percentage)

        # This is just to break the training if it takes over 50 iterations. (For demonstration purposes)
        # You shouldn't need to do this as your networks may require much longer to train.
        if i == 50:
            break


if __name__ == '__main__':
    main()
