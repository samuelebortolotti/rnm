# -*- coding: utf-8 -*-
"""follows_rnm.py
This file is used in order to highlight the capability of Relational Neural
Machines in order to learn and employ soft rules which are holding only ofr a sub-portion
of the whole dataset.

In particular, given a certain subset of images, a binary predicate link between image pairs
is considered.

Given two images x, y, whose corresponding digits are
denoted by i, j, a link between x and y is established if the second
digit follows the first one, i.e. i = j + 1.

However, it is assumed that the link predicate is noisy,
therefore for i != j + 1, there is a given degree of probability that the
link(x, y) is established anyway.

The knowledge about the link predicate can be represented by the following FOL formula:
∀x∀y∀i∀j link(x, y) ∧ digit(x, i) ∧ digit(y, j) ⇒ i = j + 1

Therefore, if they are digits and there is a connection between them, then it must be the case
that one follows the other.

Where digit(x, i) is a binary predicate indicating if a number i is the digit class of the image x.
Since the link predicate holds true also for pairs of non-consecutive digits, the above rule is
violated by a certain percentage of digit pairs.

The noisy links are used in order to let the reasoner to be flexible abut how to employ the knowledge.

When the link predicate only holds for consecutive digit pairs, RNM is able to perfectly predict
the images on the test set using this information.

When the link becomes less informative (more noisy), RNM is still able to employ
the rule as a soft suggestion. However, when the percentage of predictive links approaches 10%,
the relation is not informative at all, as it does not add any information on top of the prior probability
that two randomly picked up numbers follow each other.

In this case, RNM is still able to detect that the formula is not useful, and only
the supervised data is used to learn the predictions
"""
import mme
import tensorflow as tf
import datasets
import numpy as np
import os
from itertools import product

"""It is used in order to restrict the execution to a specific CUDA device"""
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

"""Set the logger level"""
tf.get_logger().setLevel("ERROR")


def main(lr, seed, perc_soft, l2w):
    """Main function of the follows_rnmm function

    Args:
        lr: learning rate
        seed: seed
        perc_soft: percentage of the soft rules, namely the percentage of assigning a link
        relationship to elements for which the relationship does not hold
        l2w: variable not accessed [0.01 by default]

    Returns:
        accuracy_map: accuracy map
        accuracy_nn: accuracy neural network
    """

    """Number of examples as listed in the paper"""
    num_examples = 50

    """Split the dataset in training and test, with the number of samples, the seed and perc_soft"""
    (x_train, hb_train), (x_test, hb_test) = datasets.mnist_follows(
        num_examples, seed=0, perc_soft=perc_soft
    )

    # I set the seed after since i want the dataset to be always the same
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # array ofzeros with the same shape and type as the array given
    m_e = np.zeros_like(hb_train)
    # selects all the lines, from number of samples * 10 to the end to 1
    m_e[:, num_examples * 10 :] = 1

    # set the y_e_train to the hb train * m_e, also when including tests
    y_e_train = hb_train * m_e
    y_e_test = hb_test * m_e

    """Logic Program Definition"""
    o = mme.Ontology()  # empty ontology for the moment

    # Domains: add the domains to the ontology
    # domain of images, associating the training samples
    images = mme.Domain("Images", data=x_train)
    # domain of numbers (digits) associating an np.array depicting the digits
    numbers = mme.Domain("Numbers", data=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).T)
    # we add the domains of the ontology to the ontology
    o.add_domain([images, numbers])

    # Predicates (given means probabily that it is already defined? Seems to be like that)
    # digit is a predicate (Digit(x, y) [image and value] in FOL) associated with the domain (images and Numbers)
    digit = mme.Predicate("digit", domains=[images, numbers])
    # link is a predicate associated to two images domain (rember) link(x, y) in FOL and it is given
    links = mme.Predicate("links", domains=[images, images], given=True)
    # follows is a predicate which is associated to the number number domains follows(x, y) in FOL
    follows = mme.Predicate("follows", domains=[numbers, numbers], given=True)
    # add the predicates digit, links and follows in FOL
    o.add_predicate([digit, links, follows])

    """MME definition"""
    # Supervision
    # first, it creates an array from 0 to images.num_constants * numbers.num_constants - 1
    # then it reshape the array to images.num_constants, numbers.num_constants
    # finally a matrix which contains progressive values from 0 onwards
    # with images.num_constants rows and numbers.num_constants columns
    indices = np.reshape(
        np.arange(images.num_constants * numbers.num_constants),
        [images.num_constants, numbers.num_constants],
    )
    """Creation of the supervised learning Neural Network"""
    nn = (
        tf.keras.Sequential()
    )  # neural network representation (basically a sequence of layers stacked)
    nn.add(tf.keras.layers.Input(shape=(784,)))  # input layer
    nn.add(
        tf.keras.layers.Dense(100, activation=tf.nn.sigmoid)  # fully connected
    )  # up to the last hidden layer
    nn.add(tf.keras.layers.Dense(10, use_bias=False))  # mlp # mlp

    """Network architecture in a nutshell

    The model is a 3 layer MLP receiving as input all the 784 pixels value, then applies a
    fully connected to reduce the shape to 100 with sigmoid as activation function
    It ends with a 10 logits without bias

    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     Input (input)               (None, 784)

     dense (Dense)               (None, 100)               78500

     dense_1 (Dense)             (None, 10)                1000

    =================================================================
    Total params: 79,500
    Trainable params: 79,500
    Non-trainable params: 0
    _________________________________________________________________
    """

    # Definition of potentials for supervision
    p1 = mme.potentials.SupervisionLogicalPotential(nn, indices)

    # Mutual Exclusivity (needed for inference, since SupervisionLogicalPotential already subsumes it during training)
    p2 = mme.potentials.MutualExclusivityPotential(indices=indices)

    # Logical formula
    c = mme.Formula(
        definition="links(x,y) and digit(x,i) and digit(y,j) -> follows(i,j)",
        ontology=o,
    )
    # evidence logic potential
    p3 = mme.potentials.EvidenceLogicPotential(
        formula=c, logic=mme.logic.BooleanLogic, evidence=y_e_train, evidence_mask=m_e
    )

    # global potential -> contains the list of potentials
    # [SupervisionLogicalPotential, MutualExclusivityPotential, EvidenceLogicPotential]
    P = mme.potentials.GlobalPotential([p1, p2, p3])

    # PieceWise Training, passing the global potential with the herbrand logic
    # (basically the training containing the 1/0 in the validation)
    pwt = mme.PieceWiseTraining(global_potential=P, y=hb_train)
    # compute beta logical potentials
    pwt.compute_beta_logical_potentials()

    """Training"""

    epochs = 150
    y_test = tf.reshape(hb_test[0, : num_examples * 10], [num_examples, 10])
    # 150 epochs
    for _ in range(epochs):
        # maximum likelihood step in order to train from the data
        pwt.maximize_likelihood_step(hb_train, x=x_train)
        # get the neural network prediction
        y_nn = p1.model(x_test)
        # compute the accuracy and print it
        acc_nn = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_nn, axis=1)), tf.float32
            )
        )
        #  print(acc_nn)

    """Inference"""
    steps_map = 500
    hb = hb_test
    x = x_test
    evidence = y_e_test
    evidence_mask = m_e > 0

    P.potentials[0].beta = 0.01
    # get the map inference
    map_inference = mme.inference.FuzzyMAPInference(
        y_shape=hb.shape,
        potential=P,
        logic=mme.logic.LukasiewiczLogic,
        evidence=evidence,
        evidence_mask=evidence_mask,
        learning_rate=lr,
    )  # tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=steps_map, decay_rate=0.96, staircase=True))

    y_test = tf.reshape(hb[0, : num_examples * 10], [num_examples, 10])
    for i in range(steps_map):
        map_inference.infer_step(x)
        if i % 10 == 0:
            y_map = tf.reshape(
                map_inference.map()[0, : num_examples * 10], [num_examples, 10]
            )
            acc_map = tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_map, axis=1)),
                    tf.float32,
                )
            )
            # Accuracy of the MAP
            #  print("Accuracy MAP", acc_map.numpy())
            #  print(y_map[:3])
        if mme.utils.heardEnter():
            break

    # Extract the inference of the model
    y_map = tf.reshape(map_inference.map()[0, : num_examples * 10], [num_examples, 10])
    y_nn = p1.model(x)

    # compute the accuracy of the map and of the nn
    acc_map = tf.reduce_mean(
        tf.cast(
            tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_map, axis=1)), tf.float32
        )
    )
    acc_nn = tf.reduce_mean(
        tf.cast(
            tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_nn, axis=1)), tf.float32
        )
    )

    return [acc_map, acc_nn]


"""Main execution"""
if __name__ == "__main__":
    """Set the seed"""
    seed = 0

    res = []
    # loop overt the tuples (x, 0.1) since product is the cartesian product
    for a in product([0, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 0.8, 1], [0.1]):
        # the first element of the loop indicates the percentage, the second the learning rate
        perc, lr = a
        print(f"percentage of the soft rules [not hold]: {perc}, learning rate: {lr}")
        # returns the accuracy map and the accuracy neural network
        acc_map, acc_nn = main(lr=lr, seed=seed, perc_soft=perc, l2w=0.01)
        # ghe the numpy vectors out of themp
        acc_map, acc_nn = acc_map.numpy(), acc_nn.numpy()
        # append to result the string separated by a tap of: percentage, learning rate, accuracy map, accuracy nn
        res.append("\t".join([str(a) for a in [perc, lr, acc_map, str(acc_nn) + "\n"]]))
        # print the elements of the result
        for i in res:
            print(i)

    # Write the percentage, learning rate, accuracy map and accuracy nn
    # in a sort of csv files
    with open("res_dlm_%d" % seed, "w") as file:
        file.write("perc, lr, acc_map, acc_nn\n")
        file.writelines(res)
