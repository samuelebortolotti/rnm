"""follows_sbr.py
Case with SBR: Semantic based regularization
Semantic-Based Regularization defines a learning and reasoning framework which allows
to train neural networks under the constraints imposed by the prior
logic knowledge.

The declarative language Lyrics is available to
provide a flexible and easy to use frontend for the SBR framework.
At training time, SBR employs the knowledge like done by LTN (Logic Tensor Networks),
while SBR uses a continuous relaxation Φsc of the c-th logic rule
and of the output vector at inference time.

Therefore, SBR can also be seen as a special instance of a RNM, when the λ parameters are
frozen and the continuous relaxation of the logic is used at test time.
Both LTN and SBR have a major disadvantage over RNM, as they
can not learn the weights of the reasoner, which are required to be
known a priori. This is very unlikely to happen in most of the real
world scenarios, where the strength of each rule must be co-trained
with the learning system.
"""
import mme
import tensorflow as tf
import datasets
import numpy as np
import os
from itertools import product
import argparse

"""Set the visible device"""
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

"""Set the logger level"""
tf.get_logger().setLevel("ERROR")


def main(lr, seed, perc_soft, l2w=0.01, w_rule=0.01):
    """Main function for the follows relationship SBR

    Args:
        lr: learning rate
        seed: seed
        perc_soft: percentage of soft rules
        l2w: weight for the l2 regularizer
        w_rule

    Returns:
        acc_nn: accuracy of the neural network
        acc_map: accuracy of the maximum a posteriori estimation
    """

    num_examples = 200  # number of samples

    # get the training and test data from the mnist follow
    (x_train, hb_train), (x_test, hb_test) = datasets.mnist_follows(
        num_examples, seed=0, perc_soft=perc_soft
    )

    # I set the seed after since i want the dataset to be always the same
    np.random.seed(seed)
    tf.random.set_seed(seed)

    m_e = np.zeros_like(hb_train)
    m_e[:, num_examples * 10 :] = 1

    y_e_train = hb_train * m_e
    y_e_test = hb_test * m_e

    """Logic Program Definition"""
    o = mme.Ontology()

    # Domains
    images = mme.Domain("Images", data=x_train)
    numbers = mme.Domain("Numbers", data=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).T)
    o.add_domain([images, numbers])

    # Predicates
    digit = mme.Predicate("digit", domains=[images, numbers])
    links = mme.Predicate("links", domains=[images, images], given=True)
    follows = mme.Predicate("follows", domains=[numbers, numbers], given=True)
    o.add_predicate([digit, links, follows])

    """MME definition"""
    # Supervision
    indices = np.reshape(
        np.arange(images.num_constants * numbers.num_constants),
        [images.num_constants, numbers.num_constants],
    )

    """Neural network definition"""
    nn = tf.keras.Sequential()
    nn.add(tf.keras.layers.Input(shape=(784,)))
    nn.add(
        tf.keras.layers.Dense(
            100,
            activation=tf.nn.sigmoid,
            kernel_regularizer=tf.keras.regularizers.l2(l2w),  # regularizer
        )
    )  # up to the last hidden layer
    nn.add(
        tf.keras.layers.Dense(10, use_bias=False, activation=None)
    )  # up to the last hidden layer

    # Adam
    adam = tf.keras.optimizers.Adam(lr=0.001)

    def make_hb_with_model(neural_digits, hb):
        """Make Herbrand with model

        Args:
            neural_digits: digits
            hb: herbrand
        """
        digits = tf.reshape(neural_digits, [-1, num_examples * 10])
        hb = tf.concat((digits, hb[:, num_examples * 10 :]), axis=1)
        return hb

    def training_step(logic=False):
        """Training step

        Args:
            logic: whether to use logic or not
        """
        # gradient update
        with tf.GradientTape() as tape:
            # logits from the neural network
            neural_logits = nn(x_train)

            # compute the loss
            total_loss = tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.reshape(hb_train[:, : num_examples * 10], [-1, 10]),
                    logits=neural_logits,
                )
            ) + tf.reduce_sum(nn.losses)

            # if there is also logic
            if logic:
                # compute the neural softmax
                neural_softmax = tf.nn.softmax(nn(x_train))

                # make hb with model
                hb_model_train = make_hb_with_model(neural_softmax, hb_train)

                # Set the formula
                c = mme.Formula(
                    definition="links(x,y) and digit(x,i) and digit(y,j) -> follows(i,j)",
                    ontology=o,
                )
                # get the groundings by ground
                groundings = c.ground(herbrand_interpretation=hb_model_train)
                # logical loss (using LukasiewiczLogic)
                logical_loss = tf.reduce_sum(
                    -c.compile(groundings, mme.logic.LukasiewiczLogic)
                )
                total_loss += w_rule * logical_loss

        # compute the gradient and apply them
        grads = tape.gradient(target=total_loss, sources=nn.variables)
        grad_vars = zip(grads, nn.variables)
        adam.apply_gradients(grad_vars)

    logic = False
    # 150 epochs
    epochs = 150
    y_test = tf.reshape(hb_test[0, : num_examples * 10], [num_examples, 10])

    # train the network
    for e in range(epochs):
        training_step(logic)
        # if e%10==0:
        #     y_nn = tf.nn.softmax(nn(x_test))
        #     acc_nn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_nn, axis=1)), tf.float32))
        #     print(acc_nn)
        # if mme.utils.heardEnter(): break
    y_nn = tf.nn.softmax(nn(x_test))
    acc_nn = tf.reduce_mean(
        tf.cast(
            tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_nn, axis=1)), tf.float32
        )
    )
    # accuracy of the neural network
    print(acc_nn)

    if logic:
        return acc_nn, acc_nn

    """MAP inference"""
    print("MAP Inference")
    steps_map = 500

    prior = tf.nn.softmax(nn(x_test))
    y_bb = tf.Variable(initial_value=prior)
    # prior = nn(x_test)
    # y_bb = tf.Variable(initial_value=0.5 * tf.ones_like(prior))

    # tet the formula
    c = mme.Formula(
        definition="links(x,y) and digit(x,i) and digit(y,j) -> follows(i,j)",
        ontology=o,
    )
    #
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     lr, decay_steps=steps_map, decay_rate=0.96, staircase=True)

    # Adam
    adam2 = tf.keras.optimizers.Adam(lr=lr)

    # Recreating an mme scenario
    mutual = mme.potentials.MutualExclusivityPotential(indices)

    # print(acc_map)
    def map_inference_step():
        """Map inference step"""

        with tf.GradientTape() as tape:
            y_map = tf.sigmoid(10 * (y_bb - 0.5))
            hb_model_test = make_hb_with_model(y_map, hb_test)
            groundings = c.ground(herbrand_interpretation=hb_model_test)
            # inference loss
            inference_loss = (
                w_rule
                * tf.reduce_sum(-c.compile(groundings, mme.logic.LukasiewiczLogic))
                + tf.reduce_sum(tf.square(prior - y_bb))
                + 100 * tf.reduce_sum(-mutual.call_on_groundings(y_map))
            )

        # learn
        grads = tape.gradient(target=inference_loss, sources=y_bb)
        grad_vars = [(grads, y_bb)]
        adam2.apply_gradients(grad_vars)

    # map inference step
    for e in range(steps_map):
        map_inference_step()
        # acc_map = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_bb, axis=1)), tf.float32))
        # print(acc_map)
        if mme.utils.heardEnter():
            break
    acc_map = tf.reduce_mean(
        tf.cast(
            tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_bb, axis=1)), tf.float32
        )
    )

    """Accuracy to be returned"""
    return acc_nn, acc_map


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="follows_sbr",
        description="Semantic Based Regularization argument.",
    )
    parser.add_argument(
        "--seed",
        metavar="SEED",
        required=False,
        default=None,
        type=int,
        help="Execution seed.",
    )
    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == "__main__":
    """Main method"""
    res = []
    seed = 0
    args = get_args()

    if args.seed is None:
        print("No seed specified, thus I am using the default one:", 0)
    else:
        print("Selected seed: ", seed)
        seed = args.seed

    print(tf.config.list_physical_devices("GPU"))
    # [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
    print(tf.test.is_built_with_cuda())
    # <function is_built_with_cuda at 0x7f4f5730fbf8>
    print(tf.test.gpu_device_name())
    # /device:GPU:0
    print(tf.config.get_visible_devices())

    # set the GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    """Also in this case, we are trying different settings for lr percentage and w_rule"""
    for lr, perc, w_rule, in product(
        [1, 0.3, 0.1, 0.06, 0.03],
        [0.0, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 0.8, 1],
        [0.1, 1.0, 10.0, 100],
    ):
        # get the accuracy of the neural network and the accuracy map
        acc_nn, acc_map = main(lr=lr, seed=seed, perc_soft=perc, w_rule=w_rule)
        # append everyting on the string
        res.append(
            "\t".join(
                [
                    str(a)
                    for a in [
                        lr,
                        perc,
                        w_rule,
                        acc_nn.numpy(),
                        str(acc_map.numpy()) + "\n",
                    ]
                ]
            )
        )
        # print the string
        for i in res:
            print(i)

    # append the result on the file
    with open(f"res_lyrics_cc_{seed}", "a") as file:
        file.write("lr,perc,w_rule,acc_nn, acc_map\n")
        file.writelines(res)
