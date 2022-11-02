"""dataset.py
Modules which deals with the dataset
"""
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import scipy.ndimage as img
import matplotlib.pyplot as plt

"""
Set the numpy random seed
"""
np.random.seed(0)


def mnist_linked_plus_minus_1(num_examples):
    """Minist plus the link with plus and minus knowledge

    Args:
        num_examples: number of examples
    """

    def __inner__(y_train):
        """Inner method of the dataset

        Args:
            y_train: label of the training data
        """
        hb_link = np.zeros([num_examples, num_examples])
        for i, x in enumerate(y_train):
            for j, y in enumerate(y_train):
                if abs(x - y) == 1:
                    hb_link[i, j] = 1
        hb_link = np.reshape(hb_link, [1, -1])

        hb_pm1 = np.zeros([10, 10])
        for i in range(10):
            for j in range(10):
                if abs(i - j) == 1:
                    hb_pm1[i, j] = 1
        hb_pm1 = np.reshape(hb_pm1, [1, -1])

        y_train = np.eye(10)[y_train]
        hb_digit = np.reshape(y_train, [1, -1])

        hb_equal = np.reshape(np.eye(10), [1, -1])

        hb = np.concatenate([hb_digit, hb_link, hb_pm1, hb_equal], axis=1)
        return hb

    # train and test set split
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # select the samples
    x_train = x_train[:num_examples]
    y_train = y_train[:num_examples]

    # select the samples
    x_test = x_test[:num_examples]
    y_test = y_test[:num_examples]

    x_train = np.reshape(x_train, [-1, 784])
    hb_train = __inner__(y_train)

    x_test = np.reshape(x_test, [-1, 784])
    hb_test = __inner__(y_test)

    return (x_train, hb_train), (x_test, hb_test)


def mnist_equal(num_examples):
    """Mnist equal function

    Args:
        num_examples: number of examples
    """

    def __inner__(y_train):
        """Inner method of the dataset

        Args:
            y_train: label of the training data
        """
        y_train = np.eye(10)[y_train]
        hb_digit = np.reshape(y_train, [1, -1])

        hb_equal = np.reshape(np.eye(10), [1, -1])

        hb = np.concatenate([hb_digit, hb_equal], axis=1)
        return hb

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train[:num_examples]
    y_train = y_train[:num_examples]

    x_test = x_test[:num_examples]
    y_test = y_test[:num_examples]

    x_train = np.reshape(x_train, [-1, 784])
    hb_train = __inner__(y_train)

    x_test = np.reshape(x_test, [-1, 784])
    hb_test = __inner__(y_test)

    return (x_train, hb_train), (x_test, hb_test)


def mnist_follows(num_examples, seed=0, perc_soft=0.1):
    """Mnist follows

    Args:
        num_examples: number of examples to be retrieved
        seed: seed
        perc_soft: perc soft, percentage of assigning a link relationship for pairs
        in which it does not hold. It can be seen as a noise component
    """

    """Split the dataset into training and test"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print(len(x_train))
    """If the examples requested are more than those available, then raise an exception"""
    if num_examples > len(x_test):
        raise Exception("I cannot create a test of this size.")

    """Split the dataset into train and validation subsets for the training set
        given, the training set of minist, the size (proportion of the dataset to include),
        the stratification is used in order to sample data whose labels come from the passsed data
        random state is used for reproducibility
    """
    _, x_train, _, y_train = train_test_split(
        x_train, y_train, test_size=num_examples, stratify=y_train, random_state=0
    )
    """Split the dataset into train and validation subsets for the rest set
        given, the training set of minist, the size (proportion of the dataset to include),
        the stratification is used in order to sample data whose labels come from the passsed data
        random state is used for reproducibility
    """
    _, x_test, _, y_test = train_test_split(
        x_test, y_test, test_size=num_examples, stratify=y_test, random_state=0
    )

    def _inner(x, y):
        """Inner function of the follows mnist dataset preparation

        It normalizes the data instances and prepares the link relationship:
        - for all the samples where the link instances is satisfied: 1 is returned
        - for the others, according to the noise -> soft perc probablity the link
        relationship may be satisfied

        The Herbrand Logic has the same syntax of the FOL.
        see "https://www.cs.uic.edu/~hinrichs/herbrand/html/herbrandlogic.html"

        Args:
            x: data instances (untouched)
            y: an herbrand logic set (digits: matrix of digits value),
            follows(bottom corner matrix where follows relationship holds) and links (link relationship established)
        """
        x_new = []
        # normalize the images
        for I in x:
            I = I / 255
            # I = img.rotate(I, float(np.random.rand() * 90), reshape=False)
            # I = I + 0.3 * np.random.randn(28,28)
            x_new.append(I)
        # prepare the new data set
        x = np.reshape(x_new, [-1, 28 * 28])

        # prepare the links: touple n_examples, n_examples
        links = np.zeros([num_examples, num_examples])
        # loop over the labels
        for i, y_i in enumerate(y):
            # loop over the labels
            for j, y_j in enumerate(y):

                # if the follow relationship is satisfied:
                # ∀x∀y∀i∀j link(x, y) ∧ digit(x, i) ∧ digit(y, j) ⇒ i = j + 1
                # se the link relationship of the matrix to true
                if y_i == y_j + 1:
                    # if np.random.rand() < 0.9:
                    links[i, j] = 1
                else:
                    # else se the matrix relationship according to the soft rule percentage probability
                    r = np.random.rand()
                    if r < perc_soft:
                        links[i, j] = 1

        # reshaping the link
        links = np.reshape(links, [1, -1])

        # prepare a follows 10x10 matrix of zeros
        follows = np.zeros([10, 10])
        for i in range(10):
            for j in range(10):
                if i == j + 1:
                    # assign 1 to all the values for which the follows hold
                    # in this 10x10 matrix
                    # in practice it is a lower corner matrix of 1s
                    follows[i, j] = 1

        # reshape the follows
        follows = np.reshape(follows, [1, -1])

        # the label set is defined as a 10 diagonal matrix times y matrix
        y = np.eye(10)[y]
        # basically we are setting the labels in this way as one-hot encoding
        digit = np.reshape(y, [1, -1])  # reshaping the digit as a matrix of y

        # herbrand logic is the concatenation of the digit (groudtruth)
        # links relationship
        # follows matrix
        hb = np.concatenate((digit, links, follows), axis=1)

        # returns the training set (untouched) and the concatenation of digits, link and follows
        return (x, hb)

    # returns the data instances and the herbrand logic for both train and test set
    return _inner(x_train, y_train), _inner(x_test, y_test)


def citeseer(test_size=0.5, valid_size=0.0):
    """Citeseer dataset

    Args:
        test_size: test size in percentage
        valid_size: validation set in percentage
    """
    documents = np.load("data/citeseer/words.npy")
    n = len(documents)
    documents = documents[:n]
    labels = np.load("data/citeseer/labels.npy")
    labels = labels[:n]
    labels = np.eye(6)[labels]
    citations = np.greater(np.load("data/citeseer/citations.npy"), 0).astype(np.float32)
    citations = citations[:n, :n]
    num_documents = len(documents)
    num_classes = 6

    def _inner_take_hb(idx):

        x = documents[idx]
        l = np.reshape(labels[idx].T, [1, -1])
        c = np.reshape(citations[idx][:, idx], [1, -1])

        hb = np.concatenate((l, c), axis=1)
        hb = hb.astype(np.float32)

        return x, hb

    trid, teid = train_test_split(
        np.arange(num_documents), test_size=test_size, random_state=0
    )
    trid, vaid = (
        train_test_split(trid, test_size=valid_size, random_state=0)
        if valid_size > 0
        else (trid, None)
    )

    x_train, hb_train = _inner_take_hb(trid)
    x_valid, hb_valid = _inner_take_hb(vaid) if valid_size > 0 else (None, None)
    x_test, hb_test = _inner_take_hb(teid)

    return (x_train, hb_train), (x_valid, hb_valid), (x_test, hb_test)


def citeseer_em(test_size, valid_size, seed):
    """Citeseer em dataset

    Args:
        test_size: test size in percentage
        valid_size: validation set in percentage
        seed: seed
    """
    # get the documents
    documents = np.load("data/citeseer/words.npy")
    n = len(documents)
    documents = documents[:n]
    # get the labels
    labels = np.load("data/citeseer/labels.npy")
    labels = labels[:n]
    labels = np.eye(6)[labels]  # labels lines in np.eye
    citations = np.greater(np.load("data/citeseer/citations.npy"), 0).astype(np.float32)
    citations = citations[:n, :n]
    num_documents = len(documents)

    def _inner_take_hb(idx):
        """Inner take herbrand

        Args:
            idx: indices

        Returns:
            x: documents
            hb
        """
        x = documents[idx]
        l = np.reshape(labels[idx].T, [1, -1])
        c = np.reshape(citations[idx][:, idx], [1, -1])

        hb = np.concatenate((l, c), axis=1)
        hb = hb.astype(np.float32)

        return x, hb

    # split data in training and test
    trid, teid = train_test_split(
        np.arange(num_documents), test_size=test_size, random_state=seed
    )

    # get training and validation
    trid, vaid = (
        train_test_split(trid, test_size=valid_size, random_state=seed)
        if valid_size > 0
        else (trid, None)
    )

    # mask_training labels
    mask_train_labels = np.zeros_like(labels)
    # set to 1 to all the trid
    mask_train_labels[trid] = 1

    x_train, hb_train = _inner_take_hb(trid)
    x_valid, hb_valid = _inner_take_hb(vaid) if valid_size > 0 else (None, None)
    x_test, hb_test = _inner_take_hb(teid)
    x_all, hb_all = _inner_take_hb(np.arange(n))

    return (
        (x_train, hb_train),
        (x_valid, hb_valid),
        (x_test, hb_test),
        (x_all, hb_all),
        labels,
        mask_train_labels,
        trid,
        vaid,
        teid,
    )
