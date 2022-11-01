"""test.py
Test some tensorflow functionalities
"""

import os, sys

# Add the parent to the test path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from mme import Ontology, Domain, Predicate
from itertools import product
import numpy as np
import tensorflow as tf
import datasets


num_examples = 10
rng = np.arange(num_examples)
x_ids = np.reshape(np.tile(rng, [num_examples, 1]).T, [-1])
y_ids = np.tile(rng, [num_examples])

print(x_ids)
print(y_ids)
