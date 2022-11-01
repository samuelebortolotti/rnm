"""mme.py
Implements all the methods and classes for the MiniMax Entropy Model
"""
# fix absolute import for package named the same as the files
from __future__ import absolute_import
from mme.parser import Formula
from collections import OrderedDict
import tensorflow as tf
from mme.potentials import (
    LogicPotential,
    SupervisionLogicalPotential,
    CountableGroundingPotential,
)
import numpy as np
from collections.abc import Iterable


class Domain:
    """Domain class"""

    def __init__(self, name, data):
        """Constructor of the domain

        Args:
            name: name
            data: data
        """
        if name is not None:
            self.name = str(name)
        else:
            raise Exception("Attribute 'name' is None.")
        self.data = data
        self.num_constants = len(data)  # TODO check iterable


class Predicate:
    """Predicate class"""

    def __init__(self, name, domains, given=False):
        """Constructor of the predicate

        Args:
            name: name of the predicate
            domains: domains of the predicate
            given: whether it is given or not
        """
        self.name = name
        self.domains = []
        groundings_number = 1
        for domain in domains:
            if not isinstance(domain, Domain):
                raise Exception(str(domain) + " is not an instance of " + str(Domain))
            self.domains.append(domain)
            groundings_number *= domain.num_constants
        self.groundings_number = groundings_number
        self.given = given


class Ontology:
    """Ontology class"""

    def __init__(self):
        """Constructor of the Ontology class"""

        self.domains = {}
        self.predicates = OrderedDict()
        self.herbrand_base_size = 0
        self.predicate_range = {}
        self.finalized = False
        self.constraints = []

    def add_domain(self, d):
        """Add domain to the ontology

        Args:
            d: domain to add
        """
        self.finalized = False
        if not isinstance(d, Iterable):
            D = [d]
        else:
            D = d
        for d in D:
            if d.name in self.domains:
                raise Exception("Domain %s already exists" % d.name)
            self.domains[d.name] = d

    def add_predicate(self, p):
        """Add predicate to the Ontology

        Args:
            p: predicate to add
        """
        self.finalized = False
        if not isinstance(p, Iterable):
            P = [p]
        else:
            P = p
        for p in P:
            if p.name in self.predicates:
                raise Exception("Predicate %s already exists" % p.name)
            self.predicates[p.name] = p
            self.predicate_range[p.name] = (
                self.herbrand_base_size,
                self.herbrand_base_size + p.groundings_number,
            )
            self.herbrand_base_size += p.groundings_number

    def get_constraint(self, formula):
        """Get the constraints

        Args:
            formula: formula

        Returns:
            formula: formula generated from the Ontology and the formula from input"""
        return Formula(self, formula)


class MonteCarloTraining:
    """MonteCarloTraining class"""

    def __init__(
        self,
        global_potential,
        sampler,
        learning_rate=0.001,
        p_noise=0,
        num_samples=1,
        minibatch=None,
    ):
        """Constructor of the MonteCarloTraining class

        Args:
            global_potential: global potential
            sampler: sampler
            learning_rate: learning rate [default 0.001]
            p_noise=0: noise
            num_samples=1: number of sample
            minibatch=None: minibatch
        """
        self.p_noise = p_noise
        self.num_samples = num_samples
        self.global_potential = global_potential
        self.sampler = sampler
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.minibatch = minibatch  # list of indices to gather from data

    def maximize_likelihood_step(self, y, x=None):
        """The method returns a training operation for maximizing the likelihood of the model.

        Args:
            y: input tensor
            x: optional input tensor
        """

        samples = self.samples = self.sampler.sample(
            x, self.num_samples, minibatch=self.minibatch
        )

        if self.p_noise > 0:
            noise = tf.random_uniform(shape=y.shape)
            y = tf.where(noise > self.p_noise, y, 1 - y)

        if self.minibatch is not None:
            y = tf.gather(y, self.minibatch)
            if x is not None:
                x = tf.gather(x, self.minibatch)

        with tf.GradientTape(persistent=True) as tape:

            potentials_data = self.global_potential(y, x)

            potentials_samples = self.potentials_samples = self.global_potential(
                samples, x
            )

        # Compute Gradients
        vars = self.global_potential.variables

        gradient_potential_data = [
            tf.convert_to_tensor(a) / tf.cast(tf.shape(y)[0], tf.float32)
            for a in tape.gradient(target=potentials_data, sources=vars)
        ]

        E_gradient_potential = [
            tf.convert_to_tensor(a) / self.num_samples
            for a in tape.gradient(target=potentials_samples, sources=vars)
        ]

        w_gradients = [
            b - a for a, b in zip(gradient_potential_data, E_gradient_potential)
        ]

        # Apply Gradients by means of Optimizer
        grad_vars = zip(w_gradients, vars)
        self.optimizer.apply_gradients(grad_vars)


class PieceWiseTraining:
    """PieceWiseTraining class"""

    def __init__(self, global_potential, y=None, learning_rate=0.001, minibatch=None):
        """Constructor of the PieceWiseTraining class

        Args:
            global_potential: global potential
            y: optional tensor
            learning_rate: learning rate [default 0.001]
            minibatch: mini batch
        """
        self.global_potential = global_potential
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.minibatch = minibatch  # list of indices to gather from data
        self.y = y

    def compute_beta_logical_potentials(self, y=None, x=None):
        """Compute beta logical potentials

        Args:
            y: optional input tensor
            x: optional input tensor
        """
        if y is None:
            y = self.y
        for p in self.global_potential.potentials:

            if isinstance(p, CountableGroundingPotential):

                ntrue = p(y=None)
                # nfalse = (2**p.cardinality)*p.num_groundings - ntrue
                nfalse = (2**p.cardinality) - ntrue

                g, x = p.ground(y=y, x=x)
                phi_on_groundings = p.call_on_groundings(g, x)
                avg_data = tf.reduce_mean(
                    tf.cast(phi_on_groundings, tf.float32), axis=-1
                )
                # avg_data = tf.abs(avg_data -  1e-7)
                p.beta = tf.math.log(ntrue / nfalse) + tf.math.log(
                    avg_data / (1 - avg_data)
                )
                if p.beta == np.inf:
                    p.beta = tf.Variable(100.0)

    def maximize_likelihood_step(self, y, x=None, soft_xent=False):
        """The method returns a training operation for maximizing the likelihood of the model

        Args:
            y: input tensor
            x: optional input tensor
            soft_xent: soft extent
        """

        if self.minibatch is not None:
            y = tf.gather(y, self.minibatch)
            if x is not None:
                x = tf.gather(x, self.minibatch)

        for p in self.global_potential.potentials:

            if isinstance(p, SupervisionLogicalPotential):

                with tf.GradientTape(persistent=True) as tape:

                    y = p._reshape_y(y)
                    o = p.model(x)
                    if not soft_xent:
                        xent = tf.reduce_mean(
                            tf.nn.softmax_cross_entropy_with_logits(logits=o, labels=y)
                        )
                    else:
                        xent = tf.reduce_mean(-y * tf.log(tf.nn.softmax(o)))
                    xent += tf.reduce_sum(p.model.losses)

                grad = tape.gradient(target=xent, sources=p.model.variables)

                # Apply Gradients by means of Optimizer
                grad_vars = zip(grad, p.model.variables)
                self.optimizer.apply_gradients(grad_vars)
