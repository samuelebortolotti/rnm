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
    """Domain class
    Defines an ontology domain.
    It basically consists in a name and a series of data
    """

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
    """Predicate class
    Defines the predicate for the ontology
    It basically consists in a name, some domains and a given flag
    """

    def __init__(self, name, domains, given=False):
        """Constructor of the predicate
        Raises an exception if the domain passed are not instances of Domain class

        Args:
            name: name of the predicate
            domains: domains of the predicate
            given: whether it is given or not
        """
        self.name = name
        self.domains = []
        # initial groundings number is 1
        groundings_number = 1
        # for each domain
        for domain in domains:
            # domain should be a subclass of Domain
            if not isinstance(domain, Domain):
                raise Exception(str(domain) + " is not an instance of " + str(Domain))
            # append the domain, and multiply the grounding nmbers with the constants number in the domain
            self.domains.append(domain)
            groundings_number *= domain.num_constants
        self.groundings_number = groundings_number
        self.given = given


class Ontology:
    """Ontology class
    Defines the application logic
    """

    def __init__(self):
        """Constructor of the Ontology class

        An Herbrand base is a set of all the ground atoms of whose argument terms are the Herbrand Universe.

        Attributes:
            - domains: domains of the logic
            - predicates: logical predicates
            - herbrand base size: size of the Herbrand base
            - finalized: whether they are finalized or not
            - constraints: constraints which holds
        """

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
    """PieceWiseTraining class

    From what I know, this class shoul define the training both for the supervised (subsymbolic learning)
    And for the symbolic learning
    """

    import wandb

    def __init__(self, global_potential, y=None, learning_rate=0.001, minibatch=None):
        """Constructor of the PieceWiseTraining class

        Args:
            global_potential: global potential (container of potentials)
            y: optional tensor
            learning_rate: learning rate [default 0.001]
            minibatch: mini batch
        """
        self.global_potential = global_potential
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)  # Adam optimizer
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

        # for all the potentils in the global potentials
        for p in self.global_potential.potentials:
            # if the potential is a countable grounding potential
            if isinstance(p, CountableGroundingPotential):
                # call the potential and get the number of true clauses
                ntrue = p(y=None)
                # nfalse = (2**p.cardinality)*p.num_groundings - ntrue (simple relationshp)
                nfalse = (2**p.cardinality) - ntrue

                # ground the y
                g, x = p.ground(y=y, x=x)
                # get the phi on groundings
                phi_on_groundings = p.call_on_groundings(g, x)
                # compute the average data
                avg_data = tf.reduce_mean(
                    tf.cast(phi_on_groundings, tf.float32), axis=-1
                )
                # avg_data = tf.abs(avg_data -  1e-7)
                # compute the bet a
                p.beta = tf.math.log(ntrue / nfalse) + tf.math.log(
                    avg_data / (1 - avg_data)
                )
                # top limit with the beta
                if p.beta == np.inf:
                    p.beta = tf.Variable(100.0)

    def maximize_likelihood_step(self, y, x=None, soft_xent=False):
        """The method returns a training operation for maximizing the likelihood of the model

        Args:
            y: input tensor
            x: optional input tensor
            soft_xent: soft extent ?
        """

        # if minibatch there are not None
        if self.minibatch is not None:
            y = tf.gather(y, self.minibatch)
            # if x is not None
            if x is not None:
                x = tf.gather(x, self.minibatch)

        # for potentials in potentials
        for p in self.global_potential.potentials:
            # if this is a supervision logical potential
            if isinstance(p, SupervisionLogicalPotential):
                # compute the gradients with respect to some inputs (the tf.Variable)
                # [basically learning procedure]
                with tf.GradientTape(persistent=True) as tape:

                    y = p._reshape_y(y)
                    # prediction
                    o = p.model(x)
                    if not soft_xent:
                        xent = tf.reduce_mean(
                            tf.nn.softmax_cross_entropy_with_logits(logits=o, labels=y)
                        )
                    else:
                        xent = tf.reduce_mean(-y * tf.log(tf.nn.softmax(o)))
                    xent += tf.reduce_sum(p.model.losses)

                # compute the gradient thanks to the tape
                wandb.log({"nn/loss": xent})
                grad = tape.gradient(target=xent, sources=p.model.variables)

                # Apply Gradients by means of Optimizer
                grad_vars = zip(grad, p.model.variables)
                # apply gradients using the optimizer
                self.optimizer.apply_gradients(grad_vars)
