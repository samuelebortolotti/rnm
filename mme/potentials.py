"""Module which deals with potentials, which are nothing but the functions of the prediction and the embedding
produced by the learner. Moreover, the potentials are used in order to define the probabilities of the outgoing label
given the hidden representation and the parameter of the reasoner. In practice, they represent the bidirectional links
between the atoms in Markov Logic Networks.
"""
import tensorflow as tf
import numpy as np
import abc


class Potential(tf.Module):
    """Class which defines a Potential, it is a Tf.Module"""

    def __init__(self):
        """Constructor of the Module
        It contains a tensorflow Variable initialized at all zeros
        """
        tf.Module.__init__(self)
        self.beta = tf.Variable(initial_value=tf.zeros(shape=()))

    @property
    @abc.abstractmethod
    def cardinality(self):
        """Cardinality property: default 1"""
        return 1

    def __call__(self, y, x=None):
        """No forward pass"""
        pass


class CountableGroundingPotential(Potential):
    """Countable Grounding Potential Class
    It is basically a tensorflow model

    Extends:
        Potential
    """

    def __init__(self):
        """Constructor of the Countable Grounding Potential"""
        super(CountableGroundingPotential, self).__init__()

    def ground(self, y, x=None):
        """Ground method"""
        pass

    def call_on_groundings(self, y, x=None):
        """Call on groundings method"""
        pass

    def reduce_groundings(self, y):
        """Reduce grounding method"""
        return tf.reduce_mean(y, axis=-1)

    def __call__(self, y, x=None):
        """Forward step of the CountableGroundingPotential

        Args:
            y: input tensor
            x: optional tensor

        Returns:
            r: tensor
        """
        g, x_g = self.ground(y, x)
        g = self.call_on_groundings(g, x_g)
        r = self.reduce_groundings(g)
        return r

    @property
    def num_groundings(self):
        """Property defining the number of grounding"""
        pass


class NeuralPotential(Potential):
    """Neural Potential representation

    Extends:
        Potential
    """

    def __init__(self, model):
        """Constructor of the Neural Potential

        Args:
            model: model
        """
        super(NeuralPotential, self).__init__()
        self.model = model

    def __call__(self, y, x=None):
        """Forward method of the NeuralPotential

        Args:
            y: input tensor
            x: optional tensor

        Returns:
            z: output tensor given by the model prediction
        """
        if x is not None:
            y = tf.concat([y, x], axis=-1)
        return self.model(y)


class FragmentedPotential(Potential):
    """FragmentedPotential class

    Extends:
        Potential
    """

    def __init__(self, base_potential):
        """Constructor of the FragmentedPotential

        Args:
            base_potential: base potential
        """
        super(FragmentedPotential, self).__init__()
        self.base_potential = base_potential

    @abc.abstractmethod
    def aggregate(self, phi):
        """Aggregate function"""
        pass

    @abc.abstractmethod
    def fragment(self, y, x=None):
        """Fragment function

        Args:
            y: input tensor
            x: optional input tensor
        """
        return None, None

    def call(self, y, x=None):
        """Forward method

        Args:
            y: input tensor
            x: optional input tensor
        """
        gamma_y, gamma_x = self.fragment(y, x)
        phi = self.base_potential(gamma_y, gamma_x)
        phi = tf.squeeze(phi, axis=-1)
        Phi = self.aggregate(phi)
        return Phi


class GlobalPotential(tf.Module):
    """GlobalPotential class

    Extends:
        tf.Module
    """

    def __init__(self, potentials=()):
        """Constructor for the GlobalPotential class

        Args:
            potentials: tuple of potentials
        """
        super(GlobalPotential, self).__init__()
        self.potentials = list(potentials)

    def add(self, potential):
        """Add potentials to the GlobalPotential

        Args:
            potential: potential to be added
        """
        self.potentials.append(potential)

    def __call__(self, y, x=None):
        """Forward method of the GlobalPotential

        Args:
            y: input tensor
            x: optional input tensor

        Returns:
            res: result of the GlobalPotential
        """
        res = 0.0
        for Phi in self.potentials:
            n = Phi.beta * Phi(y, x)
            res = res + n
        return res

    def save(self, path):
        """Save the parameter of the model

        Args:
            path: path where to save the model parameters
        """
        print(self.variables)
        ckpt = tf.train.Checkpoint(obj=self)
        ckpt.save(path)

    def restore(self, path):
        """Resore the parameter of the model

        Args:
            path: path where to restore the model parameters
        """
        ckpt = tf.train.Checkpoint(obj=self)
        ckpt.restore(path)


class LogicPotential(CountableGroundingPotential):
    """LogicPotential class

    Extends:
        CountableGroundingPotential
    """

    def __init__(self, formula, logic):
        """Constructor of the LogicPotential

        Args:
            formula: formula
            logic: type of logic
        """
        super(LogicPotential, self).__init__()
        self.formula = formula
        self.logic = logic

    @property
    def cardinality(self):
        """Cartinality property

        Returns:
            number of not given predicates in the formula atoms
        """
        return len([0 for i in self.formula.atoms if not i.predicate.given])

    @property
    def num_grounding(self):
        """Number of groundings property

        Returns:
            groundings: number of groundings of the formula
        """
        return self.formula.num_groundings

    def ground(self, y, x=None):
        """Ground

        Args:
            y: input tensor
            x: optional input tensor

        Returns:
            groundings, x: groundings of the formula and input x
        """
        if y is not None:
            groundings = self.formula.ground(
                herbrand_interpretation=y
            )  # num_examples, num_groundings, 1, num_variables_in_grounding
        else:
            groundings = self.formula.all_grounding_assignments()
        return groundings, x

    def call_on_groundings(self, y, x=None):
        """Call on groundings

        Args:
            y: input tensor
            x: optional input tensor

        Returns:
            number of satisfied groundings?
        """
        t = self.formula.compile(
            groundings=y, logic=self.logic
        )  # num_examples, num_groundings, num_possible_assignment_to_groundings
        t = tf.cast(t, tf.float32)
        return tf.reduce_sum(t, axis=-1)

    @property
    def num_groundings(self):
        """Number of groundings property

        Returns:
            number of groundings of the formula
        """
        return self.formula.num_groundings


class SupervisionLogicalPotential(Potential):
    """Logical potential with supervision

    Extends:
        Potential
    """

    def __init__(self, model, indices):
        """Constructor of the SupervisionLogicalPotential

        Args:
            model: model (neural network basically)
            indices: indices
        """
        super(SupervisionLogicalPotential, self).__init__()
        self.model = model
        self.beta = tf.Variable(initial_value=tf.ones(shape=()))
        self.indices = indices

    def _reshape_y(self, y):
        """Reshape y private method

        Args:
            y: input tensor
        """
        y = tf.gather(y, self.indices, axis=-1)
        return y

    def __call__(self, y, x=None):
        """Forward method of the SupervisionLogicalPotential

        Args:
            y: input tensor
            x: optional input tensor
        """
        y = tf.cast(y, tf.float32)  # num_examples x num_variables
        y = self._reshape_y(
            y
        )  # num_examples x num_groundings x num_variable_in_grounding
        o = self.model(x)
        o = tf.reshape(o, [y.shape[0], x.shape[-2], -1])
        t = tf.reduce_mean(o * y, axis=tf.range(len(y.shape))[1:])
        return t


class MutualExclusivityPotential(CountableGroundingPotential):
    """MutualExclusivityPotential

    Extends:
        CountableGroundingPotential
    """

    def __init__(self, indices):
        """Constructor of the MutualExclusivityPotential

        Args:
            indices: indices
        """
        super(MutualExclusivityPotential, self).__init__()
        self.indices = indices

    @property
    def cardinality(self):
        """Cardinality property

        Returns:
            the lenght of the indices at index 0
        """
        return len(self.indices[0])

    @property
    def num_groundings(self):
        """Number of groundings property

        Returns:
            length of the indices
        """
        return len(self.indices)

    def ground(self, y, x=None):
        """Ground

        Args:
            y: input tensor
            x: optional input tensor
        """
        if y is not None:
            g = tf.gather(y, self.indices, axis=-1)
        else:
            g = None
        return g, x

    def call_on_groundings(self, y, x=None):
        """Call on groundings

        Args:
            y: input tensor
            x: optional input tensor
        """
        if y is None:
            return self.cardinality * tf.ones([1, self.num_groundings])
        else:
            y = tf.cast(y, tf.float32)
        n = len(y.shape) - 1
        # o_m_y = 1 - y
        y_exp = tf.expand_dims(1 - y, axis=-2) * (1 - tf.eye(self.cardinality))
        y_exp_p_1 = y_exp + tf.eye(self.cardinality)
        ya = tf.reduce_prod(y_exp_p_1, axis=-1)
        yya = y * ya
        t = 1 - yya
        y = 1 - tf.reduce_prod(t, axis=-1)
        if len(y.shape) > 2:
            ax = tf.range(len(y.shape))[-(n - 2) :]
            return tf.reduce_sum(y, axis=ax)
        else:
            return y


class EvidenceLogicPotential(CountableGroundingPotential):
    """EvidenceLogicPotential

    Extends:
        CountableGroundingPotential
    """

    def __init__(self, formula, logic, evidence, evidence_mask):
        """Constructor of the EvidenceLogicPotential class

        Args:
            formula: logical formula
            logic: type of logic
            evidence: evidence
            evidence_mask: evidence mask
        """
        super(EvidenceLogicPotential, self).__init__()
        self.formula = formula
        self.logic = logic
        self.evidence = evidence
        self.evidence_mask = evidence_mask

    @property
    def cardinality(self):
        """Cardinality property"""
        return len([0 for i in self.formula.atoms if not i.predicate.given])

    @property
    def num_groundings(self):
        """Number of groundings property"""
        return self.formula.num_groundings

    def ground(self, y, x=None):
        """Ground

        Args:
            y: input tensor
            x: optional input tensor
        """
        if y is not None:
            groundings = self.formula.ground(
                herbrand_interpretation=y
            )  # num_examples, num_groundings, 1, num_variables_in_grounding
        else:
            groundings = self.formula.all_sample_groundings_given_evidence(
                evidence=self.evidence, evidence_mask=self.evidence_mask
            )
        return groundings, x

    def call_on_groundings(self, y, x=None):
        """Call on groundings

        Args:
            y: input tensor
            x: optional input tensor
        """
        y = self.logic.cast(y)
        t = self.formula.compile(
            groundings=y, logic=self.logic
        )  # num_examples, num_groundings, num_possible_assignment_to_groundings
        t = tf.cast(t, tf.float32)
        return tf.reduce_sum(t, axis=-1)
