"""logic.py
This module deals with the implementation fo the logic
"""
import tensorflow as tf


class BooleanLogic:
    """Class which depicts the Boolean Logic, it provides all the operators which are needed in a
    FOL (First Order Logic)
    """

    @staticmethod
    def cast(y):
        """Static method which casts the input value to a Tensorflow boolean variable

        Args:
            y: The value to be converted

        Returns:
            tf.bool: tensorflow boolean variable
        """
        return tf.cast(y, tf.bool)

    @staticmethod
    def _not(args):
        """Static method which returns the negation of a boolean variable
        It performs the cast to tf.bool and then it returns the logical not
        Asserts an exception if the n-ary array passed in input contains more than one element

        Args:
            arg: The n-ary array of the single element to be negated

        Returns:
            n-ary tf.bool single element array of the negation
        """
        assert len(args) == 1, "N-Ary negation not defined"
        args = [tf.cast(a, tf.bool) for a in args]
        return tf.logical_not(args[0])

    @staticmethod
    def _and(args):
        """Static method which returns the conjunction of an n-ary boolean vector
        It performs the cast to tf.bool and then it returns the logical and element-wise

        Args:
            arg: The n-ary array to be conjuncted

        Returns:
            n-ary tf.bool containing the result of the element-wise conjunction
        """
        args = [tf.cast(a, tf.bool) for a in args]
        t = tf.stack(args, axis=-1)
        return tf.reduce_all(t, axis=-1)

    @staticmethod
    def _or(args):
        """Static method which returns the disjunction of an n-ary boolean vector
        It performs the cast to tf.bool and then it returns the logical or element-wise

        Args:
            arg: The n-ary array to be disjuncted

        Returns:
            n-ary tf.bool containing the result of the disjunction
        """
        args = [tf.cast(a, tf.bool) for a in args]
        t = tf.stack(args, axis=-1)
        return tf.reduce_any(t, axis=-1)

    @staticmethod
    def _implies(args):
        """Static method which returns the implication of an n-ary boolean vector
        It performs the cast to tf.bool and then it returns the logical implication element-wise
        It asserts an exception if the array contains more or less than two elements.
        This operation is performed by simply rewriting:
        a => b      as      not a or b

        Args:
            arg: The n-ary array included in the implication

        Returns:
            n-ary tf.bool contiaining the result of the implication
        """
        assert len(args) == 2, "N-Ary implies not defined"
        args = [tf.cast(a, tf.bool) for a in args]
        t = tf.logical_or(tf.logical_not(args[0]), args[1])
        return t

    @staticmethod
    def _iff(args):
        """Static method which returns the iff operation of an n-ary boolean vector
        It performs the cast to tf.bool and then it returns the logical iff element-wise
        It asserts an exception if the array contains more than two elements.

        Args:
            arg: The n-ary array in the iff

        Returns:
            n-ary tf.bool containing the result of the iff
        """
        assert len(args) == 2, "N-Ary iff not defined"
        args = [tf.cast(a, tf.bool) for a in args]
        t = tf.equal(args[0], args[1])
        return t

    @staticmethod
    def _xor(args):
        """Static method which returns the xor operation of an n-ary boolean vector
        It performs the cast to tf.bool and then it returns the logical xor element-wise
        It asserts an exception if the array contains more than two elements.

        Args:
            arg: The n-ary array to be xored

        Returns:
            n-ary tf.bool containing the result of the xor operation
        """
        assert len(args) == 2, "N-Ary xor not defined"
        args = [tf.cast(a, tf.bool) for a in args]
        t = tf.math.logical_xor(args[0], args[1])
        return t


class LukasiewiczLogic:
    """Class which depicts the Lukasiewicz Logic, it provides all the operators which are needed for the fuzzy logic
    see https://en.wikipedia.org/wiki/%C5%81ukasiewicz_logic
    """

    @staticmethod
    def cast(y):
        """Static method which returns the element in input as no cast is required.
        It implements the identity

        Args:
            y: The value to be converted

        Returns:
            y: input
        """
        return y

    @staticmethod
    def _not(args):
        """Static method which returns the negation of a boolean variable in the Lukasiewicz logic
        Asserts an exception if the n-ary array passed in input contains more than one element

        Args:
            arg: The n-ary array of the single element to be negated

        Returns:
            n-ary tf.bool single element array of the negation (1 - value)
        """
        assert len(args) == 1, "N-Ary negation not defined"
        return 1 - args[0]

    @staticmethod
    def _and(args):
        """Static method which returns the conjunction of boolean variables in the Lukasiewicz logic

        Args:
            arg: The n-ary array of the single element to be conjuncted

        Returns:
            n-ary tf.bool result of the conjunction
        """
        t = tf.stack(args, axis=-1)
        return tf.reduce_sum(t - 1, axis=-1) + 1

    @staticmethod
    def _or(args):
        """Static method which returns the disjunction of boolean variables in the Lukasiewicz logic

        Args:
            arg: The n-ary array of the single element to be conjuncted

        Returns:
            n-ary tf.bool result of the conjunction
        """
        t = tf.stack(args, axis=-1)
        return tf.minimum(1.0, tf.reduce_sum(t, axis=-1))

    @staticmethod
    def _implies(args):
        """Static method which returns the implication of boolean variables in the Lukasiewicz logic
        Throws an assertion if the variables are more or less than two elements

        Args:
            arg: The n-ary array of the single element included in the implication

        Returns:
            n-ary tf.bool result of the implication
        """
        assert len(args) == 2, "N-Ary implies not defined"
        t = tf.minimum(1.0, 1 - args[0] + args[1])
        return t

    @staticmethod
    def _iff(args):
        """Static method which returns the result of the iff of boolean variables in the Lukasiewicz logic

        Args:
            arg: The n-ary array of the single element included in the iff

        Returns:
            n-ary tf.bool result of the iff
        """
        t = 1 - tf.abs(args[0] - args[1])
        return t

    @staticmethod
    def _xor(args):
        """Static method which returns the result of the xor of boolean variables in the Lukasiewicz logic
        Throws an assertion if the variables are more or less than two elements

        Args:
            arg: The n-ary array of the single element to be xored

        Returns:
            n-ary tf.bool result of the xor
        """
        assert len(args) == 2, "N-Ary xor not defined"
        return tf.abs(args[0] - args[1])


class ProductLogic:
    """Class which depicts the Product Logic, it provides all the operators which are needed for the fuzzy logic"""

    @staticmethod
    def cast(y):
        """Static method which returns the element in input as no cast is required.
        It implements the identity

        Args:
            y: The value to be converted

        Returns:
            y: input
        """
        return y

    @staticmethod
    def _not(args):
        """Static method which returns the negation of a boolean variable in the Product logic
        Asserts an exception if the n-ary array passed in input contains more than one element

        Args:
            arg: The n-ary array of the single element to be negated

        Returns:
            n-ary tf.bool single element array of the negation (1 - value)
        """
        assert len(args) == 1, "N-Ary negation not defined"
        return 1 - args[0]

    @staticmethod
    def _and(args):
        """Static method which returns the conjunction of boolean variables in the Product logic

        Args:
            arg: The n-ary array of the single element to be conjucted

        Returns:
            n-ary tf.bool the result of the conjunction
        """
        t = tf.stack(args, axis=-1)
        return tf.reduce_prod(t - 1, axis=-1) + 1

    @staticmethod
    def _or(args):
        """Static method which returns the disjunction of boolean variables in the Product logic
        Asserts an exception if the n-ary array passed in input contains more than one element

        Args:
            arg: The n-ary array of the single element to be disjuncted

        Returns:
            n-ary tf.bool the result of the disjuncted
        """
        assert len(args) == 1, "N-Ary or not defined for product t-norm"
        return args[0] + args[1] + args[0] * args[1]

    @staticmethod
    def _implies(args):
        """Static method which returns the implication of boolean variables in the Product logic
        Asserts an exception if the n-ary array passed in input contains more or less than two element

        Args:
            arg: The n-ary array of the single element in the implication

        Returns:
            n-ary tf.bool the result of the implication
        """
        assert len(args) == 2, "N-Ary implies not defined"
        a = args[0]
        b = args[1]
        return tf.where(a > b, b / (a + 1e-12), tf.ones_like(a))

    @staticmethod
    def _iff(args):
        """Static method which returns the disjunction of boolean variables in the Product logic
        Asserts an exception if the n-ary array passed in input contains more or less than two element

        Args:
            arg: The n-ary array of the single element to be disjuncted

        Returns:
            n-ary tf.bool the result of the iff
        """
        assert len(args) == 2, "N-Ary <-> not defined"
        a = args[0]
        b = args[1]
        return 1 - a * (1 - b) + (1 - a) * b - a * (1 - b) * (1 - a) * b

    @staticmethod
    def _xor(args):
        """Static method which returns the xor of boolean variables in the Product logic
        Asserts an exception if the n-ary array passed in input contains more or less than two element

        Args:
            arg: The n-ary array of the single element to be xored

        Returns:
            n-ary tf.bool the result of the xor
        """
        assert len(args) == 2, "N-Ary xor not defined"
        a = args[0]
        b = args[1]
        return a * (1 - b) + (1 - a) * b - a * (1 - b) * (1 - a) * b
