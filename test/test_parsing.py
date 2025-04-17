from unittest import TestCase

from phileas import iteration
from phileas.iteration.node import CartesianProduct
from phileas.parsing import load_iteration_tree_from_yaml_file


class TestParsing(TestCase):
    def test_numeric_range_parsing(self):
        content = """
!range
start: 1
end: 2
"""
        self.assertEqual(
            load_iteration_tree_from_yaml_file(content), iteration.NumericRange(1, 2)
        )

    def test_integer_range_parsing(self):
        content = """
!range
start: 1
end: 2
resolution: 1
"""
        self.assertEqual(
            load_iteration_tree_from_yaml_file(content),
            iteration.IntegerRange(1, 2, step=1),
        )

    def test_linear_range_parsing(self):
        content = """
!range
start: 1
end: 2
steps: 12
"""
        self.assertEqual(
            load_iteration_tree_from_yaml_file(content),
            iteration.LinearRange(1, 2, steps=12),
        )

    def test_geometric_range_parsing(self):
        content = """
!range
start: 1
end: 2
steps: 12
progression: geometric
"""
        self.assertEqual(
            load_iteration_tree_from_yaml_file(content),
            iteration.GeometricRange(1, 2, steps=12),
        )

    def test_sequence_parsing(self):
        content = """
!sequence
elements: [1, 2, 3]
default: 12
"""
        self.assertEqual(
            load_iteration_tree_from_yaml_file(content),
            iteration.Sequence([1, 2, 3], default_value=12),
        )

    def test_inline_sequence_parsing(self):
        content = """!sequence [1, 2, 3]"""
        self.assertEqual(
            load_iteration_tree_from_yaml_file(content),
            iteration.Sequence([1, 2, 3]),
        )

    def test_configurations(self):
        content = """
c: !configurations
  _default: config1
  _move_up: false
  _insert_name: false
  config1:
    param1: 1_1
    param2: 1_2
  config2:
    param1: 2_1
    param2: 2_2
"""
        tree = load_iteration_tree_from_yaml_file(content)
        expected_tree = iteration.CartesianProduct(
            children={
                "c": iteration.Configurations(
                    children={
                        "config1": iteration.CartesianProduct(
                            children={
                                "param1": iteration.IterationLiteral(value=11),
                                "param2": iteration.IterationLiteral(value=12),
                            },
                        ),
                        "config2": iteration.CartesianProduct(
                            children={
                                "param1": iteration.IterationLiteral(value=21),
                                "param2": iteration.IterationLiteral(value=22),
                            },
                        ),
                    },
                    default_configuration="config1",
                    move_up=False,
                    insert_name=False,
                )
            },
        )

        self.assertEqual(tree, expected_tree)

    def test_shuffle(self):
        content = """
b: !shuffle
    child: !range
        start: 0
        end: 10
        resolution: 1
"""
        tree = load_iteration_tree_from_yaml_file(content)
        expected_tree = CartesianProduct(
            children={
                "b": iteration.Shuffle(
                    child=iteration.IntegerRange(start=0, end=10, step=1),
                    seed=None,
                )
            },
        )
        self.assertEqual(tree, expected_tree)

    def test_pick(self):
        content = """
!pick
    a: !range
        start: 1
        end: 5
        resolution: 1
    b: !sequence [1, 2, 3, 4]
"""
        tree = load_iteration_tree_from_yaml_file(content)
        expected_tree = iteration.Pick(
            children={
                "a": iteration.IntegerRange(start=1, end=5, step=1),
                "b": iteration.Sequence(elements=[1, 2, 3, 4]),
            },
        )
        self.assertEqual(tree, expected_tree)

    def test_literal_tree(self):
        content = """12"""
        self.assertEqual(
            load_iteration_tree_from_yaml_file(content), iteration.IterationLiteral(12)
        )

    def test_empty_tree(self):
        content = ""
        self.assertEqual(
            load_iteration_tree_from_yaml_file(content),
            iteration.IterationLiteral(None),
        )
