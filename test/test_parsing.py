from unittest import TestCase

from phileas import iteration
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
