from unittest import TestCase

from phileas import parsing


class TestParsing(TestCase):
    def test_simple_dict(self):
        content = "a: b\nc: d"
        self.assertEqual(
            parsing.load_yaml_dict_from_file(content), {"a": "b", "c": "d"}
        )

    def test_empty_file(self):
        self.assertEqual(parsing.load_yaml_dict_from_file(""), {})

    def test_non_map_top_level(self):
        file_content = "- item1\n- item2\n"
        with self.assertRaises(ValueError):
            parsing.load_yaml_dict_from_file(file_content)

    def test_numeric_range_linear_steps(self):
        r = parsing.NumericRange(start=1, end=2, steps=10).to_array()
        self.assertEqual(len(r), 10)

    def test_numeric_range_linear_resolution(self):
        r = parsing.NumericRange(start=1, end=2, resolution=0.3).to_array()
        self.assertGreaterEqual(0.3, r[1] - r[0])

    def test_numeric_range_geometric_steps(self):
        r = parsing.NumericRange(
            start=1, end=2, steps=10, progression="geometric"
        ).to_array()
        self.assertEqual(len(r), 10)

    def test_numeric_range_geometric_resolution(self):
        r = parsing.NumericRange(
            start=1, end=2, resolution=1.1, progression="geometric"
        ).to_array()
        self.assertGreaterEqual(1.1, r[1] / r[0])

    def test_invalid_numeric_range(self):
        with self.assertRaises(ValueError):
            _ = parsing.NumericRange(start=0, end=1, steps=10, resolution=0.1)

        with self.assertRaises(ValueError):
            _ = parsing.NumericRange(start=0, end=1)

    def test_nested_numeric_range(self):
        r = parsing.load_yaml_dict_from_file(
            """
a: !range
  start: 1
  end: 2
  steps: 10
b:
  - !range
    start: 2
    end: 3
    resolution: 0.1
"""
        )
        self.assertIsInstance(r["a"], parsing.NumericRange)
        self.assertIsInstance(r["b"][0], parsing.NumericRange)

    def test_iteration_simple_dict(self):
        config = {"a": parsing.NumericRange(0, 1, 11)}
        configs = list(parsing.configurations_iterator(config))
        self.assertEqual(len(configs), 11)
        self.assertAlmostEqual(configs[0]["a"], 0)
        self.assertAlmostEqual(configs[1]["a"], 0.1)
        self.assertAlmostEqual(configs[10]["a"], 1)

    def test_iteration_nested_dict(self):
        config = {"a": {"b": parsing.NumericRange(0, 1, 11)}}
        configs = list(parsing.configurations_iterator(config))
        self.assertEqual(len(configs), 11)
        self.assertAlmostEqual(configs[0]["a"]["b"], 0)
        self.assertAlmostEqual(configs[1]["a"]["b"], 0.1)
        self.assertAlmostEqual(configs[10]["a"]["b"], 1)

    def test_iteration_nested_list(self):
        config = {"a": [parsing.NumericRange(0, 1, 11)]}
        configs = list(parsing.configurations_iterator(config))
        self.assertEqual(len(configs), 11)
        self.assertAlmostEqual(configs[0]["a"][0], 0)
        self.assertAlmostEqual(configs[1]["a"][0], 0.1)
        self.assertAlmostEqual(configs[10]["a"][0], 1)

    def test_iteration_product(self):
        config = {
            "a": parsing.NumericRange(0, 1, 3),
            "b": [parsing.NumericRange(0, 1, 3)],
        }
        configs = list(parsing.configurations_iterator(config))
        self.assertEqual(len(configs), 9)
        self.assertAlmostEqual(configs[5]["a"], 0.5)
        self.assertAlmostEqual(configs[5]["b"][0], 1)
