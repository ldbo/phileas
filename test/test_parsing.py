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
        with self.assertRaises(ValueError):
            file_content = "- item1\n- item2\n"
            parsing.load_yaml_dict_from_file(file_content)
