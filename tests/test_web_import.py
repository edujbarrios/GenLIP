import unittest


class TestWebImport(unittest.TestCase):
    def test_import_web_app(self):
        import web.app  # noqa: F401


if __name__ == "__main__":
    unittest.main()

