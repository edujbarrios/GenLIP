import unittest


def _has_deps() -> bool:
    try:
        import PIL  # noqa: F401
        import torch  # noqa: F401
        import torchvision  # noqa: F401

        return True
    except Exception:
        return False


@unittest.skipUnless(_has_deps(), "torch/torchvision/Pillow not installed")
class TestPreprocessing(unittest.TestCase):
    def test_basic_preprocess_shape(self):
        from PIL import Image

        from genlip_infer.preprocessing import preprocess_image_basic

        img = Image.new("RGB", (32, 48), color=(123, 20, 220))
        x = preprocess_image_basic(img)
        self.assertEqual(tuple(x.shape), (1, 3, 224, 224))


if __name__ == "__main__":
    unittest.main()

