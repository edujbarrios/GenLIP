import tempfile
import unittest


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


@unittest.skipUnless(_has_torch(), "torch not installed")
class TestCheckpointLoading(unittest.TestCase):
    def test_load_state_dict_smoke(self):
        import torch

        from genlip_infer.checkpoint import load_state_dict

        state = {"layer.weight": torch.zeros((2, 3))}
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(state, f.name)
            loaded = load_state_dict(f.name)

        self.assertIn("layer.weight", loaded)


if __name__ == "__main__":
    unittest.main()

