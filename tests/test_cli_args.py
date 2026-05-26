import unittest


class TestCliArgs(unittest.TestCase):
    def test_argparse_parses_required(self):
        from scripts.infer import build_arg_parser

        p = build_arg_parser()
        args = p.parse_args(
            [
                "--image",
                "x.jpg",
                "--checkpoint",
                "ckpt.pt",
                "--config",
                "cfg.yaml",
                "--device",
                "cpu",
                "--dtype",
                "fp32",
                "--json",
            ]
        )
        self.assertEqual(args.image, "x.jpg")
        self.assertEqual(args.checkpoint, "ckpt.pt")
        self.assertEqual(args.config, "cfg.yaml")
        self.assertTrue(args.json)


if __name__ == "__main__":
    unittest.main()

