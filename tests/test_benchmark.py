from __future__ import annotations

import unittest

from scripts.benchmark import _percentile


class BenchmarkTests(unittest.TestCase):
    def test_percentile_uses_interpolation_not_max(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 100.0]

        self.assertAlmostEqual(_percentile(values, 0.95), 80.8)


if __name__ == "__main__":
    unittest.main()
