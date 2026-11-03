import sys
import tempfile
import unittest
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from nae import Engine, EngineError, NeuralWeights


def cells(engine, width):
    return {(i % width, i // width) for i, v in enumerate(engine.read()) if v > 0.5}


class LifeBindingTests(unittest.TestCase):
    def test_blinker_oscillates(self):
        with Engine.life(5, 5) as engine:
            frame = array("f", [0.0] * 25)
            for x in (1, 2, 3):
                frame[2 * 5 + x] = 1.0
            engine.write(frame)

            engine.step()
            self.assertEqual(cells(engine, 5), {(2, 1), (2, 2), (2, 3)})
            engine.step()
            self.assertEqual(cells(engine, 5), {(1, 2), (2, 2), (3, 2)})
            self.assertEqual(engine.generation, 2)

    def test_glider_translates(self):
        start = {(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)}
        with Engine.life(8, 8) as engine:
            frame = array("f", [0.0] * 64)
            for x, y in start:
                frame[y * 8 + x] = 1.0
            engine.write(frame)
            engine.step(4)
            self.assertEqual(cells(engine, 8), {(x + 1, y + 1) for x, y in start})
            engine.step(28)
            self.assertEqual(cells(engine, 8), start)

    def test_seed_is_deterministic(self):
        with Engine.life(32, 32) as a, Engine.life(32, 32) as b:
            a.seed_random(42, 0.3)
            b.seed_random(42, 0.3)
            self.assertEqual(a.read(), b.read())
            a.step(10)
            b.step(10)
            self.assertEqual(a.read(), b.read())

    def test_seed_density_is_plausible(self):
        with Engine.life(64, 64) as engine:
            engine.seed_random(5, 0.25)
            alive = sum(1 for v in engine.read() if v > 0.5)
            self.assertTrue(0.2 * 4096 < alive < 0.3 * 4096)

    def test_custom_rule_highlife(self):
        with Engine.life(16, 16, birth=(3, 6), survive=(2, 3)) as engine:
            engine.seed_random(9, 0.3)
            engine.step(5)
            self.assertEqual(engine.generation, 5)

    def test_closed_engine_raises(self):
        engine = Engine.life(4, 4)
        engine.close()
        with self.assertRaises(EngineError):
            engine.step()

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            Engine.life(4, 4, boundary="torus")
        with self.assertRaises(EngineError):
            Engine.life(0, 4)
        with self.assertRaises(ValueError):
            Engine.life(4, 4, birth=(40,))

    def test_write_shape_mismatch(self):
        with Engine.life(4, 4) as engine:
            with self.assertRaises(EngineError):
                engine.write(array("f", [0.0] * 10))


class NeuralBindingTests(unittest.TestCase):
    def test_drift(self):
        weights = NeuralWeights.zeros(2, 1)
        weights.b1[0] = 1.0
        weights.w2[0] = 0.5
        weights.w2[1] = 0.25
        with Engine.neural(6, 5, weights) as engine:
            engine.step(3)
            frame = engine.read()
            self.assertAlmostEqual(frame[0], 1.5, places=5)
            self.assertAlmostEqual(frame[1], 0.75, places=5)
            self.assertEqual(engine.shape, (5, 6, 2))

    def test_seed_point_and_clip(self):
        weights = NeuralWeights.random(3, 8, seed=1, step_size=0.5, state_clip=1.0)
        with Engine.neural(16, 16, weights) as engine:
            engine.seed_point(8, 8, 1.0)
            engine.step(20)
            self.assertTrue(all(-1.0 <= v <= 1.0 for v in engine.read()))

    def test_weight_validation(self):
        with self.assertRaises(ValueError):
            NeuralWeights(2, 2, w1=[0.0], b1=[0.0] * 2, w2=[0.0] * 4, b2=[0.0] * 2)

    def test_weight_roundtrip(self):
        weights = NeuralWeights.random(2, 4, seed=11)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "w.json"
            weights.save(path)
            loaded = NeuralWeights.load(path)
        self.assertEqual(loaded.w1, weights.w1)
        self.assertEqual(loaded.b2, weights.b2)
        self.assertEqual(loaded.hidden, 4)

    def test_oversized_network_rejected(self):
        weights = NeuralWeights.zeros(2, 300)
        with self.assertRaises(EngineError):
            Engine.neural(8, 8, weights)


if __name__ == "__main__":
    unittest.main()
