import unittest
from math import pi

import numpy as np

from pyspac import (
    HG,
    HG1G2,
    HG12,
    HG12PEN,
    HBetaLinear,
    PhaseCurve,
    allowedPhaseCurveModels,
    call_model_function,
    degree_to_radians,
    get_model_class_parameters,
)


class TestFunctions(unittest.TestCase):
    def test_degree_to_radians(self):
        self.assertAlmostEqual(degree_to_radians(0), 0)
        self.assertAlmostEqual(degree_to_radians(45), pi / 4)
        self.assertAlmostEqual(degree_to_radians(-90), -pi / 2)
        self.assertAlmostEqual(degree_to_radians(360), 2 * pi)

    def test_HG(self):
        self.assertAlmostEqual(HG(20, 7, 0.15), 8.00010912)

    def test_HG12(self):
        self.assertAlmostEqual(HG12(20, 7, 0.15), 7.85185626)

    def test_HG12PEN(self):
        self.assertAlmostEqual(HG12PEN(20, 7, 0.15), 7.95718260)

    def test_HG1G2(self):
        self.assertAlmostEqual(HG1G2(20, 7, 0.15, 0.25), 8.42062809)

    def test_HBetaLinea(self):
        self.assertAlmostEqual(HBetaLinear(20, 7, 0.15), 10)


class TestCallModelFunctions(unittest.TestCase):
    def test_call_model_function(self):
        self.assertEqual(call_model_function("HG"), HG)
        self.assertEqual(call_model_function("HG12"), HG12)
        self.assertEqual(call_model_function("HG12PEN"), HG12PEN)
        self.assertEqual(call_model_function("HG1G2"), HG1G2)
        self.assertEqual(call_model_function("LINEAR"), HBetaLinear)


class TestGetModelClassParameters(unittest.TestCase):
    def test_HG(self):
        instance = PhaseCurve(
            angle=0, magnitude=0, H=1.0, G=2.0
        )  # Create an instance of PhaseCurve with desired values
        result = get_model_class_parameters("HG", instance)
        self.assertEqual(result, [1.0, 2.0])

    def test_HG12(self):
        instance = PhaseCurve(angle=0, magnitude=0, H=1.0, G12=3.0)
        result = get_model_class_parameters("HG12", instance)
        self.assertEqual(result, [1.0, 3.0])

    def test_HG12PEN(self):
        instance = PhaseCurve(angle=0, magnitude=0, H=1.0, G12=3.0)
        result = get_model_class_parameters("HG12PEN", instance)
        self.assertEqual(result, [1.0, 3.0])

    def test_HG1G2(self):
        instance = PhaseCurve(angle=0, magnitude=0, H=1.0, G1=2.0, G2=3.0)
        result = get_model_class_parameters("HG1G2", instance)
        self.assertEqual(result, [1.0, 2.0, 3.0])

    def test_LINEAR(self):
        instance = PhaseCurve(angle=0, magnitude=0, H=1.0, beta=0.5)
        result = get_model_class_parameters("LINEAR", instance)
        self.assertEqual(result, [1.0, 0.5])


class TestPhaseCurve(unittest.TestCase):
    # case of angle and magnitude float
    def test_init_angle_magnitude_float(self):
        angle = 0.5
        magnitude = 10.0
        h = 7
        g = 0.15
        g12 = 0.15
        g1 = 0.15
        g2 = 0.25
        bt = 0.15
        phase_curve = PhaseCurve(
            angle=angle, magnitude=magnitude, H=h, G=g, G12=g12, G1=g1, G2=g2, beta=bt
        )
        self.assertEqual(phase_curve.angle, angle)
        self.assertEqual(phase_curve.magnitude, magnitude)
        self.assertEqual(phase_curve.H, h)
        self.assertEqual(phase_curve.G, g)
        self.assertEqual(phase_curve.G12, g12)
        self.assertEqual(phase_curve.G1, g1)
        self.assertEqual(phase_curve.G2, g2)
        self.assertEqual(phase_curve.beta, bt)
        for allowed_model in allowedPhaseCurveModels:
            self.assertIsInstance(phase_curve.model(allowed_model.lower()), float)
            self.assertIsInstance(phase_curve.model(allowed_model), float)

    # case of angle and magnitude list
    def test_init_angle_magnitude_list(self):
        angle = [0.5, 1.0, 3.0, 10]
        magnitude = [10.0, 11.0, 12.0, 20]
        h = 7
        g = 0.15
        g12 = 0.15
        g1 = 0.15
        g2 = 0.25
        bt = 0.15
        phase_curve = PhaseCurve(
            angle=angle, magnitude=magnitude, H=h, G=g, G12=g12, G1=g1, G2=g2, beta=bt
        )
        self.assertEqual(phase_curve.angle, angle)
        self.assertEqual(phase_curve.magnitude, magnitude)
        self.assertEqual(phase_curve.H, h)
        self.assertEqual(phase_curve.G, g)
        self.assertEqual(phase_curve.G12, g12)
        self.assertEqual(phase_curve.G1, g1)
        self.assertEqual(phase_curve.G2, g2)
        self.assertEqual(phase_curve.beta, bt)
        for allowed_model in allowedPhaseCurveModels:
            self.assertIsInstance(phase_curve.model(allowed_model.lower()), list)
            self.assertIsInstance(phase_curve.model(allowed_model), list)

    # case of angle and magnitude np array
    def test_init_angle_magnitude_np_array(self):
        angle = np.array([0.5, 1.0, 3.0, 10])
        magnitude = np.array([10.0, 11.0, 12.0, 20])
        h = 7
        g = 0.15
        g12 = 0.15
        g1 = 0.15
        g2 = 0.25
        bt = 0.15
        phase_curve = PhaseCurve(
            angle=angle, magnitude=magnitude, H=h, G=g, G12=g12, G1=g1, G2=g2, beta=bt
        )
        self.assertTrue(np.array_equal(phase_curve.angle, angle))
        self.assertTrue(np.array_equal(phase_curve.magnitude, magnitude))
        self.assertEqual(phase_curve.H, h)
        self.assertEqual(phase_curve.G, g)
        self.assertEqual(phase_curve.G12, g12)
        self.assertEqual(phase_curve.G1, g1)
        self.assertEqual(phase_curve.G2, g2)
        self.assertEqual(phase_curve.beta, bt)
        for allowed_model in allowedPhaseCurveModels:
            self.assertIsInstance(phase_curve.model(allowed_model.lower()), np.ndarray)
            self.assertIsInstance(phase_curve.model(allowed_model), np.ndarray)


if __name__ == "__main__":
    unittest.main()
