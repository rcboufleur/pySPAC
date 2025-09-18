import unittest
from math import pi

import numpy as np

from phase_curve_lib import PhaseCurve
from constants import ALLOWED_PHASE_CURVE_MODELS

class TestPhaseCurve(unittest.TestCase):
    
    # Test cases for __init__ method
    
    def test_init_angle_magnitude_float(self):
        angle = 0.5
        magnitude = 10.0
        h = 7
        g = 0.15
        g12 = 0.15
        g1 = 0.15
        g2 = 0.25
        beta = 0.15
        
        phase_curve = PhaseCurve(
            angle=angle, 
            magnitude=magnitude, 
            H=h, G=g, G12=g12, G1=g1, G2=g2, beta=beta
        )
        
        self.assertEqual(phase_curve.angle, angle)
        self.assertEqual(phase_curve.magnitude, magnitude)
        self.assertEqual(phase_curve.params.get('H'), h)
        self.assertEqual(phase_curve.params.get('G'), g)
        self.assertEqual(phase_curve.params.get('G12'), g12)
        self.assertEqual(phase_curve.params.get('G1'), g1)
        self.assertEqual(phase_curve.params.get('G2'), g2)
        self.assertEqual(phase_curve.params.get('beta'), beta)

    def test_init_angle_magnitude_list(self):
        angle = [0.5, 1.0, 3.0, 10]
        magnitude = [10.0, 11.0, 12.0, 20]
        h = 7
        g = 0.15
        g12 = 0.15
        g1 = 0.15
        g2 = 0.25
        beta = 0.15
        
        phase_curve = PhaseCurve(
            angle=angle, 
            magnitude=magnitude, 
            H=h, G=g, G12=g12, G1=g1, G2=g2, beta=beta
        )
        
        self.assertIsInstance(phase_curve.angle, np.ndarray)
        self.assertIsInstance(phase_curve.magnitude, np.ndarray)
        self.assertTrue(np.array_equal(phase_curve.angle, np.array(angle)))
        self.assertTrue(np.array_equal(phase_curve.magnitude, np.array(magnitude)))
        self.assertEqual(phase_curve.params.get('H'), h)
        self.assertEqual(phase_curve.params.get('G'), g)
        self.assertEqual(phase_curve.params.get('G12'), g12)
        self.assertEqual(phase_curve.params.get('G1'), g1)
        self.assertEqual(phase_curve.params.get('G2'), g2)
        self.assertEqual(phase_curve.params.get('beta'), beta)

    def test_init_angle_magnitude_np_array(self):
        angle = np.array([0.5, 1.0, 3.0, 10])
        magnitude = np.array([10.0, 11.0, 12.0, 20])
        h = 7
        g = 0.15
        g12 = 0.15
        g1 = 0.15
        g2 = 0.25
        beta = 0.15
        
        phase_curve = PhaseCurve(
            angle=angle, 
            magnitude=magnitude, 
            H=h, G=g, G12=g12, G1=g1, G2=g2, beta=beta
        )
        
        self.assertIsInstance(phase_curve.angle, np.ndarray)
        self.assertIsInstance(phase_curve.magnitude, np.ndarray)
        self.assertTrue(np.array_equal(phase_curve.angle, angle))
        self.assertTrue(np.array_equal(phase_curve.magnitude, magnitude))
        self.assertEqual(phase_curve.params.get('H'), h)
        self.assertEqual(phase_curve.params.get('G'), g)
        self.assertEqual(phase_curve.params.get('G12'), g12)
        self.assertEqual(phase_curve.params.get('G1'), g1)
        self.assertEqual(phase_curve.params.get('G2'), g2)
        self.assertEqual(phase_curve.params.get('beta'), beta)

    # Test cases for generateModel method

    def test_generateModel_float(self):
        pc = PhaseCurve(angle=10.0, magnitude=15.0, H=10.0, G=0.15)
        result = pc.generateModel("HG")
        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, 10.3551525, places=5)
    
    def test_generateModel_list(self):
        pc = PhaseCurve(angle=[10.0, 20.0], magnitude=[15.0, 16.0], H=10.0, G=0.15)
        result = pc.generateModel("HG")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0], 10.3551525, places=5)
        self.assertAlmostEqual(result[1], 10.603095, places=5)
        
    def test_generateModel_with_degrees_list(self):
        pc = PhaseCurve(angle=10, magnitude=15, H=10.0, G=0.15)
        result = pc.generateModel("HG", degrees=[10, 20, 30])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 10.3551525, places=5)
        self.assertAlmostEqual(result[1], 10.603095, places=5)

    def test_generateModel_unknown_model(self):
        pc = PhaseCurve(angle=10, magnitude=15, H=10, G=0.15)
        with self.assertRaises(ValueError):
            pc.generateModel("INVALID")

    def test_generateModel_missing_parameters(self):
        pc = PhaseCurve(angle=10, magnitude=15, G=0.15)
        with self.assertRaises(ValueError):
            pc.generateModel("HG")
            
    # Test cases for fitModel method

    def test_fitModel_HG(self):
        angle = np.array([5, 10, 20, 30])
        magnitude = np.array([12.0, 12.3, 12.8, 13.5])
        pc = PhaseCurve(angle=angle, magnitude=magnitude)
        
        fit_result = pc.fitModel(model="HG", method="SLSQP")
        
        self.assertTrue(fit_result.success)
        self.assertTrue(pc.fitting_status)
        self.assertEqual(pc.fitting_model, "HG")
        self.assertIsNotNone(pc.fit_residual)
        self.assertIn('H', pc.params)
        self.assertIn('G', pc.params)

    def test_fitModel_HG1G2(self):
        angle = np.array([5, 10, 20, 30, 40])
        magnitude = np.array([12.0, 12.3, 12.8, 13.5, 14.2])
        pc = PhaseCurve(angle=angle, magnitude=magnitude)
        
        fit_result = pc.fitModel(model="HG1G2", method="SLSQP")
        
        self.assertTrue(fit_result.success)
        self.assertTrue(pc.fitting_status)
        self.assertEqual(pc.fitting_model, "HG1G2")
        self.assertIsNotNone(pc.fit_residual)
        self.assertIn('H', pc.params)
        self.assertIn('G1', pc.params)
        self.assertIn('G2', pc.params)

    def test_fitModel_not_enough_data_points(self):
        angle = np.array([10, 20])
        magnitude = np.array([15, 16])
        pc = PhaseCurve(angle=angle, magnitude=magnitude)
        with self.assertRaises(ValueError) as cm:
            pc.fitModel(model="HG1G2", method="SLSQP")
        self.assertIn("Not enough data points", str(cm.exception))

    def test_fitModel_invalid_method(self):
        angle = np.array([10, 20])
        magnitude = np.array([15, 16])
        pc = PhaseCurve(angle=angle, magnitude=magnitude)
        with self.assertRaises(ValueError) as cm:
            pc.fitModel(model="HG", method="INVALID_METHOD")
        self.assertIn("Invalid fitting method", str(cm.exception))

    def test_fitModel_with_constraints_and_unsupported_method(self):
        angle = np.array([5, 10, 20, 30])
        magnitude = np.array([12.0, 12.3, 12.8, 13.5])
        pc = PhaseCurve(angle=angle, magnitude=magnitude)
        
        # Assume a hypothetical method that doesn't support constraints
        # This test case verifies the check for `c > 0`
        with self.assertRaises(ValueError) as cm:
            _fit_model(pc.angle, pc.magnitude, "HG", "Nelder-Mead")
        self.assertIn("The 'HG' model has constraints", str(cm.exception))

    # Test cases for monteCarloUnknownRotation method

    def test_monteCarloUnknownRotation_basic(self):
        angle = np.array([5, 10, 20, 30])
        magnitude = np.array([12.0, 12.3, 12.8, 13.5])
        pc = PhaseCurve(angle=angle, magnitude=magnitude)

        n_sims = 10
        amplitude = 0.2
        model = "HG"

        results = pc.monteCarloUnknownRotation(
            n_simulations=n_sims,
            amplitude_variation=amplitude,
            model=model,
            n_threads=1
        )
        self.assertIsInstance(results, dict)
        self.assertIn('H', results)
        self.assertIn('G', results)
        self.assertEqual(len(results['H']), n_sims)
        self.assertEqual(len(results['G']), n_sims)
        self.assertIsNotNone(pc.montecarlo_uncertainty)

    # Test cases for toJSON and fromJSON methods
    
    def test_toJSON_fromJSON(self):
        angle = np.array([5, 10, 20, 30])
        magnitude = np.array([12.0, 12.3, 12.8, 13.5])
        pc_original = PhaseCurve(angle=angle, magnitude=magnitude, H=12.0, G=0.15)

        json_str = pc_original.toJSON()
        pc_restored = PhaseCurve.fromJSON(json_str)

        self.assertIsInstance(pc_restored, PhaseCurve)
        self.assertTrue(np.array_equal(pc_restored.angle, pc_original.angle))
        self.assertTrue(np.array_equal(pc_restored.magnitude, pc_original.magnitude))
        self.assertEqual(pc_restored.params, pc_original.params)
        self.assertEqual(pc_restored.fitting_status, pc_original.fitting_status)
        
    def test_toJSON_fromJSON_fitted_model(self):
        angle = np.array([5, 10, 20, 30])
        magnitude = np.array([12.0, 12.3, 12.8, 13.5])
        pc_original = PhaseCurve(angle=angle, magnitude=magnitude)
        pc_original.fitModel(model="HG", method="SLSQP")
        
        json_str = pc_original.toJSON()
        pc_restored = PhaseCurve.fromJSON(json_str)
        
        self.assertTrue(pc_restored.fitting_status)
        self.assertEqual(pc_restored.fitting_model, "HG")
        self.assertEqual(pc_restored.fitting_method, "SLSQP")
        self.assertIsInstance(pc_restored.fit_residual, list)

if __name__ == "__main__":
    # Note: _fit_model is a private function, so we need to
    # import it directly for the purpose of this specific test.
    from phase_curve_lib import _fit_model
    unittest.main()
