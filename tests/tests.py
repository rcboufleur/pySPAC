# test_pyspac_methods.py

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pyspac import PhaseCurve


def test_init():
    """Test PhaseCurve initialization."""
    pc = PhaseCurve(angle=[10, 20], magnitude=[12.0, 12.5], H=8.0, G=0.15)
    assert np.allclose(pc.angle, [10, 20])
    assert np.allclose(pc.magnitude, [12.0, 12.5])
    assert pc.params["H"] == 8.0
    assert pc.params["G"] == 0.15


def test_str():
    """Test __str__ method."""
    pc = PhaseCurve(angle=[10], magnitude=[12.0])
    assert "1 data points" in str(pc)
    assert "Not yet fitted" in str(pc)


def test_clear_fit_results():
    """Test _clear_fit_results resets fitting state."""
    pc = PhaseCurve(angle=[10], magnitude=[12.0], H=8.0)
    pc.fitting_status = True
    pc.params["extra"] = 999
    pc._clear_fit_results()
    assert not pc.fitting_status
    assert "extra" not in pc.params
    assert "H" in pc.params  # initial param preserved


def test_summary():
    """Test summary method prints basic info."""
    pc = PhaseCurve(angle=[10, 20], magnitude=[12.0, 12.5], H=8.0)
    with patch('builtins.print') as mock_print:
        pc.summary()
        printed = " ".join([str(call) for call in mock_print.call_args_list])
        assert "Data Points" in printed


def test_estimate_uncertainty_from_rms():
    """Test RMS uncertainty estimation (mocked)."""
    # Use 3 data points so degrees of freedom > 0 for 2 fit parameters
    pc = PhaseCurve(angle=[10, 20, 30], magnitude=[12.0, 12.5, 12.2])
    with patch.object(pc, 'fitModel') as mock_fit:
        mock_result = MagicMock()
        # residual length must match number of data points
        mock_result.residual = np.array([0.1, -0.1, 0.05])
        # two params (H and G) -> dof = 3 - 2 = 1 (>0)
        mock_result.params = {"H": MagicMock(), "G": MagicMock()}
        mock_fit.return_value = mock_result

        uncs = pc._estimate_uncertainty_from_rms("HG", "SLSQP")
        assert np.all(uncs > 0)
        assert uncs.shape == pc.magnitude.shape


def test_generate_model():
    """Test generateModel for all models."""
    pc = PhaseCurve(angle=10.0, magnitude=12.0, H=8.0, G=0.15, G12=0.2, G1=0.1, G2=0.3, beta=0.03)
    for model in ["HG", "HG12", "HG12PEN", "HG1G2", "LINEAR"]:
        result = pc.generateModel(model)
        assert isinstance(result, (float, np.ndarray))


def test_fit_model():
    """Test fitModel method (mocked)."""
    pc = PhaseCurve(angle=[10, 20], magnitude=[12.0, 12.5])
    with patch('pyspac.pyspac._fit_model') as mock_fit:
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.residual = np.array([0.1, -0.1])
        mock_params = MagicMock()
        mock_h = MagicMock()
        mock_h.value = 8.0
        mock_g = MagicMock()
        mock_g.value = 0.15
        mock_params.items.return_value = [('H', mock_h), ('G', mock_g)]
        mock_result.params = mock_params
        mock_fit.return_value = mock_result

        result = pc.fitModel("HG", "SLSQP")
        assert result == mock_result
        assert pc.fitting_status


def test_monte_carlo_uncertainty():
    """Test monteCarloUncertainty (mocked)."""
    pc = PhaseCurve(angle=[10, 20], magnitude=[12.0, 12.5], magnitude_unc=[0.1, 0.1])
    with patch('pyspac.pyspac.Pool') as mock_pool:
        with patch('pyspac.pyspac._fit_wrapper') as mock_wrapper:
            mock_result = MagicMock()
            mock_result.success = True
            mock_h = MagicMock()
            mock_h.value = 8.0
            mock_g = MagicMock()
            mock_g.value = 0.15
            mock_result.params = {"H": mock_h, "G": mock_g}
            mock_wrapper.return_value = mock_result

            mock_pool_instance = MagicMock()
            mock_pool.return_value.__enter__.return_value = mock_pool_instance
            mock_pool_instance.imap_unordered.return_value = [mock_result] * 5

            pc.monteCarloUncertainty
