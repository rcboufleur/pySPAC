import json
from multiprocessing import Pool
from typing import Dict, List, Optional, Union

import numpy as np
import sbpy.photometry as pm
from lmfit import Parameters, minimize
from lmfit.model import ModelResult

from .constants import (
    ALLOWED_PHASE_CURVE_MODELS,
    FITTING_METHODS,
    PARAMETER_KEYS,
    NUM_PARAMETERS,
    NUM_CONSTRAINTS,
)


# Private helper functions for model evaluation
def _HG(angle: np.ndarray, H: float, G: float) -> np.ndarray:
    """Evaluates the HG model."""
    return pm.HG.evaluate(np.deg2rad(angle), H, G)


def _HG12(angle: np.ndarray, H: float, G12: float) -> np.ndarray:
    """Evaluates the HG12 model."""
    return pm.HG12.evaluate(np.deg2rad(angle), H, G12)


def _HG12PEN(angle: np.ndarray, H: float, G12: float) -> np.ndarray:
    """Evaluates the HG12PEN model."""
    return pm.HG12_Pen16.evaluate(np.deg2rad(angle), H, G12)


def _HG1G2(angle: np.ndarray, H: float, G1: float, G2: float) -> np.ndarray:
    """Evaluates the HG1G2 model."""
    return pm.HG1G2.evaluate(np.deg2rad(angle), H, G1, G2)


def _HBetaLinear(angle: np.ndarray, H: float, beta: float) -> np.ndarray:
    """Evaluates the LINEAR model."""
    return angle * beta + H


def _call_model_function(model: str):
    """Returns the correct model function based on its name."""
    function_dict = {
        "HG": _HG,
        "HG12": _HG12,
        "HG12PEN": _HG12PEN,
        "HG1G2": _HG1G2,
        "LINEAR": _HBetaLinear,
    }
    return function_dict[model]


def _montecarlo_simulated_magnitudes(
    magnitudes: np.ndarray,
    n_simulations: int,
    distribution: str,
    amplitude_variation: float,
) -> np.ndarray:
    """Generates simulated magnitude datasets with added random noise."""
    if distribution not in ["uniform", "sinusoidal"]:
        raise ValueError(
            "Invalid distribution. Must be 'uniform' or 'sinusoidal'."
        )

    magnitudes = np.asarray(magnitudes)
    if distribution == "uniform":
        random_values = (
            np.random.uniform(-1, 1, (n_simulations, len(magnitudes)))
            * amplitude_variation
        )
    elif distribution == "sinusoidal":
        random_values = (
            np.sin(
                np.random.uniform(
                    0, 2 * np.pi, (n_simulations, len(magnitudes))
                )
            )
            * amplitude_variation
        )
    return magnitudes + random_values


def _fit_residual(
    params: Parameters,
    angles: np.ndarray,
    magnitudes: np.ndarray,
    model: str,
    param_keys: List[str],
) -> np.ndarray:
    """Calculates the residuals for the fitting function."""
    parameters = [params[key].value for key in param_keys]
    fn = _call_model_function(model)
    model_values = fn(angles, *parameters)
    return magnitudes - model_values


def _fitting_model_parameter_object(
    model: str, avg_H: float, initial_conditions: Optional[List[List]] = None
) -> Union[Parameters, List[Parameters]]:
    """Sets up the initial parameters and constraints."""

    def parse_initial_conditions(conditions):
        valid_conditions = []
        for ic in conditions:
            if all(isinstance(x, (float, int)) for x in ic[:3]) and isinstance(
                ic[3], bool
            ):
                valid_conditions.append(ic)
        return valid_conditions

    # Default initial conditions
    h, h_min, h_max, h_vary = avg_H, None, None, True
    g0, g0_min, g0_max, g0_vary = 0.15, None, None, True
    g1, g1_min, g1_max, g1_vary = 0.15, None, None, True

    if initial_conditions is not None:
        initial_conditions = parse_initial_conditions(initial_conditions)
        if len(initial_conditions) > 0:
            h, h_min, h_max, h_vary = initial_conditions[0]
        if len(initial_conditions) > 1:
            g0, g0_min, g0_max, g0_vary = initial_conditions[1]
        if model == "HG1G2" and len(initial_conditions) > 2:
            g1, g1_min, g1_max, g1_vary = initial_conditions[2]

    params = Parameters()
    if model == "HG":
        params.add_many(
            ("H", h, h_vary, h_min, h_max), ("G", g0, g0_vary, g0_min, g0_max)
        )
        params.add("constraint", expr="1-G")
        return params
    elif model == "HG12":
        params.add_many(
            ("H", h, h_vary, h_min, h_max),
            ("G12", g0, g0_vary, g0_min, g0_max),
        )
        params.add("Constraint1", expr="G12 + 0.0818919")
        params.add("Constraint2", expr="0.2 - G12")

        params2 = Parameters()
        params2.add_many(
            ("H", h, h_vary, h_min, h_max),
            ("G12", g0, g0_vary, g0_min, g0_max),
        )
        params2.add("constraint_1", expr="G12 - 0.2")
        params2.add("constraint_2", expr="0.909714 - G12")
        return [params, params2]
    elif model == "HG12PEN":
        params.add_many(
            ("H", h, h_vary, h_min, h_max),
            ("G12", g0, g0_vary, g0_min, g0_max),
        )
        params.add("constraint", expr="1-G12")
        return params
    elif model == "HG1G2":
        params.add_many(
            ("H", h, h_vary, h_min, h_max),
            ("G1", g0, g0_vary, g0_min, g0_max),
            ("G2", g1, g1_vary, g1_min, g1_max),
        )
        params.add("constraint", expr="1-G1-G2")
        return params
    elif model == "LINEAR":
        params.add_many(
            ("H", h, h_vary, h_min, h_max),
            ("beta", g0, g0_vary, g0_min, g0_max),
        )
        return params
    return None


def _fit_model(
    angle: np.ndarray,
    magnitude: np.ndarray,
    model: str,
    method: str,
    initial_conditions: Optional[List[List]] = None,
) -> ModelResult:
    """Performs the least-squares fitting of a phase curve model to data."""
    if method not in FITTING_METHODS:
        raise ValueError(
            f"Invalid fitting method. "
            f"Choose from: {', '.join(FITTING_METHODS)}"
        )

    # Get the number of parameters and constraints for the given model
    n = NUM_PARAMETERS.get(model, 0)
    c = NUM_CONSTRAINTS.get(model, 0)

    # Check for method compatibility with constraints
    if c > 0 and method not in ["Cobyla", "SLSQP", "trust-constr"]:
        raise ValueError(
            f"The '{model}' model has constraints (c={c}), "
            f"but the selected method "
            f"'{method}' does not support them. Please use a"
            f" compatible method."
        )

    # Ensure there are enough data points
    if len(angle) < (n + 1 + c):
        raise ValueError(
            f"Not enough data points. At least {n + 1 + c} are required."
        )

    mean_mag = np.mean(magnitude)
    params = _fitting_model_parameter_object(
        model, mean_mag, initial_conditions
    )

    if isinstance(params, list):
        # Special case for HG12 model with two sets of constraints
        res1 = minimize(
            _fit_residual,
            params[0],
            args=(angle, magnitude, model, PARAMETER_KEYS[model]),
            method=method,
        )
        res2 = minimize(
            _fit_residual,
            params[1],
            args=(angle, magnitude, model, PARAMETER_KEYS[model]),
            method=method,
        )
        # Use sum of squares of residuals for comparison
        res1_sum_sq = np.sum(res1.residual**2)
        res2_sum_sq = np.sum(res2.residual**2)
        result = res1 if res1_sum_sq < res2_sum_sq else res2
    else:
        result = minimize(
            _fit_residual,
            params,
            args=(angle, magnitude, model, PARAMETER_KEYS[model]),
            method=method,
        )

    if not result.success:
        raise RuntimeError(f"Fitting failed: {result.message}")

    return result


class PhaseCurve:
    """
    A class to represent and analyze asteroid phase curves.

    Parameters
    ----------
    angle : Union[float, List[float], np.ndarray]
        The phase angle(s) in degrees.
    magnitude : Optional[Union[float, List[float], np.ndarray]]
        The magnitude(s) corresponding to the angles.
    magnitude_unc : Optional[Union[float, List[float], np.ndarray]]
        The uncertainty of the magnitudes.
    H, G, G12, G1, G2, beta : Optional[float]
        Model parameters.
    fitting_status : Optional[bool]
        Status of the last fit.
    fitting_model : Optional[str]
        Name of the last fitted model.
    fitting_method : Optional[str]
        Method used for the last fit.
    fit_residual : Optional[List[float]]
        Residuals from the last fit.
    montecarlo_uncertainty : Optional[Dict]
        Results from the last Monte Carlo simulation.
    """

    def __init__(
        self,
        angle: Union[float, List[float], np.ndarray],
        magnitude: Optional[Union[float, List[float], np.ndarray]] = None,
        magnitude_unc: Optional[Union[float, List[float], np.ndarray]] = None,
        H: Optional[float] = None,
        G: Optional[float] = None,
        G12: Optional[float] = None,
        G1: Optional[float] = None,
        G2: Optional[float] = None,
        beta: Optional[float] = None,
        fitting_status: Optional[bool] = False,
        fitting_model: Optional[str] = None,
        fitting_method: Optional[str] = None,
        fit_residual: Optional[List[float]] = None,
        montecarlo_uncertainty: Optional[Dict] = None,
    ) -> None:
        self.angle = (
            np.asarray(angle)
            if isinstance(angle, (list, np.ndarray))
            else angle
        )
        self.magnitude = (
            np.asarray(magnitude)
            if isinstance(magnitude, (list, np.ndarray))
            else magnitude
        )

        if isinstance(self.angle, np.ndarray) and isinstance(
            self.magnitude, np.ndarray
        ):
            if self.angle.shape != self.magnitude.shape:
                raise ValueError(
                    "`angle` and `magnitude` must have the same shape."
                )
            if np.isnan(self.angle).any() or np.isinf(self.angle).any():
                raise ValueError("`angle` array contains NaN or Inf values.")
            if (
                np.isnan(self.magnitude).any()
                or np.isinf(self.magnitude).any()
            ):
                raise ValueError(
                    "`magnitude` array contains NaN or Inf values."
                )

        # Store parameters in a dictionary for a more flexible design
        self.params = {}
        if H is not None:
            self.params["H"] = H
        if G is not None:
            self.params["G"] = G
        if G12 is not None:
            self.params["G12"] = G12
        if G1 is not None:
            self.params["G1"] = G1
        if G2 is not None:
            self.params["G2"] = G2
        if beta is not None:
            self.params["beta"] = beta

        self.magnitude_unc = (
            np.asarray(magnitude_unc)
            if isinstance(magnitude_unc, (list, np.ndarray))
            else magnitude_unc
        )
        self.fitting_status = fitting_status
        self.fitting_model = fitting_model
        self.fitting_method = fitting_method
        self.fit_residual = fit_residual
        self.montecarlo_uncertainty = montecarlo_uncertainty

    def generateModel(
        self,
        model: str,
        degrees: Optional[Union[float, List[float], np.ndarray]] = None,
    ) -> Union[float, List[float], np.ndarray]:
        """
        Generates magnitude values based on a specified phase curve model
        and the instance's stored parameters.

        Parameters
        ----------
        model : str
            The name of the phase curve model to use (e.g., "HG", "HG12").
        degrees : Optional[Union[float, List[float], np.ndarray]]
            The phase angle(s) for which to generate the model values. If None,
            the instance's own `angle` data will be used.

        Returns
        -------
        Union[float, List[float], np.ndarray]
            The calculated magnitudes for the given angles.

        Raises
        ------
        ValueError
            If an invalid model is specified or if model parameters are not
            instantiated in the class.
        """
        model = model.upper()
        if model not in ALLOWED_PHASE_CURVE_MODELS:
            raise ValueError(
                f"Invalid model '{model}'. Must be one of:"
                f" {', '.join(ALLOWED_PHASE_CURVE_MODELS)}"
            )

        params_to_extract = PARAMETER_KEYS.get(model, [])
        params = [self.params.get(key) for key in params_to_extract]

        if None in params:
            raise ValueError(
                f"One or more parameters for model '{model}' were not set."
            )

        fn = _call_model_function(model)
        degrees_arr = (
            np.asarray(degrees) if degrees is not None else self.angle
        )

        values = fn(degrees_arr, *params)
        return values.tolist() if isinstance(degrees, list) else values

    def fitModel(
        self,
        model: str,
        method: str,
        initial_conditions: Optional[List[List]] = None,
    ) -> ModelResult:
        """
        Fits a phase curve model to the instance's angle and magnitude data.

        Parameters
        ----------
        model : str
            The name of the model to fit (e.g., "HG", "HG12", "LINEAR").
        method : str
            The fitting method to use, must be one of `FITTING_METHODS`.
        initial_conditions : Optional[List[List]]
            A list of initial conditions for the parameters.

        Returns
        -------
        ModelResult
            The result object from the `lmfit.minimize` function.

        Raises
        ------
        ValueError
            If `angle` or `magnitude` are not set.
        RuntimeError
            If the fitting procedure fails.
        """
        if self.angle is None or self.magnitude is None:
            raise ValueError(
                "`angle` and `magnitude` must be set before fitting."
            )

        try:
            fit_result = _fit_model(
                self.angle, self.magnitude, model, method, initial_conditions
            )
            self.fitting_status = True
            self.fitting_model = model
            self.fitting_method = method
            self.fit_residual = fit_result.residual.tolist()

            self.params.clear()
            for key, param in fit_result.params.items():
                self.params[key] = param.value

            if model in ["HG12", "HG12PEN"]:
                self.params["G1"] = (
                    pm.HG12._G12_to_G1(self.params["G12"])
                    if model == "HG12"
                    else pm.HG12_Pen16._G12_to_G1(self.params["G12"])
                )
                self.params["G2"] = (
                    pm.HG12._G12_to_G2(self.params["G12"])
                    if model == "HG12"
                    else pm.HG12_Pen16._G12_to_G2(self.params["G12"])
                )

            return fit_result
        except Exception as e:
            raise RuntimeError(
                f"Fitting procedure failed. Original error: {e}"
            )

    def monteCarloUnknownRotation(
        self,
        n_simulations: int,
        amplitude_variation: float,
        model: str,
        distribution: str = "sinusoidal",
        method: str = "Cobyla",
        n_threads: int = 1,
    ) -> Dict:
        """
        Performs a Monte Carlo simulation to estimate uncertainties in
        fitted parameters, specifically for the case of UNKNOWN ROTATION.

        This method simulates rotational light curve variations by adding
        random noise (either uniform or sinusoidal) to the observed
        magnitudes. The randomness represents the unknown rotation of the
        object, which can introduce scatter in the phase curve data. The
        simulation then fits the chosen model to each perturbed dataset to
        determine the distribution of the fitted parameters.

        Parameters
        ----------
        n_simulations : int
            The number of simulations to run.
        amplitude_variation : float
            The amplitude of the random noise added to the magnitudes.
        model : str
            The phase curve model to fit for each simulation.
        distribution : str, optional
            The distribution of the noise ("uniform" or "sinusoidal").
            Defaults to "sinusoidal".
        method : str, optional
            The fitting method to use for each simulation. Defaults to
            "Cobyla".
        n_threads : int, optional
            The number of CPU cores to use for parallel processing.
            Defaults to 1.

        Returns
        -------
        Dict
            A dictionary containing lists of the fitted parameter values from
            each simulation.
        """
        if self.angle is None or self.magnitude is None:
            raise ValueError(
                "`angle` and `magnitude` must be set before simulation."
            )

        if distribution not in ["uniform", "sinusoidal"]:
            raise ValueError(
                "Invalid distribution. Must be 'uniform' or 'sinusoidal'."
            )

        if method not in FITTING_METHODS:
            raise ValueError(
                f"Invalid fitting method. Must be one of:"
                f" {', '.join(FITTING_METHODS)}"
            )

        magnitudes = np.asarray(self.magnitude)
        angles = np.asarray(self.angle)

        simulated_magnitudes = _montecarlo_simulated_magnitudes(
            magnitudes, n_simulations, distribution, amplitude_variation
        )

        all_results = {key: [] for key in PARAMETER_KEYS[model]}

        with Pool(n_threads) as p:
            # Use a generator to avoid creating a large intermediate list
            results_generator = p.starmap(
                _fit_model,
                ((angles, m, model, method) for m in simulated_magnitudes),
            )

            for res in results_generator:
                if res and res.success:
                    for key in PARAMETER_KEYS[model]:
                        all_results[key].append(res.params[key].value)

        # Calculate additional parameters for HG12 and HG12PEN
        if model == "HG12":
            all_results["G1"] = list(
                map(pm.HG12._G12_to_G1, all_results["G12"])
            )
            all_results["G2"] = list(
                map(pm.HG12._G12_to_G2, all_results["G12"])
            )
        elif model == "HG12PEN":
            all_results["G1"] = list(
                map(pm.HG12_Pen16._G12_to_G1, all_results["G12"])
            )
            all_results["G2"] = list(
                map(pm.HG12_Pen16._G12_to_G2, all_results["G12"])
            )

        self.montecarlo_uncertainty = all_results
        return all_results

    def toJSON(self) -> str:
        """
        Serializes the PhaseCurve object to a JSON string.
        """
        data = self.__dict__.copy()
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
        return json.dumps(data)

    @classmethod
    def fromJSON(cls, json_str: str) -> "PhaseCurve":
        """
        Creates a PhaseCurve instance from a JSON string.
        """
        data = json.loads(json_str)
        return cls(**data)
