import json
from multiprocessing import Pool
from typing import Dict, List, Optional, Union

import numpy as np
import sbpy.photometry as pm
from lmfit import Parameters, minimize
from lmfit.model import ModelResult
from tqdm import tqdm

from .constants import (
    ALLOWED_PHASE_CURVE_MODELS,
    GENERAL_CONSTRAINT_METHODS,
    ALL_FITTING_METHODS,
    MODEL_PROPERTIES,
)


# ==============================================================================
# TOP-LEVEL PRIVATE HELPER FUNCTIONS
# ==============================================================================


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
    uncertainties: Optional[np.ndarray],
    n_simulations: int,
    distribution: str,
    amplitude_variation: float,
) -> np.ndarray:
    """
    Generates simulated magnitude datasets by first simulating observational
    error (if uncertainties are given) and then adding rotational variation.
    """
    base_magnitudes = np.asarray(magnitudes)

    if uncertainties is not None:
        uncertainties = np.asarray(uncertainties)
        base_magnitudes = np.random.normal(
            loc=base_magnitudes,
            scale=uncertainties,
            size=(n_simulations, len(base_magnitudes)),
        )

    if amplitude_variation != 0:
        if distribution == "uniform":
            rotational_variation = (
                np.random.uniform(-1, 1, (n_simulations, len(magnitudes))) * amplitude_variation
            )
        elif distribution == "sinusoidal":
            rotational_variation = (
                np.sin(np.random.uniform(0, 2 * np.pi, (n_simulations, len(magnitudes))))
                * amplitude_variation
            )
        return base_magnitudes + rotational_variation

    return base_magnitudes


def _fit_residual(
    params: Parameters,
    angles: np.ndarray,
    magnitudes: np.ndarray,
    model: str,
    param_keys: List[str],
    uncertainties: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Calculates the (potentially weighted) residuals for the fitting function."""
    parameters = [params[key].value for key in param_keys]
    fn = _call_model_function(model)
    model_values = fn(angles, *parameters)

    residual = magnitudes - model_values

    if uncertainties is not None:
        return residual / uncertainties

    return residual


def _fitting_model_parameter_object(
    model: str, avg_H: float, initial_conditions: Optional[List[List]] = None
) -> Union[Parameters, List[Parameters]]:
    """Sets up the initial parameters, bounds, and constraints."""

    def parse_initial_conditions(conditions):
        valid_conditions = []
        for ic in conditions:
            if all(isinstance(x, (float, int)) for x in ic[:3]) and isinstance(ic[3], bool):
                valid_conditions.append(ic)
        return valid_conditions

    h, h_min, h_max, h_vary = avg_H, None, None, True
    g0, g0_min, g0_max, g0_vary = 0.15, 0.0, 1.0, True
    g1, g1_min, g1_max, g1_vary = 0.15, 0.0, 1.0, True

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
        params.add_many(("H", h, h_vary, h_min, h_max), ("G", g0, g0_vary, g0_min, g0_max))
        params.add("constraint", expr="1-G")
        return params
    elif model == "HG12":
        params.add_many(("H", h, h_vary, h_min, h_max), ("G12", g0, True, None, None))
        params.add("Constraint1", expr="G12 + 0.0818919")
        params.add("Constraint2", expr="0.2 - G12")
        params2 = Parameters()
        params2.add_many(("H", h, h_vary, h_min, h_max), ("G12", g0, True, None, None))
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
        params.add_many(("H", h, h_vary, h_min, h_max), ("beta", g0, g0_vary, 0.0, None))
        return params
    return None


def _fit_model(
    angle: np.ndarray,
    magnitude: np.ndarray,
    model: str,
    method: str,
    initial_conditions: Optional[List[List]] = None,
    uncertainties: Optional[np.ndarray] = None,
) -> ModelResult:
    """Performs the least-squares fitting of a phase curve model to data."""
    if model not in ALLOWED_PHASE_CURVE_MODELS:
        raise ValueError(f"Model '{model}' is not a valid choice.")

    props = MODEL_PROPERTIES[model]

    if props["constrained"]:
        if method not in GENERAL_CONSTRAINT_METHODS:
            raise ValueError(
                f"Model '{model}' is constrained and must use a method from: "
                f"{GENERAL_CONSTRAINT_METHODS}"
            )
    else:
        if method not in ALL_FITTING_METHODS:
            raise ValueError(
                f"Method '{method}' is not a valid choice. " f"Choose from: {ALL_FITTING_METHODS}"
            )

    n_params = props["n_params"]
    if len(angle) < (n_params + 1):
        raise ValueError(f"Not enough data points. At least {n_params + 1} are required.")

    mean_mag = np.mean(magnitude)
    params = _fitting_model_parameter_object(model, mean_mag, initial_conditions)
    fit_args = (angle, magnitude, model, props["params"], uncertainties)

    if isinstance(params, list):
        res1 = minimize(
            _fit_residual,
            params[0],
            args=fit_args,
            method=method,
            calc_covar=True,
        )
        res2 = minimize(
            _fit_residual,
            params[1],
            args=fit_args,
            method=method,
            calc_covar=True,
        )
        res1_sum_sq = np.sum(res1.residual**2)
        res2_sum_sq = np.sum(res2.residual**2)
        result = res1 if res1_sum_sq < res2_sum_sq else res2
    else:
        result = minimize(
            _fit_residual,
            params,
            args=fit_args,
            method=method,
            calc_covar=True,
        )

    if not result.success:
        raise RuntimeError(f"Fitting failed: {result.message}")
    return result


def _fit_wrapper(args):
    """
    Helper to unpack arguments for multiprocessing. Includes an exception
    handler to prevent a single failed fit from crashing the entire pool.
    """
    try:
        return _fit_model(*args)
    except Exception:
        return None


# ==============================================================================
# MAIN PhaseCurve CLASS (Final Version)
# ==============================================================================


class PhaseCurve:
    """A class to represent and analyze asteroid phase curves."""

    def __init__(
        self,
        angle: Union[float, List[float], np.ndarray],
        magnitude: Optional[Union[float, List[float], np.ndarray]] = None,
        magnitude_unc: Optional[Union[float, List[float], np.ndarray]] = None,
        params: Optional[Dict] = None,
        H: Optional[float] = None,
        G: Optional[float] = None,
        G12: Optional[float] = None,
        G1: Optional[float] = None,
        G2: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> None:
        self.angle = np.asarray(angle)
        self.magnitude = np.asarray(magnitude) if magnitude is not None else None
        self.magnitude_unc = np.asarray(magnitude_unc) if magnitude_unc is not None else None

        if params is not None:
            self.params = params
        else:
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

        self.fitting_status = False
        self.fitting_model = None
        self.fitting_method = None
        self.fit_residual = None
        self.fit_result = None
        self.montecarlo_uncertainty = None
        self.uncertainty_results = None
        self.uncertainty_source = None

        if self.magnitude is not None:
            if self.angle.shape != self.magnitude.shape:
                raise ValueError("`angle` and `magnitude` must have the same shape.")
            if np.isnan(self.angle).any() or np.isinf(self.angle).any():
                raise ValueError("`angle` array contains NaN or Inf values.")
            if np.isnan(self.magnitude).any() or np.isinf(self.magnitude).any():
                raise ValueError("`magnitude` array contains NaN or Inf values.")

    def __str__(self) -> str:
        """Returns a concise string representation of the PhaseCurve object."""
        n_points = self.angle.size
        if self.fitting_status:
            fit_info = f"Fitted with '{self.fitting_model}' using '{self.fitting_method}'"
        else:
            fit_info = "Not yet fitted"
        return f"<PhaseCurve object with {n_points} data points | {fit_info}>"

    def _clear_fit_results(self):
        """Resets all fitting and uncertainty results to their initial state."""
        self.fitting_status = False
        self.fitting_model = None
        self.fitting_method = None
        self.fit_residual = None
        self.fit_result = None
        self.montecarlo_uncertainty = None
        self.uncertainty_results = None
        self.uncertainty_source = None
        initial_keys = ["H", "G", "G12", "G1", "G2", "beta"]
        self.params = {k: v for k, v in self.params.items() if k in initial_keys}

    def summary(self) -> None:
        """Prints a detailed summary of the phase curve data and fit results."""
        n_points = self.angle.size
        print("================== PhaseCurve Summary ==================")
        print(f"Data Points:         {n_points}")
        if n_points > 1 and self.magnitude is not None:
            print(f"Angle Range:         {np.min(self.angle):.2f}° to {np.max(self.angle):.2f}°")
            print(
                f"Magnitude Range:     {np.min(self.magnitude):.2f} to {np.max(self.magnitude):.2f}"
            )
        print("------------------------------------------------------")

        if self.fitting_status:
            print("Fitting Status:      SUCCESS")
            print(f"Model:               {self.fitting_model}")
            print(f"Method:              {self.fitting_method}")
            if self.fit_result:
                rms = np.sqrt(self.fit_result.redchi)
                print(f"Fit RMS:             {rms:.4f}")
        else:
            print("Fitting Status:      NOT YET FITTED")

        if self.params:
            print("------------------- Model Parameters -------------------")
            # Check if lmfit errors are available to print the header
            if self.fit_result and any(
                p.stderr is not None for p in self.fit_result.params.values()
            ):
                print("(Errors are lmfit's 1-sigma estimates from covariance matrix)")

            # Always iterate over self.params to show all parameters
            for param_name, param_value in self.params.items():
                if param_name.lower().startswith("constraint"):
                    continue

                display_name = param_name
                # Check for the (derived) tag and format the name
                if self.fitting_model in [
                    "HG12",
                    "HG12PEN",
                ] and param_name in ["G1", "G2"]:
                    display_name = f"{param_name} (derived)"

                # Start with the basic value display
                display_value = f"{param_value:.4f}"

                # If the full fit result exists, try to add the standard error
                if self.fit_result and param_name in self.fit_result.params:
                    stderr = self.fit_result.params[param_name].stderr
                    if stderr is not None:
                        display_value += f" +/- {stderr:.4f}"

                print(f"  {display_name:<20} {display_value}")

        if self.uncertainty_results:
            print("--------------- Monte Carlo Uncertainties --------------")
            if self.uncertainty_source:
                print(f"Source:              {self.uncertainty_source}")
            p_info = self.uncertainty_results[next(iter(self.uncertainty_results))]["percentiles"]
            print(f"(Median and {p_info[0]:.2f}% / {p_info[2]:.2f}% percentile errors)")
            for param, data in self.uncertainty_results.items():
                median = data["median"]
                upper_err = data["upper_error"]
                lower_err = data["lower_error"]
                print(f"  {param:<18} {median:.4f} (+{upper_err:.4f} / {lower_err:.4f})")

        print("======================================================")

    def _estimate_uncertainty_from_rms(self, model: str, method: str) -> np.ndarray:
        """
        Performs a preliminary fit to estimate the uncertainty from the
        Root Mean Square (RMS) of the residuals.
        """
        print(
            "Note: Magnitude uncertainties not provided. "
            "Estimating a uniform uncertainty from the RMS of the model residuals."
        )
        fit_result = self.fitModel(model=model, method=method)
        residuals = fit_result.residual
        degrees_of_freedom = len(self.magnitude) - len(fit_result.params)
        if degrees_of_freedom <= 0:
            raise ValueError(
                "Cannot estimate RMS: Degrees of freedom is non-positive. "
                "Not enough data points for the given model."
            )
        rms_error = np.sqrt(np.sum(residuals**2) / degrees_of_freedom)
        return np.full_like(self.magnitude, fill_value=rms_error)

    def generateModel(
        self,
        model: str,
        degrees: Optional[Union[float, List, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """Generates magnitude values based on a specified phase curve model."""
        model = model.upper()
        if model not in ALLOWED_PHASE_CURVE_MODELS:
            raise ValueError(
                f"Invalid model '{model}'. Must be one of: {', '.join(ALLOWED_PHASE_CURVE_MODELS)}"
            )
        params_to_extract = MODEL_PROPERTIES[model]["params"]
        params = [self.params.get(key) for key in params_to_extract]
        if None in params:
            raise ValueError(f"One or more parameters for model '{model}' were not set.")
        fn = _call_model_function(model)
        degrees_arr = np.asarray(degrees) if degrees is not None else self.angle
        return fn(degrees_arr, *params)

    def fitModel(
        self,
        model: str,
        method: str,
        initial_conditions: Optional[List[List]] = None,
    ) -> ModelResult:
        """Fits a phase curve model to the instance's angle and magnitude data."""
        self._clear_fit_results()
        if self.angle is None or self.magnitude is None:
            raise ValueError("`angle` and `magnitude` must be set before fitting.")
        try:
            fit_result = _fit_model(
                self.angle,
                self.magnitude,
                model,
                method,
                initial_conditions,
                uncertainties=self.magnitude_unc,
            )
            self.fitting_status = True
            self.fitting_model = model
            self.fitting_method = method
            self.fit_result = fit_result
            self.fit_residual = fit_result.residual.tolist()
            self.params.clear()
            for key, param in fit_result.params.items():
                self.params[key] = param.value
            if model in ["HG12", "HG12PEN"]:
                g12_val = self.params["G12"]
                g1_func = pm.HG12._G12_to_G1 if model == "HG12" else pm.HG12_Pen16._G12_to_G1
                g2_func = pm.HG12._G12_to_G2 if model == "HG12" else pm.HG12_Pen16._G12_to_G2
                self.params["G1"] = g1_func(g12_val)
                self.params["G2"] = g2_func(g12_val)
            return fit_result
        except Exception as e:
            raise RuntimeError(f"Fitting procedure failed. Original error: {e}")

    def monteCarloUnknownRotation(
        self,
        n_simulations: int,
        amplitude_variation: float,
        model: str,
        distribution: str = "sinusoidal",
        method: str = "Cobyla",
        n_threads: int = 1,
        verbose: bool = True,
    ) -> Dict:
        """
        Performs a Monte Carlo simulation including rotational variation to estimate uncertainties.
        If observational errors are not provided, they are estimated from RMS.
        """
        if self.magnitude_unc is None:
            local_uncertainties = self._estimate_uncertainty_from_rms(model, method)
            self.uncertainty_source = "Estimated RMS + Unknown Rotation"
        else:
            local_uncertainties = self.magnitude_unc
            self.uncertainty_source = "Observational Error + Unknown Rotation"
        magnitudes = np.asarray(self.magnitude)
        angles = np.asarray(self.angle)
        all_results = {key: [] for key in MODEL_PROPERTIES[model]["params"]}
        progress_bar = tqdm(total=n_simulations, desc="Rotation Simulation") if verbose else None
        attempts, max_attempts = 0, n_simulations * 10
        try:
            with Pool(n_threads) as p:
                while len(all_results["H"]) < n_simulations:
                    if attempts > max_attempts:
                        raise RuntimeError(f"Failed after {max_attempts} attempts.")
                    needed = n_simulations - len(all_results["H"])
                    batch_size = max(int(needed * 1.25), n_threads)
                    simulated_batch = _montecarlo_simulated_magnitudes(
                        magnitudes,
                        local_uncertainties,
                        batch_size,
                        distribution,
                        amplitude_variation,
                    )
                    arg_iterable = (
                        (angles, m, model, method, None, local_uncertainties)
                        for m in simulated_batch
                    )
                    results_iterator = p.imap_unordered(_fit_wrapper, arg_iterable)
                    for res in results_iterator:
                        attempts += 1
                        if res and res.success and len(all_results["H"]) < n_simulations:
                            for key in MODEL_PROPERTIES[model]["params"]:
                                all_results[key].append(res.params[key].value)
                            if progress_bar:
                                progress_bar.update(1)
        finally:
            if progress_bar:
                progress_bar.n = min(progress_bar.n, progress_bar.total)
                progress_bar.refresh()
                progress_bar.close()
        if model in ["HG12", "HG12PEN"]:
            g12_results = all_results.get("G12", [])
            g1_func = pm.HG12._G12_to_G1 if model == "HG12" else pm.HG12_Pen16._G12_to_G1
            g2_func = pm.HG12._G12_to_G2 if model == "HG12" else pm.HG12_Pen16._G12_to_G2
            all_results["G1"] = list(map(g1_func, g12_results))
            all_results["G2"] = list(map(g2_func, g12_results))
        self.montecarlo_uncertainty = all_results
        self.calculate_uncertainties()
        return all_results

    def monteCarloUncertainty(
        self,
        n_simulations: int,
        model: str,
        method: str,
        n_threads: int = 1,
        verbose: bool = True,
    ) -> Dict:
        """
        Performs a Monte Carlo simulation based only on observational errors to estimate uncertainties.
        If errors are not provided, they are estimated from the RMS of the residuals.
        """
        if self.magnitude_unc is None:
            local_uncertainties = self._estimate_uncertainty_from_rms(model, method)
            self.uncertainty_source = "Estimated from RMS"
        else:
            local_uncertainties = self.magnitude_unc
            self.uncertainty_source = "Observational Error"
        magnitudes = np.asarray(self.magnitude)
        angles = np.asarray(self.angle)
        all_results = {key: [] for key in MODEL_PROPERTIES[model]["params"]}
        progress_bar = tqdm(total=n_simulations, desc="Uncertainty Simulation") if verbose else None
        attempts, max_attempts = 0, n_simulations * 10
        try:
            with Pool(n_threads) as p:
                while len(all_results["H"]) < n_simulations:
                    if attempts > max_attempts:
                        raise RuntimeError(f"Failed after {max_attempts} attempts.")
                    needed = n_simulations - len(all_results["H"])
                    batch_size = max(int(needed * 1.25), n_threads)
                    simulated_batch = _montecarlo_simulated_magnitudes(
                        magnitudes,
                        local_uncertainties,
                        batch_size,
                        "sinusoidal",
                        0,
                    )
                    arg_iterable = (
                        (angles, m, model, method, None, local_uncertainties)
                        for m in simulated_batch
                    )
                    results_iterator = p.imap_unordered(_fit_wrapper, arg_iterable)
                    for res in results_iterator:
                        attempts += 1
                        if res and res.success and len(all_results["H"]) < n_simulations:
                            for key in MODEL_PROPERTIES[model]["params"]:
                                all_results[key].append(res.params[key].value)
                            if progress_bar:
                                progress_bar.update(1)
        finally:
            if progress_bar:
                progress_bar.n = min(progress_bar.n, progress_bar.total)
                progress_bar.refresh()
                progress_bar.close()
        if model in ["HG12", "HG12PEN"]:
            g12_results = all_results.get("G12", [])
            g1_func = pm.HG12._G12_to_G1 if model == "HG12" else pm.HG12_Pen16._G12_to_G1
            g2_func = pm.HG12._G12_to_G2 if model == "HG12" else pm.HG12_Pen16._G12_to_G2
            all_results["G1"] = list(map(g1_func, g12_results))
            all_results["G2"] = list(map(g2_func, g12_results))
        self.montecarlo_uncertainty = all_results
        self.calculate_uncertainties()
        return all_results

    def calculate_uncertainties(self, percentiles: List[float] = [15.87, 50, 84.13]) -> None:
        """
        Calculates the median and asymmetric 1-sigma equivalent errors from the
        Monte Carlo simulation results using percentiles.
        """
        if self.montecarlo_uncertainty is None:
            raise ValueError("Monte Carlo simulation has not been run yet.")
        if len(percentiles) != 3 or not all(0 <= p <= 100 for p in percentiles):
            raise ValueError("`percentiles` must be a list of 3 values between 0 and 100.")
        percentiles.sort()
        lower_p, median_p, upper_p = percentiles
        self.uncertainty_results = {}
        for param, values in self.montecarlo_uncertainty.items():
            if not values:
                continue
            p_values = np.percentile(values, [lower_p, median_p, upper_p])
            lower_val, median_val, upper_val = p_values
            self.uncertainty_results[param] = {
                "median": median_val,
                "lower_error": lower_val - median_val,
                "upper_error": upper_val - median_val,
                "percentiles": percentiles,
            }

    def toJSON(self) -> str:
        """Serializes the PhaseCurve object to a JSON string."""
        data = self.__dict__.copy()
        data.pop("fit_result", None)
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
        return json.dumps(data)

    @classmethod
    def fromJSON(cls, json_str: str) -> "PhaseCurve":
        """Creates a PhaseCurve instance from a JSON string."""
        data = json.loads(json_str)
        init_keys = {
            "angle",
            "magnitude",
            "magnitude_unc",
            "params",
            "H",
            "G",
            "G12",
            "G1",
            "G2",
            "beta",
        }
        init_kwargs = {key: value for key, value in data.items() if key in init_keys}
        new_instance = cls(**init_kwargs)
        for key, value in data.items():
            if key not in init_keys:
                setattr(new_instance, key, value)
        return new_instance
