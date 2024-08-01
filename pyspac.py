import json
from multiprocessing import Pool
from typing import Dict, List, Optional, Union, Callable
import math
import numpy as np
import sbpy.photometry as pm
from lmfit import Parameters, minimize

# Constants
ALLOWED_PHASE_CURVE_MODELS = ["HG", "HG12", "HG12PEN", "HG1G2", "LINEAR"]
FITTING_MODELS = ["HG1G2", "HG", "HG12", "HG12PEN", "LINEAR"]
FITTING_METHODS = ["Cobyla", "SLSQP", "trust-constr"]
PARAMETER_KEYS = {
    "HG": ["H", "G"],
    "HG12": ["H", "G12"],
    "HG12PEN": ["H", "G12"],
    "HG1G2": ["H", "G1", "G2"],
    "LINEAR": ["H", "beta"],
}


# Model Functions
def HG(angle: float, H: float, G: float) -> float:
    return pm.HG.evaluate(np.deg2rad(angle), H, G)


def HG12(angle: float, H: float, G12: float) -> float:
    return pm.HG12.evaluate(np.deg2rad(angle), H, G12)


def HG12PEN(angle: float, H: float, G12: float) -> float:
    return pm.HG12_Pen16.evaluate(np.deg2rad(angle), H, G12)


def HG1G2(angle: float, H: float, G1: float, G2: float) -> float:
    return pm.HG1G2.evaluate(np.deg2rad(angle), H, G1, G2)


def HBetaLinear(angle: float, H: float, beta: float) -> float:
    return angle * beta + H


def call_model_function(model: str) -> Callable:
    function_dict = {
        "HG": HG,
        "HG12": HG12,
        "HG12PEN": HG12PEN,
        "HG1G2": HG1G2,
        "LINEAR": HBetaLinear,
    }
    return function_dict[model]


def get_model_class_parameters(
    model: str, instance: "PhaseCurve"
) -> List[float]:
    parameters_list = {
        "HG": [instance.H, instance.G],
        "HG12": [instance.H, instance.G12],
        "HG12PEN": [instance.H, instance.G12],
        "HG1G2": [instance.H, instance.G1, instance.G2],
        "LINEAR": [instance.H, instance.beta],
    }
    return parameters_list[model]


def fitting_model_parameter_object(
    model: str, avg_H: float, initial_conditions: Optional[List] = None
) -> Union[Parameters, List[Parameters]]:
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
        h, h_min, h_max, h_vary = initial_conditions[0]
        g0, g0_min, g0_max, g0_vary = initial_conditions[1]
        if model in ["HG1G2"]:
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


def fit_residual(
    params: Parameters,
    angles: np.ndarray,
    magnitudes: np.ndarray,
    model: str,
    param_keys: List[str],
) -> np.ndarray:
    parameters = [params[key].value for key in param_keys]
    fn = call_model_function(model)
    model_values = fn(angles, *parameters)
    return magnitudes - model_values


def fit_model(
    angle: Union[List[float], np.ndarray],
    magnitude: Union[List[float], np.ndarray],
    model: str,
    method: str,
    initial_conditions: Optional[List[float]] = None,
) -> Dict:
    if method not in FITTING_METHODS:
        raise ValueError(
            f"Invalid fitting method. Choose from:"
            f"{', '.join(FITTING_METHODS)}"
        )

    angles = np.array(angle)
    magnitudes = np.array(magnitude)
    mean_mag = np.mean(magnitudes)

    # Number of parameters for each model
    num_parameters = {
        "HG": 2,
        "HG12": 2,
        "HG12PEN": 2,
        "HG1G2": 3,
        "LINEAR": 2,
    }

    # Number of constraints for each model
    num_constraints = {
        "HG": 1,
        "HG12": 2,
        "HG12PEN": 1,
        "HG1G2": 1,
        "LINEAR": 0,
    }

    # Get the number of parameters and constraints for the given model
    n = num_parameters.get(model, 0)
    c = num_constraints.get(model, 0)

    # Ensure there are enough data points
    if len(angles) < (n + 1 + c):
        raise ValueError(
            f"Not enough data points. At least {n + 1 + c} are required."
        )

    params = fitting_model_parameter_object(
        model, mean_mag, initial_conditions
    )
    if isinstance(params, list):
        res1 = minimize(
            fit_residual,
            params[0],
            args=(angles, magnitudes, model, PARAMETER_KEYS[model]),
            method=method,
        )
        res2 = minimize(
            fit_residual,
            params[1],
            args=(angles, magnitudes, model, PARAMETER_KEYS[model]),
            method=method,
        )
        # Use sum of squares of residuals for comparison
        res1_sum_sq = np.sum(res1.residual**2)
        res2_sum_sq = np.sum(res2.residual**2)
        result = res1 if res1_sum_sq < res2_sum_sq else res2
    else:
        result = minimize(
            fit_residual,
            params,
            args=(angles, magnitudes, model, PARAMETER_KEYS[model]),
            method=method,
        )

    return result


def montecarlo_simulated_magnitudes(
    magnitudes: Union[List[float], np.ndarray],
    n_simulations: int,
    distribution: str,
    amplitude_variation: float,
) -> np.ndarray:
    magnitudes = np.array(magnitudes)
    if distribution not in ["uniform", "sinusoidal"]:
        raise ValueError(
            "Invalid distribution. Must be 'uniform' or 'sinusoidal'."
        )

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


class PhaseCurve:
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
        montecarlo_unknownRotation: Optional[List[float]] = None,
    ) -> None:
        if isinstance(magnitude, (float, list, np.ndarray)) and type(
            angle
        ) != type(magnitude):
            raise ValueError(
                "<<angle>> and <<magnitude>> must have the same type."
            )
        elif isinstance(angle, (list, np.ndarray)) and isinstance(
            magnitude, (list, np.ndarray)
        ):
            if len(angle) != len(magnitude):
                raise ValueError(
                    "<<angle>> and <<magnitude>> must have the same "
                    "length when they are lists or arrays."
                )

        self.angle = angle if isinstance(angle, float) else list(angle)
        self.magnitude = (
            magnitude if isinstance(magnitude, float) else list(magnitude)
        )
        self.magnitude_unc = (
            magnitude_unc
            if isinstance(magnitude_unc, float)
            else list(magnitude_unc)
        )
        self.H = H
        self.G = G
        self.G12 = G12
        self.G1 = G1
        self.G2 = G2
        self.beta = beta
        self.fitting_status = fitting_status
        self.fitting_model = fitting_model
        self.fitting_method = fitting_method
        self.fit_residual = fit_residual
        montecarlo_unknownRotation = montecarlo_unknownRotation

    def generateModel(
        self,
        model: str,
        degrees: Optional[Union[float, List[float], np.ndarray]] = None,
    ) -> Union[float, List[float], np.ndarray]:
        model = model.upper()
        if model not in ALLOWED_PHASE_CURVE_MODELS:
            raise ValueError(
                f"Invalid parameter. '{model}' must be "
                "one of: {', '.join(ALLOWED_PHASE_CURVE_MODELS)}"
            )

        fn = call_model_function(model)
        params = get_model_class_parameters(model, self)
        if None in params:
            raise ValueError(
                "One or more model parameters were not "
                "instantiated in the class [H, G, G12, ...]."
            )

        degrees = degrees if degrees is not None else self.angle
        angle_type_parsed = (
            np.array(degrees) if isinstance(degrees, list) else degrees
        )
        values = fn(angle_type_parsed, *params)
        values_type_parsed = (
            list(values) if isinstance(degrees, list) else values
        )

        return values_type_parsed

    def fitModel(
        self,
        model: str,
        method: str,
        initial_conditions: Optional[List[float]] = None,
    ) -> Union[Dict, str]:
        if self.angle is None or self.magnitude is None:
            raise ValueError(
                "angle and magnitude must be instantiated "
                "before calling fit_model."
            )

        try:
            fit_result = fit_model(
                self.angle, self.magnitude, model, method, initial_conditions
            )
            self.fitting_status = True
            self.fitting_model = model
            self.fitting_method = method
            self.fit_residual = list(fit_result.residual)

            if model == "HG":
                self.H = fit_result.params["H"].value
                self.G = fit_result.params["G"].value
            elif model == "HG12":
                self.H = fit_result.params["H"].value
                self.G12 = fit_result.params["G12"].value
                self.G1 = pm.HG12._G12_to_G1(fit_result.params["G12"].value)
                self.G2 = pm.HG12._G12_to_G2(fit_result.params["G12"].value)
            elif model == "HG12PEN":
                self.H = fit_result.params["H"].value
                self.G12 = fit_result.params["G12"].value
                self.G1 = pm.HG12_Pen16._G12_to_G1(
                    fit_result.params["G12"].value
                )
                self.G2 = pm.HG12_Pen16._G12_to_G2(
                    fit_result.params["G12"].value
                )
            elif model == "HG1G2":
                self.H = fit_result.params["H"].value
                self.G1 = fit_result.params["G1"].value
                self.G2 = fit_result.params["G2"].value
            elif model == "LINEAR":
                self.H = fit_result.params["H"].value
                self.beta = fit_result.params["beta"].value

            return fit_result

        except Exception as e:
            raise ValueError(f"Fitting procedure failed. {e}")

    def montecarlo_unknownRotation(
        self,
        n_simulations: int,
        amplitude_variation: float,
        model: str,
        distribution: Optional[str] = "sinusoidal",
        method: Optional[str] = "nelder",
        n_threads: Optional[int] = 1,
    ) -> List[float]:
        magnitudes = np.array(self.magnitude)
        angles = np.array(self.angle)

        simulated_magnitudes = montecarlo_simulated_magnitudes(
            magnitudes, n_simulations, distribution, amplitude_variation
        )
        chunk_size = 100
        num_chunks = math.ceil(n_simulations / chunk_size)
        all_results = {}

        with Pool(n_threads) as p:
            for chunk_index in range(num_chunks):
                start_index = chunk_index * chunk_size
                end_index = min((chunk_index + 1) * chunk_size, n_simulations)
                chunk_simulated_magnitudes = simulated_magnitudes[
                    start_index:end_index
                ]
                results = p.starmap(
                    fit_model,
                    [
                        (angles, m, model, method)
                        for m in chunk_simulated_magnitudes
                    ],
                )
                final_results = np.array(
                    [list(r.params.valuesdict().values()) for r in results]
                ).T
                dict_results = {
                    k: list(final_results[i])
                    for i, k in enumerate(results[0].params.keys())
                }
                for key, value in dict_results.items():
                    all_results.setdefault(key, []).extend(value)

        if model == "HG12":
            all_results["G1"] = list(
                map(pm.HG12._G12_to_G1, all_results["G12"])
            )
            all_results["G2"] = list(
                map(pm.HG12._G12_to_G2, all_results["G12"])
            )

        if model == "HG12PEN":
            all_results["G1"] = list(
                map(pm.HG12_Pen16._G12_to_G1, all_results["G12"])
            )
            all_results["G2"] = list(
                map(pm.HG12_Pen16._G12_to_G2, all_results["G12"])
            )
        self.montecarlo_unknownRotation = all_results
        return all_results

    def toJSON(self) -> Dict:
        return json.dumps(self.__dict__)

    @staticmethod
    def fromJSON(json_str: str) -> "PhaseCurve":
        data = json.loads(json_str)
        return PhaseCurve(**data)
