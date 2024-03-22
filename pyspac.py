from math import pi
from typing import List, Optional, Union

import numpy as np
import sbpy.photometry as pm
from lmfit import Parameters, minimize, report_fit

allowedPhaseCurveModels = ["HG", "HG12", "HG12PEN", "HG1G2", "LINEAR"]
fittingModels = ["HG1G2", "HG", "HG12", "HG12PEN", "LINEAR"]
fittingMethods = ["Cobyla"]
parameter_keys = {
    "HG": ["H", "G"],
    "HG12": ["H", "G12"],
    "HG12PEN": ["H", "G12"],
    "HG1G2": ["H", "G1", "G2"],
    "HBetaLinear": ["H", "beta"],
}


def degree_to_radians(angle: float) -> float:
    return angle * pi / 180


def HG(angle: float, H: float, G: float) -> float:
    return pm.HG.evaluate(degree_to_radians(angle), H, G)


def HG12(angle: float, H: float, G12: float) -> float:
    return pm.HG12.evaluate(degree_to_radians(angle), H, G12)


def HG12PEN(angle: float, H: float, G12: float) -> float:
    return pm.HG12_Pen16.evaluate(degree_to_radians(angle), H, G12)


def HG1G2(angle: float, H: float, G1: float, G2: float) -> float:
    return pm.HG1G2.evaluate(degree_to_radians(angle), H, G1, G2)


def HBetaLinear(angle: float, H: float, beta: float) -> float:
    return angle * beta + H


# Callable typing was ignored on the return because
# one case HG1G2 has more input parameters (4)...
def call_model_function(model: str):
    function_dict = {
        "HG": HG,
        "HG12": HG12,
        "HG12PEN": HG12PEN,
        "HG1G2": HG1G2,
        "LINEAR": HBetaLinear,
    }
    return function_dict[model]


# Instance is not defining type... revisit later...
def get_model_class_parameters(model: str, instance: "PhaseCurve") -> List[float]:
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

    # general initial conditions
    h, h_min, h_max, h_vary = avg_H, avg_H - 2, avg_H + 2, True
    g, g_min, g_max, g_vary = 0.05, 0, 1, True
    g2, g2_min, g2_max, g2_vary = 0.1, 0, 1, True

    valid_initial_conditions = []
    if initial_conditions is not None:
        for ic in initial_conditions:
            valid_initial_conditions.append(
                all(isinstance(x, (float, int)) for x in ic[:3])
                and isinstance(ic[3], bool)
            )

        if model in fittingModels[1:]:
            h, h_min, h_max, h_vary = initial_conditions[0]
            g, g_min, g_max, g_vary = initial_conditions[1]
        elif model in fittingModels[0]:
            h, h_min, h_max, h_vary = initial_conditions[0]
            g, g_min, g_max, g_vary = initial_conditions[1]
            g2, g2_min, g2_max, g2_vary = initial_conditions[2]

    if model == "HG":
        params = Parameters()
        params.add_many(("H", h, h_vary, h_min, h_max), ("G", g, g_vary, g_min, g_max))
        params.add("constraint", expr="1-G")
        return params

    elif model == "HG12":
        params = Parameters()
        params.add_many(
            ("H", h, h_vary, h_min, h_max), ("G12", g, g_vary, g_min, g_max)
        )
        params.add("G1", expr="G12 + 0.0818919")
        params.add("G2", expr="0.2 - G1")

        params2 = Parameters()
        params2.add_many(
            ("H", h, h_vary, h_min, h_max), ("G12", g, g_vary, g_min, g_max)
        )
        params2.add("G1", expr="G12 - 0.2")
        params2.add("G2", expr="0.909714 - G1")

        return [params, params2]

    elif model == "HG12PEN":
        params = Parameters()
        params.add_many(
            ("H", h, h_vary, h_min, h_max), ("G12", g, g_vary, g_min, g_max)
        )
        params.add("constraint", expr="1-G12")
        return params

    elif model == "HG1G2":
        params = Parameters()
        params.add_many(
            ("H", h, h_vary, h_min, h_max),
            ("G1", g, g_vary, g_min, g_max),
            ("G2", g2, g2_vary, g2_min, g2_max),
        )
        params.add("constraint", expr="1-G1-G2")
        return params

    elif model == "LINEAR":
        params = Parameters()
        params.add_many(
            ("H", h, h_vary, h_min, h_max), ("beta", g, g_vary, g_min, g_max)
        )
        params.add("constraint", expr="1-G")
        return params

    else:
        return None


def fit_residual(
    params: Parameters,
    angles: np.ndarray,
    magnitudes: np.ndarray,
    model: str,
    param_keys: List[str],
) -> np.ndarray:
    parameters = [params[key] for key in param_keys]
    fn = call_model_function(model)
    modelo = fn(angles, *parameters)
    return magnitudes - modelo


class PhaseCurve:
    """
    This class represents a phase curve with specified parameters.

    Attributes:
        angle: The angle(s) of the phase curve data (float, list, or np.ndarray).
        magnitude: The magnitude(s) of the phase curve data (float, list, or np.ndarray).
        H: A float value or None.
        G: A float value or None.
        G12: A float value or None.
        G1: A float value or None.
        G2: A float value or None.
    """

    def __init__(
        self,
        angle: Union[float, List[float], np.ndarray],
        magnitude: Optional[Union[float, List[float], np.ndarray]] = None,
        H: Optional[float] = None,
        G: Optional[float] = None,
        G12: Optional[float] = None,
        G1: Optional[float] = None,
        G2: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> None:

        if isinstance(magnitude, (float, list, np.ndarray)) and type(angle) != type(
            magnitude
        ):
            raise ValueError("<<angle>> and <<magnitude>> must have the same type.")
        elif isinstance(angle, (list, np.ndarray)) and isinstance(
            magnitude, (list, np.ndarray)
        ):
            if len(angle) != len(magnitude):
                raise ValueError(
                    "<<angle>> and <<magnitude>> must have the same length when they are lists or arrays."
                )

        self.angle = angle
        self.magnitude = magnitude
        self.H = H
        self.G = G
        self.G12 = G12
        self.G1 = G1
        self.G2 = G2
        self.beta = beta

        # self.fitting_status = False
        # self.fitting_model = ''
        # self.fitting_metod = ''
        # self.fit_report = ''

    def model(
        self,
        model: str,
        degrees: Optional[Union[float, List[float], np.ndarray]] = None,
    ) -> Union[float, List[float], np.ndarray]:
        model = model.upper()
        if model not in allowedPhaseCurveModels:
            raise ValueError(
                f"Invalid parameter. '{model}' must be one of: {', '.join(allowedPhaseCurveModels)}"
            )

        fn, params = call_model_function(model), get_model_class_parameters(model, self)
        if None in params:
            raise ValueError(
                "One or more model parameters were not instantiated in the class [H, G, G12, ...]."
            )

        degrees = degrees if degrees is not None else self.angle
        angle_type_parsed = np.array(degrees) if isinstance(degrees, list) else degrees
        values = fn(angle_type_parsed, *params)
        values_type_parsed = list(values) if isinstance(degrees, list) else values

        return values_type_parsed

    def fit(
        self, model: str, method: str, initial_conditions: Optional[List[float]] = None
    ) -> None:
        angles = np.array(self.angle) if isinstance(self.angle, list) else self.angle
        magnitudes = (
            np.array(self.magnitude)
            if isinstance(self.magnitude, list)
            else self.magnitude
        )
        mean_mag = np.mean(np.array(magnitudes))
        params = fitting_model_parameter_object(model, mean_mag, initial_conditions)
        if isinstance(params, list):
            res1 = minimize(
                fit_residual,
                params[0],
                args=(angles, magnitudes, model, parameter_keys[model]),
                method=method,
            )
            res2 = minimize(
                fit_residual,
                params[1],
                args=(angles, magnitudes, model, parameter_keys[model]),
                method=method,
            )
            if res1.fun < res2.fun:
                result = res1
            else:
                result = res2
        else:
            result = minimize(
                fit_residual,
                params,
                args=(angles, magnitudes, model, parameter_keys[model]),
                method=method,
            )

        print(report_fit(result))

        return None
