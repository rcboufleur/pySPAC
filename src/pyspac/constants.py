# constants.py

ALLOWED_PHASE_CURVE_MODELS = ["HG", "HG12", "HG12PEN", "HG1G2", "LINEAR"]

# A strict list of methods that can handle general mathematical constraints
GENERAL_CONSTRAINT_METHODS = ["SLSQP", "COBYLA", "trust-constr"]

# A comprehensive list of all recommended fitting methods available in lmfit
ALL_FITTING_METHODS = sorted(
    [
        "SLSQP",
        "COBYLA",
        "trust-constr",
        "L-BFGS-B",
        "TNC",
        "Powell",
        "least_squares",
        "BFGS",
        "CG",
        "Nelder-Mead",
        "leastsq",
    ]
)

# Updated properties dictionary with a simple boolean flag
MODEL_PROPERTIES = {
    "HG": {
        "params": ["H", "G"],
        "n_params": 2,
        "constrained": True,  # Has a general constraint
    },
    "HG12": {
        "params": ["H", "G12"],
        "n_params": 2,
        "constrained": True,  # Has general constraints
    },
    "HG12PEN": {
        "params": ["H", "G12"],
        "n_params": 2,
        "constrained": True,  # Has a general constraint
    },
    "HG1G2": {
        "params": ["H", "G1", "G2"],
        "n_params": 3,
        "constrained": True,  # Has a general constraint
    },
    "LINEAR": {
        "params": ["H", "beta"],
        "n_params": 2,
        "constrained": False,  # Only has a simple bound, treat as flexible
    },
}
