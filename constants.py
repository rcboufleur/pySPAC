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

NUM_PARAMETERS = {
    "HG": 2,
    "HG12": 2,
    "HG12PEN": 2,
    "HG1G2": 3,
    "LINEAR": 2,
}

NUM_CONSTRAINTS = {
    "HG": 1,
    "HG12": 2,
    "HG12PEN": 1,
    "HG1G2": 1,
    "LINEAR": 0,
}
