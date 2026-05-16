"""
fuzzy/fuzzy_mission.py
-----------------------
Fuzzy Logic System for SpaceRL V3 — SpaceMissionEnv.

Academic context (Tema 5: Razonamiento con Incertidumbre):
    In the base V3 environment, each resource has hard discrete levels:
        0 = critical, 1 = stable, 2 = optimal

    Fuzzy logic replaces this binary categorisation with smooth membership
    functions that better reflect real-world uncertainty. A resource at
    level 1 is not purely "stable" — it may partially belong to "critical"
    and partially to "stable" depending on context.

Membership functions (triangular, universe = [0, 2]):
    LOW    : trimf(0, 0, 1)  — fully LOW at 0, zero at 1
    MEDIUM : trimf(0, 1, 2)  — fully MEDIUM at 1, zero at 0 and 2
    HIGH   : trimf(1, 2, 2)  — fully HIGH at 2, zero at 1

Fuzzy rules (IF-THEN, course Tema 5 format):
    Rule 1: IF energy   is LOW   → RISK is CRITICAL
    Rule 2: IF oxygen   is LOW   → RISK is CRITICAL
    Rule 3: IF hull     is LOW   → RISK is CRITICAL
    Rule 4: IF food     is LOW   OR fuel is LOW → RISK is HIGH
    Rule 5: IF all resources are MEDIUM         → RISK is MODERATE
    Rule 6: IF all resources are HIGH           → RISK is SAFE
    Rule 7: IF energy is HIGH AND oxygen is HIGH AND hull is HIGH → RISK is SAFE

Aggregation  : max operator (standard fuzzy OR)
Defuzzification : centroid method (centre of gravity)

Mission Health Index (MHI):
    Scalar in [0, 1] derived from defuzzification.
    0.0 = mission failure imminent
    1.0 = mission fully optimal
"""

import numpy as np


# ── Membership functions ──────────────────────────────────────────────────────

def trimf(x: float, a: float, b: float, c: float) -> float:
    """
    Triangular membership function.

    Args:
        x: input value
        a: left foot  (membership = 0)
        b: peak       (membership = 1)
        c: right foot (membership = 0)

    Returns:
        Degree of membership in [0, 1].
    """
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    return (c - x) / (c - b) if c != b else 1.0


def membership_low(value: float) -> float:
    """Degree of membership to fuzzy set LOW (universe [0, 2])."""
    return trimf(value, -0.01, 0.0, 1.0)


def membership_medium(value: float) -> float:
    """Degree of membership to fuzzy set MEDIUM (universe [0, 2])."""
    return trimf(value, 0.0, 1.0, 2.0)


def membership_high(value: float) -> float:
    """Degree of membership to fuzzy set HIGH (universe [0, 2])."""
    return trimf(value, 1.0, 2.0, 2.01)


def get_memberships(value: float) -> dict:
    """
    Return the degree of membership of a resource value to each fuzzy set.

    Args:
        value: resource level (0, 1 or 2 from SpaceMissionEnv)

    Returns:
        dict with keys 'low', 'medium', 'high' and float values in [0, 1].
    """
    return {
        "low"   : membership_low(value),
        "medium": membership_medium(value),
        "high"  : membership_high(value),
    }


# ── Fuzzy rule engine ─────────────────────────────────────────────────────────

# Risk output universe: 0 = safe, 1 = moderate, 2 = high, 3 = critical
RISK_SAFE     = 0.0
RISK_MODERATE = 1.0
RISK_HIGH     = 2.0
RISK_CRITICAL = 3.0


def evaluate_rules(energy: float, oxygen: float, food: float,
                   fuel: float, hull: float) -> dict:
    """
    Evaluate all fuzzy IF-THEN rules and return the activation degree
    of each risk output level.

    Rules (Mamdani style, min for AND, max for OR, max for aggregation):

        Rule 1: IF energy   is LOW              → RISK is CRITICAL
        Rule 2: IF oxygen   is LOW              → RISK is CRITICAL
        Rule 3: IF hull     is LOW              → RISK is CRITICAL
        Rule 4: IF food     is LOW OR fuel is LOW → RISK is HIGH
        Rule 5: IF all resources are MEDIUM     → RISK is MODERATE
        Rule 6: IF all resources are HIGH       → RISK is SAFE
        Rule 7: IF energy is HIGH AND oxygen is HIGH AND hull is HIGH → RISK is SAFE

    Args:
        energy, oxygen, food, fuel, hull : resource levels (floats in [0, 2])

    Returns:
        dict mapping risk level → activation degree (float in [0, 1])
    """
    # Get memberships for each resource
    me = get_memberships(energy)
    mo = get_memberships(oxygen)
    mf = get_memberships(food)
    mu = get_memberships(fuel)
    mh = get_memberships(hull)

    # Rule activations
    r1 = me["low"]                                          # energy LOW → CRITICAL
    r2 = mo["low"]                                          # oxygen LOW → CRITICAL
    r3 = mh["low"]                                          # hull   LOW → CRITICAL
    r4 = max(mf["low"], mu["low"])                          # food OR fuel LOW → HIGH
    r5 = min(me["medium"], mo["medium"],                    # all MEDIUM → MODERATE
             mf["medium"], mu["medium"], mh["medium"])
    r6 = min(me["high"], mo["high"],                        # all HIGH → SAFE
             mf["high"], mu["high"], mh["high"])
    r7 = min(me["high"], mo["high"], mh["high"])            # energy+oxygen+hull HIGH → SAFE

    # Aggregate by output level (max operator = fuzzy OR)
    activations = {
        RISK_SAFE    : max(r6, r7),
        RISK_MODERATE: r5,
        RISK_HIGH    : r4,
        RISK_CRITICAL: max(r1, r2, r3),
    }
    return activations


# ── Defuzzification ───────────────────────────────────────────────────────────

def defuzzify_centroid(activations: dict, resolution: int = 100) -> float:
    """
    Defuzzify using the centroid (centre of gravity) method.

    The output universe is [0, 3] (SAFE → CRITICAL).
    Each risk level activates a triangular output function clipped at its
    activation degree (Mamdani clipping).

    Args:
        activations : dict {risk_level: activation_degree}
        resolution  : number of discretisation points

    Returns:
        Crisp risk value in [0, 3]. Lower = safer.
    """
    universe = np.linspace(0.0, 3.0, resolution)
    aggregated = np.zeros(resolution)

    # Output membership functions (one triangle per risk level)
    output_mfs = {
        RISK_SAFE    : (0.0, 0.0, 1.0),
        RISK_MODERATE: (0.0, 1.0, 2.0),
        RISK_HIGH    : (1.0, 2.0, 3.0),
        RISK_CRITICAL: (2.0, 3.0, 3.0),
    }

    for risk_level, activation in activations.items():
        a, b, c = output_mfs[risk_level]
        mf_values = np.array([trimf(x, a, b, c) for x in universe])
        clipped   = np.minimum(mf_values, activation)   # Mamdani clipping
        aggregated = np.maximum(aggregated, clipped)     # max aggregation

    # Centroid
    denom = np.sum(aggregated)
    if denom == 0:
        return 1.5   # neutral mid-point if no rule fires
    crisp_risk = np.sum(universe * aggregated) / denom
    return float(crisp_risk)


def mission_health_index(crisp_risk: float) -> float:
    """
    Convert crisp risk [0, 3] to Mission Health Index (MHI) in [0, 1].
    MHI = 1 - (crisp_risk / 3)

    MHI = 1.0 → optimal mission status
    MHI = 0.0 → critical failure imminent
    """
    return float(np.clip(1.0 - (crisp_risk / 3.0), 0.0, 1.0))


# ── Main evaluation function ──────────────────────────────────────────────────

def evaluate_mission_state(energy: int, oxygen: int, food: int,
                            fuel: int, hull: int) -> dict:
    """
    Full fuzzy evaluation of a SpaceMissionEnv state.

    Args:
        energy, oxygen, food, fuel, hull : resource levels (0, 1 or 2)

    Returns:
        dict with:
            'memberships'  : per-resource membership degrees
            'activations'  : per-rule activation degrees
            'crisp_risk'   : defuzzified risk value [0, 3]
            'mhi'          : Mission Health Index [0, 1]
            'status'       : human-readable status label
    """
    memberships = {
        "energy": get_memberships(float(energy)),
        "oxygen": get_memberships(float(oxygen)),
        "food"  : get_memberships(float(food)),
        "fuel"  : get_memberships(float(fuel)),
        "hull"  : get_memberships(float(hull)),
    }

    activations = evaluate_rules(
        float(energy), float(oxygen), float(food),
        float(fuel), float(hull)
    )

    crisp_risk = defuzzify_centroid(activations)
    mhi        = mission_health_index(crisp_risk)

    # Human-readable status
    if mhi >= 0.75:
        status = "OPTIMAL"
    elif mhi >= 0.50:
        status = "STABLE"
    elif mhi >= 0.25:
        status = "WARNING"
    else:
        status = "CRITICAL"

    return {
        "memberships": memberships,
        "activations": activations,
        "crisp_risk" : round(crisp_risk, 4),
        "mhi"        : round(mhi, 4),
        "status"     : status,
    }


# ── Batch evaluation for training metrics ────────────────────────────────────

def evaluate_episode_states(states_history: list) -> list:
    """
    Evaluate fuzzy MHI for a sequence of (energy, oxygen, food, fuel, hull)
    tuples representing the resource trajectory of an episode.

    Args:
        states_history: list of (energy, oxygen, food, fuel, hull) tuples

    Returns:
        list of MHI values (one per step)
    """
    return [
        evaluate_mission_state(*s)["mhi"]
        for s in states_history
    ]
