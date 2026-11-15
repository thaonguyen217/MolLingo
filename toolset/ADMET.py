import torch
from toolset.admet_inference import admet_inference
from toolset.config import MODEL_WEIGHTS_DIR
from toolset.admet_inference import admet_inference_dict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def PredictAMES(smiles):
    if not isinstance(smiles, list):
        smiles = [smiles]
    return list(admet_inference(smiles, name='AMES', device=device, ckpt_path=MODEL_WEIGHTS_DIR))

def PredictDILI(smiles):
    if not isinstance(smiles, list):
        smiles = [smiles]
    return list(admet_inference(smiles, name='DILI', device=device, ckpt_path=MODEL_WEIGHTS_DIR))

def PredicthERG(smiles):
    if not isinstance(smiles, list):
        smiles = [smiles]
    return list(admet_inference(smiles, name='hERG', device=device, ckpt_path=MODEL_WEIGHTS_DIR))

def PredictBBBP(smiles):
    if not isinstance(smiles, list):
        smiles = [smiles]
    return list(admet_inference(smiles, name='BBBP', device=device, ckpt_path=MODEL_WEIGHTS_DIR))

def PredictPgp(smiles):
    if not isinstance(smiles, list):
        smiles = [smiles]
    return list(admet_inference(smiles, name='Pgp', device=device, ckpt_path=MODEL_WEIGHTS_DIR))

# def PredictCYP3A4(smiles):
#     if not isinstance(smiles, list):
#         smiles = [smiles]
#     return list(admet_inference(smiles, name='CYP3A4', device=device, ckpt_path=MODEL_WEIGHTS_DIR))

# def PredictCYP2D6(smiles):
#     if not isinstance(smiles, list):
#         smiles = [smiles]
#     return list(admet_inference(smiles, name='DILI', device=device, ckpt_path=MODEL_WEIGHTS_DIR))

# def PredictCYP2C9(smiles):
#     if not isinstance(smiles, list):
#         smiles = [smiles]
#     return list(admet_inference(smiles, name='DILI', device=device, ckpt_path=MODEL_WEIGHTS_DIR))

# def PredictCYP2C19(smiles):
#     if not isinstance(smiles, list):
#         smiles = [smiles]
#     return list(admet_inference(smiles, name='DILI', device=device, ckpt_path=MODEL_WEIGHTS_DIR))

def PredictHIA(smiles):
    if not isinstance(smiles, list):
        smiles = [smiles]
    return list(admet_inference(smiles, name='HIA', device=device, ckpt_path=MODEL_WEIGHTS_DIR))

def PredictBioavailability(smiles):
    if not isinstance(smiles, list):
        smiles = [smiles]
    return list(admet_inference(smiles, name='Bioavailability', device=device, ckpt_path=MODEL_WEIGHTS_DIR))

# def PredictLipophilicity(smiles):
#     if not isinstance(smiles, list):
#         smiles = [smiles]
#     return list(admet_inference(smiles, name='DILI', device=device, ckpt_path=MODEL_WEIGHTS_DIR))

# def PredictSolubility(smiles):
#     if not isinstance(smiles, list):
#         smiles = [smiles]
#     return list(admet_inference(smiles, name='DILI', device=device, ckpt_path=MODEL_WEIGHTS_DIR))

# def PredictPPBR(smiles):
#     if not isinstance(smiles, list):
#         smiles = [smiles]
#     return list(admet_inference(smiles, name='DILI', device=device, ckpt_path=MODEL_WEIGHTS_DIR))

# def PredictHalflife(smiles):
#     if not isinstance(smiles, list):
#         smiles = [smiles]
#     return list(admet_inference(smiles, name='DILI', device=device, ckpt_path=MODEL_WEIGHTS_DIR))

# def PredictClearance(smiles):
#     if not isinstance(smiles, list):
#         smiles = [smiles]
#     return list(admet_inference(smiles, name='DILI', device=device, ckpt_path=MODEL_WEIGHTS_DIR))

ADMET_ENDPOINTS = ('AMES', 'DILI', 'hERG', 'BBBP', 'Pgp', 'HIA')


def ADMET_Profile(smiles):
    """
    {smiles: {endpoint: probability}} for every molecule that scores.

    Each endpoint is evaluated once over the whole list rather than once per
    molecule, and results are keyed by SMILES so that a molecule the
    featurizer drops cannot shift the predictions of the others. Molecules
    missing from any endpoint are omitted, as before.
    """
    if not isinstance(smiles, list):
        smiles = [smiles]
    if not smiles:
        return {}

    per_endpoint = {}
    for endpoint in ADMET_ENDPOINTS:
        try:
            per_endpoint[endpoint] = admet_inference_dict(
                smiles, name=endpoint, device=device,
                ckpt_path=MODEL_WEIGHTS_DIR,
            )
        except Exception as exc:
            # Previously a bare "except:" that printed only the SMILES, so a
            # missing checkpoint or an out-of-memory GPU was indistinguishable
            # from an unparseable molecule -- and the caller saw an empty dict
            # that reads like a legitimate "no result".
            print(f"ADMET endpoint {endpoint} failed: {type(exc).__name__}: {exc}")
            return {}

    d = {}
    for smi in smiles:
        if all(smi in per_endpoint[e] for e in ADMET_ENDPOINTS):
            d[smi] = {e: per_endpoint[e][smi] for e in ADMET_ENDPOINTS}
        else:
            print(f"Failed to predict ADMET properties for {smi}")
    return d

# Fraction of the maximum attainable score that a molecule must exceed to be
# forwarded to a later stage (Appendix D: "thresholding at ADMET(m) > 0.5x").
ADMET_THRESHOLD_FRACTION = 0.5


def admet_score_threshold(main_task=None):
    """Hard-filter threshold for the configured objective: 2.5 of 5, or 0.5 of 1."""
    return ADMET_THRESHOLD_FRACTION * (5.0 if main_task is None else 1.0)


def ADMET_based_scoring(scores, BBBP=True, main_task=None):
    """
    Scalar ADMET objective; higher is always better. Returns (score, max_val).

    scores: {'AMES': prob, 'DILI': prob, 'hERG': prob, 'BBBP': prob,
             'Pgp': prob, 'HIA': prob}

    Aggregate objective (main_task=None), per Appendix D:

        ADMET(m) = (1 - p_DILI) + (1 - p_AMES) + (1 - p_hERG)
                   + (1 - p_Pgp) + p_HIA

    For the toxicity and efflux endpoints a lower probability is favorable, so
    the complement is taken; HIA enters directly because higher absorption is
    better. The score therefore ranges from 0 to 5. BBB penetration is
    deliberately EXCLUDED from the aggregate: its desirability is
    target-dependent (required for CNS targets, undesirable for
    peripherally-acting drugs) and is assessed separately, via `bbbp_desired`
    and the dedicated 'BBBP' task.

    For a single endpoint the same complement convention applies, giving a
    score in [0, 1]. The `BBBP` argument sets the desired direction and is
    used only by the 'BBBP' task.

    The hard filter is `score > admet_score_threshold(main_task)`.
    """
    if main_task is None:
        score = ((1 - scores['AMES'])
                 + (1 - scores['DILI'])
                 + (1 - scores['hERG'])
                 + (1 - scores['Pgp'])
                 + scores['HIA'])
        return score, 5.0

    tasks = list(scores.keys())
    assert main_task in tasks, f'task must be in {tasks}'
    if main_task == 'HIA':
        return scores['HIA'], 1.0
    elif main_task == 'BBBP':
        # Higher is better only when CNS penetration is wanted.
        return (scores['BBBP'] if BBBP else 1 - scores['BBBP']), 1.0
    else:  # AMES, DILI, hERG, Pgp -- lower probability is favorable
        return 1 - scores[main_task], 1.0