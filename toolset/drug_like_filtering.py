from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski



def remove_star_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    
    # Create an editable molecule
    rw_mol = Chem.RWMol(mol)
    
    # Identify indices of star atoms (AtomicNum 0)
    star_atoms = [atom.GetIdx() for atom in rw_mol.GetAtoms() 
                  if atom.GetAtomicNum() == 0]

    # Strategy 1: Attempt to remove them entirely
    for idx in sorted(star_atoms, reverse=True):
        rw_mol.RemoveAtom(idx)
    
    try:
        # Check if the molecule is still chemically "sane"
        Chem.SanitizeMol(rw_mol)
        return Chem.MolToSmiles(rw_mol)
    except:
        # Strategy 2: If removal fails (e.g., valence issues), replace * with C
        mol_alt = Chem.MolFromSmiles(smiles)
        rw_mol_alt = Chem.RWMol(mol_alt)
        for atom in rw_mol_alt.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomicNum(6) # Change 0 (Wildcard) to 6 (Carbon)
        
        Chem.SanitizeMol(rw_mol_alt)
        return Chem.MolToSmiles(rw_mol_alt)

def calculate_logS(mol, logP, mw, rot_bonds):
    aromatic_atoms = sum(
        1 for atom in mol.GetAtoms() if atom.GetIsAromatic()
    )
    heavy_atoms = mol.GetNumHeavyAtoms()
    aromatic_proportion = aromatic_atoms / heavy_atoms if heavy_atoms > 0 else 0

    logS = (
        0.16
        - 0.63 * logP
        - 0.0062 * mw
        + 0.066 * rot_bonds
        - 0.74 * aromatic_proportion
    )
    return logS

def analyze_molecule(smiles: str,
                     logS_threshold: float = -8.0,
                     relaxed_logP_limit: float = 8):
    
    if '*' in smiles:
        smiles = remove_star_atoms(smiles)
        
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    mw = Descriptors.MolWt(mol)
    logP = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)

    logS = calculate_logS(mol, logP, mw, rot_bonds)

    # Modified rule using logS
    lipinski_logS = (
        mw <= 700 and
        logP <= relaxed_logP_limit and
        logS >= logS_threshold and
        hbd <= 8 and
        hba <= 13
    )
    return lipinski_logS


if __name__ == '__main__':
    import os
    import pandas as pd
    from tqdm import tqdm

    # Anchored to this file (../data/corpus_all.csv) so it resolves from any
    # working directory. This module is imported both as toolset.* and flat,
    # so the path is resolved locally rather than via toolset.config.
    _data_dir = os.environ.get(
        'MOLLINGO_DATA_DIR',
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'),
    )
    df = pd.read_csv(os.path.join(_data_dir, 'corpus_all.csv'))
    # df = pd.read_csv('toy_corpus.csv')
    SMILES = list(df['SMILES'].values)

    SMILES_ = []
    for sm in tqdm(SMILES):
        try:
            if analyze_molecule(sm):
                SMILES_.append(sm)
        except:
            pass
        
    with open('drug_like.txt', 'w') as f:
        f.write('\n'.join(SMILES_))

# ---------------------------------------------------------------------- #
# Chemical plausibility                                                   #
# ---------------------------------------------------------------------- #
#
# RDKit parsing is not a plausibility test. An LLM asked to rewrite a molecule
# can produce a string that sanitizes cleanly but describes a species that does
# not exist, and the property oracles will score it anyway -- they have no
# applicability domain and no way to refuse. Observed in a raw-SMILES
# optimization run:
#
#   parent    ...c2ccccc2S1(=O)=O        a sulfonamide
#   proposed  ...c2ccccc2[SH]1(C)=O      sulfur at valence 6 carrying a hydrogen
#
# RDKit accepts it, so `smiles_after_valid` was True, and the DILI oracle
# returned 0.09 against the parent's 0.98 -- an apparent 0.89 improvement
# produced entirely by leaving the chemical space the oracle was trained on.
#
# The two rules below were calibrated against the 67 marketed drugs of the
# DILI benchmark: both flag zero of them, so nothing real is excluded, while
# each catches an artefact class actually seen in generated output. They are
# deliberately narrow -- this is not a general plausibility model, and a
# stricter filter (PAINS, or Brenk) rejects 12% and 49% of those same real
# drugs respectively, which would corrupt a benchmark far worse than the
# artefacts do.

_PERIODIC_TABLE = Chem.GetPeriodicTable()


def implausible_reason(molecule):
    """
    Why `molecule` cannot exist, or None if nothing is obviously wrong.

    Accepts a SMILES string or an RDKit Mol. Catches:

    * explicit hydrogens on an atom already past its normal valence, e.g.
      [SH](C)(=O) in a ring -- a hypervalent sulfur that still bears an H.
      Legitimate cases pass: [nH] pyrrole (valence 3 of 3), [NH2+] (4 of 3+1),
      a C[SH] thiol (2 of 2), and any sulfonamide (valence 6, no explicit H).
    * unpaired electrons, i.e. a radical such as [CH2]c1ccccc1.

    Known limitation: an H-phosphonate, [PH](=O)(O)O, is a real compound that
    the first rule flags. None occur in the benchmark sets, but a run over
    phosphorus chemistry would need the rule relaxed for P.
    """
    mol = Chem.MolFromSmiles(molecule) if isinstance(molecule, str) else molecule
    if mol is None:
        return "not a parseable molecule"

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            continue
        if atom.GetNumRadicalElectrons():
            return (f"unpaired electron on {atom.GetSymbol()} "
                    f"(radical, not a stable molecule)")
        if atom.GetNumExplicitHs():
            allowed = (_PERIODIC_TABLE.GetDefaultValence(atom.GetAtomicNum())
                       + atom.GetFormalCharge())
            if allowed > 0 and atom.GetTotalValence() > allowed:
                return (f"{atom.GetSymbol()} carries an explicit hydrogen at "
                        f"valence {atom.GetTotalValence()}, above the {allowed} "
                        f"its charge state allows (hypervalent)")
    return None
