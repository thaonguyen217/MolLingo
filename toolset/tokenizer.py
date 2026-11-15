import os
import re
import sys

# Allow running this file directly ("python3 toolset/tokenizer.py"), not just
# as a module. Executed as a script, sys.path[0] is toolset/, so the absolute
# "from toolset.* import ..." lines below would fail; put the repo root first.
if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolToSmiles as m2s
from rdkit.Chem import MolFromSmiles as s2m
from rdkit.Chem import FragmentOnBonds
from itertools import combinations
import json
from rdkit.Chem import BRICS
import pubchempy as pcp

from toolset.drug_like_filtering import analyze_molecule, remove_star_atoms
from toolset.config import VOCAB_PATH, MIN_BLOCK_FREQ

vocab_path = VOCAB_PATH
min_freq = MIN_BLOCK_FREQ

with open(vocab_path, "rb") as f:
    vocab = json.load(f)

# The vocab file stores corpus frequencies and is prebuilt with a cutoff of 20.
# Drop blocks below min_freq so both the require_vocab gate and the frequency
# std scoring below operate on the same restricted block set.
if min_freq > 20:
    vocab = {block: freq for block, freq in vocab.items() if freq >= min_freq}

vocab_sm = set(vocab.keys())

def smiles2name(smiles):
    compound = pcp.get_compounds(smiles, 'smiles')

    if compound:
        compound = compound[0]
        return compound.synonyms[0] if compound.synonyms else "N/A"
    else:
        print("No compound found for this SMILES")

def get_wildcard_pairs(mol, splitting_bonds):
    wildcard_pairs = []
    for bond in mol.GetBonds():
        if bond.GetIdx() in splitting_bonds:
            wildcard_pairs.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    return wildcard_pairs

def filter_close_values(values, tol=2):
    values = sorted(values)  # sort ascending
    result = []

    for v in values:
        if not result:
            result.append(v)
        elif v - result[-1] > tol:
            result.append(v)
    return result

def tokenize(mol):
    if mol is None:
        print('Invalid SMILES')
        return None, []
    if not analyze_molecule:
        print("Molecule violates Lipinski's rule of five, or contains wildcard atoms, or contains disconnected structures.")
        return mol, []

    brics_bonds = list(BRICS.FindBRICSBonds(mol))
    breakable_bonds = []
    for (a1, a2), labels in brics_bonds:
        if a1 != 0 and a2 != 0:
            bond = mol.GetBondBetweenAtoms(a1, a2)
            if bond is not None:
                breakable_bonds.append(bond.GetIdx())

    max_bonds = len(breakable_bonds)
    SPLITTING_BONDS = []

    if max_bonds > 1:
        for num_bonds in range(max_bonds, 0, -1):
            for bondset in combinations(breakable_bonds, num_bonds):
                frags_sm = m2s(FragmentOnBonds(mol, bondset)).split('.')
                if len([i for i in frags_sm if i.count('*')==1])==2 and len([i for i in frags_sm if i.count('*')>2])==0:
                    SPLITTING_BONDS.append(bondset)
                    if len(SPLITTING_BONDS) > 3:
                        return mol, SPLITTING_BONDS
            if len(SPLITTING_BONDS) > 0:
                return mol, SPLITTING_BONDS
        return mol, []
    else:
        return mol, [breakable_bonds]
        
def find_next_token(bond_pairs, frags_sm, begin):
    a, b = next(p for p in bond_pairs if begin in p)
    end = b if begin == a else a
    tag = f'[{end}*]'

    for sm in frags_sm:
        if tag in sm:
            temp = sm.replace(tag, '')
            try:
                n = int(re.search(r"\[(\d+)\*\]", temp).group(1))
                if n != 2:
                    sm = sm.replace(tag, '[2*]').replace(f'[{n}*]', '[1*]')
                else:
                    if end != 1:
                        sm = sm.replace(f'[{n}*]', '[1*]').replace(tag, '[2*]')
                    else:
                        sm = sm.replace(f'[{n}*]', '[3*]').replace(tag, '[2*]').replace(f'[3*]', '[1*]')
                return sm, n
            except:
                sm = sm.replace(tag, '[2*]')
    return None, None

def get_blocks_(smi: str, num_tokens= None, return_bonds=False):
    mol = s2m(smi)
    try:
        mol, splitting_bonds_groups_ = tokenize(mol)
        splitting_bonds_groups = [filter_close_values(i) for i in splitting_bonds_groups_]

        if mol is None or len(splitting_bonds_groups) == 0:
            return ((smi, smi), mol, []) if return_bonds else (smi, smi)

        # combine() rebuilds every cut as a SINGLE bond, so cutting a double or
        # triple bond silently alters the molecule (C=C comes back as C-C, two
        # hydrogens heavier). Restrict candidates to true single bonds.
        splittable = {
            b.GetIdx()
            for b in mol.GetBonds()
            if b.GetBondType() == Chem.BondType.SINGLE and not b.GetIsAromatic()
        }
        splitting_bonds_groups = [
            [i for i in group if i in splittable] for group in splitting_bonds_groups
        ]

        # --- build the [1*]/[2*] token chain for one chosen bond set ---------
        def build_sequences(bonds, nt):
            fragments = FragmentOnBonds(mol, bonds, addDummies=True)
            frags_sm = m2s(fragments).split(".")
            if len(frags_sm) == 0:
                return None

            bond_pairs = [
                [b.GetBeginAtomIdx(), b.GetEndAtomIdx()]
                for b in mol.GetBonds()
                if b.GetIdx() in bonds
            ]

            # The chain is linear, so exactly two blocks carry a single
            # attachment point (the two ends).
            first_tokens = [i for i in frags_sm if i.count("*") == 1]
            if len(first_tokens) != 2:
                return None

            case1 = []
            first = first_tokens[0]
            number = int(re.search(r"\[(\d+)\*\]", first).group(1))
            case1.append(first.replace(f"[{number}*]", "[1*]"))

            for _ in range(nt - 2):
                sm_next, number = find_next_token(bond_pairs, frags_sm, number)
                if sm_next is None:
                    return None
                case1.append(sm_next)

            last = first_tokens[1]
            number = int(re.search(r"\[(\d+)\*\]", last).group(1))
            case1.append(last.replace(f"[{number}*]", "[2*]"))

            case2 = case1[::-1]
            case2 = [
                i.replace("[2*]", "__tmp__")
                 .replace("[1*]", "[2*]")
                 .replace("__tmp__", "[1*]")
                for i in case2
            ]
            return ("^".join(case1), "^".join(case2))

        # --- enumerate every candidate fragmentation, bucketed by block count -
        # A candidate is admissible only if EVERY one of its blocks clears the
        # frequency cutoff. `vocab` is already filtered to min_freq at load
        # time, so a missing key means that block is below the cutoff.
        candidates = {}  # num_blocks -> [(std, bonds), ...]  all blocks >= min_freq
        fallback = {}    # num_blocks -> [(std, bonds), ...]  best effort, any blocks
        for splitting_bonds in splitting_bonds_groups:
            n = len(splitting_bonds)
            if n == 0:
                continue

            for i in range(1, n + 1):
                for bonds_candidate in combinations(splitting_bonds, i):
                    fragments_candidate = FragmentOnBonds(mol, bonds_candidate, addDummies=True)
                    frags_candidate = m2s(fragments_candidate).split(".")
                    frags_candidate = [re.sub(r"\[\d+\*\]", "[*]", sm) for sm in frags_candidate]

                    # Splitting a ring bond does not disconnect the molecule,
                    # so the block count is not simply i + 1.
                    if len(frags_candidate) < 2:
                        continue

                    nblocks = len(frags_candidate)
                    vals = [vocab.get(f) for f in frags_candidate]

                    # Best-effort tier: unknown blocks score 0, so this ranks
                    # every candidate, admissible or not.
                    fallback.setdefault(nblocks, []).append(
                        (float(np.array([v or 0 for v in vals]).std()), bonds_candidate)
                    )

                    if any(v is None for v in vals):
                        continue  # a block is below min_freq -> not admissible

                    candidates.setdefault(nblocks, []).append(
                        (float(np.array(vals).std()), bonds_candidate)
                    )

        # --- coarsest split first: 2 blocks, then 3, ... ----------------------
        # Within a level, prefer the most uniform block frequencies (lowest
        # std). Only if NO candidate at that level is admissible do we allow a
        # finer split. num_tokens, when given, acts as a floor on the count.
        floor = 2 if num_tokens is None else max(2, num_tokens)

        def pick(pool):
            for nt in sorted(k for k in pool if k >= floor):
                for _std, bonds in sorted(pool[nt], key=lambda c: c[0]):
                    built = build_sequences(bonds, nt)
                    if built is not None:
                        return built, bonds
            return None, None

        # Primary path: every block clears min_freq.
        built, bonds = pick(candidates)

        # Best-effort path: no level was fully admissible, so rather than
        # leaving the molecule as one opaque SMILES string, emit the coarsest
        # lowest-std fragmentation and accept some below-cutoff blocks.
        if built is None:
            built, bonds = pick(fallback)

        if built is None:
            return ((smi, smi), mol, []) if return_bonds else (smi, smi)
        return (built, mol, list(bonds)) if return_bonds else built

    except Exception:
        # fallback if anything fails
        return ((smi, smi), mol, []) if return_bonds else (smi, smi)

def get_blocks(smiles, num_tokens=None, verbose=False):
    r = []
    if '.' in smiles:
        for i in smiles.split('.'):
            r.append(get_blocks_(i, num_tokens=num_tokens))
        r0 = '.'.join([i[0] for i in r])
        r1 = '.'.join([i[1] for i in r])
        result = (r0, r1)
    else:
        result = get_blocks_(smiles, num_tokens=num_tokens)

    if verbose:
        _report(smiles, result, num_tokens)
    return result


def _report(smiles, result, num_tokens=None):
    """Print a readable breakdown of a get_blocks() result."""
    forward = result[0]
    blocks = forward.replace('.', '^').split('^')
    freqs = [vocab.get(re.sub(r"\[\d+\*\]", "[*]", b)) for b in blocks]
    primary = all(f is not None for f in freqs)

    width = 74
    print("=" * width)
    print("  BFE tokenizer")
    print("=" * width)
    print(f"  input        : {smiles}")
    print(f"  vocab        : {os.path.basename(vocab_path)}  "
          f"({len(vocab):,} blocks at min_freq={min_freq})")
    if num_tokens is not None:
        print(f"  num_tokens   : {num_tokens} (floor on block count)")
    print("-" * width)
    if len(blocks) == 1:
        print("  NOT FRAGMENTED - returned whole (no admissible split found)")
    else:
        print(f"  {len(blocks)} blocks   path: "
              f"{'primary (every block >= min_freq)' if primary else 'best-effort fallback'}")
        std = float(np.array([f or 0 for f in freqs]).std())
        print(f"  std of block frequencies: {std:,.1f}")
        print()
        for i, (b, f) in enumerate(zip(blocks, freqs), 1):
            shown = f"{f:,}" if f is not None else "below cutoff"
            print(f"    {i}. freq={shown:>14}  {b}")
    print("=" * width)

def _canon_block(block):
    """Canonical SMILES of a block with its attachment points unlabelled."""
    plain = re.sub(r"\[\d+\*\]", "[*]", str(block))
    mol = s2m(plain)
    return m2s(mol) if mol is not None else plain


def _build_substituent_table():
    """
    Correct names for the blocks that recur most in drug-like space.

    A block is a SUBSTITUENT, not a molecule. Naming it by the parent compound
    you get after deleting its attachment point is wrong in a way that misleads
    the LLM: [*]OC is a methoxy group, not methanol; [*]C(=O)O is a carboxyl
    group, not formic acid; [*]c1ccccc1 is phenyl, not benzene.
    """
    table = {
        # one attachment point
        '[*]C': 'methyl group',
        '[*]CC': 'ethyl group',
        '[*]O': 'hydroxyl group',
        '[*]OC': 'methoxy group',
        '[*]OCC': 'ethoxy group',
        '[*]N': 'amino group',
        '[*]NC': 'methylamino group',
        '[*]NCC': 'ethylamino group',
        '[*]C(=O)O': 'carboxyl group',
        '[*]C(=O)N': 'carboxamide group',
        '[*]C(=O)C': 'acetyl group',
        '[*]C#N': 'nitrile group',
        '[*][N+](=O)[O-]': 'nitro group',
        '[*]C(F)(F)F': 'trifluoromethyl group',
        '[*]S(=O)(=O)N': 'sulfonamide group',
        '[*]F': 'fluoro substituent',
        '[*]Cl': 'chloro substituent',
        '[*]Br': 'bromo substituent',
        '[*]I': 'iodo substituent',
        '[*]c1ccccc1': 'phenyl group',
        '[*]c1ccncc1': 'pyridin-4-yl group',
        '[*]c1cccnc1': 'pyridin-3-yl group',
        '[*]c1ccccn1': 'pyridin-2-yl group',
        '[*]N1CCCC1': 'pyrrolidin-1-yl group',
        '[*]N1CCCCC1': 'piperidin-1-yl group',
        '[*]N1CCOCC1': 'morpholin-4-yl group',
        '[*]N1CCN(C)CC1': '4-methylpiperazin-1-yl group',
        '[*]CN1CCN(C)CC1': '(4-methylpiperazin-1-yl)methyl group',
        # two attachment points (linkers)
        '[*]c1ccc([*])cc1': '1,4-phenylene linker',
        '[*]c1cccc([*])c1': '1,3-phenylene linker',
        '[*]NC([*])=O': 'amide linker',
        '[*]C(=O)N[*]': 'amide linker',
        '[*]O[*]': 'ether linker',
        '[*]N[*]': 'secondary amine linker',
        '[*]C[*]': 'methylene linker',
        '[*]CC[*]': 'ethylene linker',
        '[*]S(=O)(=O)[*]': 'sulfonyl linker',
    }
    built = {}
    for smiles, label in table.items():
        built[_canon_block(smiles)] = label
    return built


_SUBSTITUENT_NAMES = _build_substituent_table()

# PubChem's first synonym is often a registry identifier rather than a name.
_REGISTRY_ID = re.compile(
    r'^(SCHEMBL|CHEMBL|ZINC|AKOS|DTXSID|DTXCID|CID|MFCD|BDBM|NSC|EN300|'
    r'STL|SureCN|UNII|CAS|HMS|SMR|MLS|Z\d|AC1|A\d{6})',
    re.IGNORECASE,
)
_name_cache = {}


def _looks_like_a_name(text):
    """True for something a chemist would recognise, not a catalogue number."""
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < 3 or _REGISTRY_ID.match(text):
        return False
    # Pure identifiers: digits, dashes and a letter or two.
    return bool(re.search(r'[A-Za-z]{3}', text)) and not re.fullmatch(r'[\d\-\s]+', text)


def block_name(block):
    """
    A chemically correct label for one block.

    Order of preference:
      1. the curated substituent table (offline, deterministic, correct);
      2. the parent compound's common name from PubChem, stated as
         "<parent>-derived block" so the fragment is never presented as being
         that compound;
      3. the block SMILES itself.

    Results are cached, since the previous code issued one network call per
    block per iteration and a lookup miss silently degraded the molecular
    description the whole method depends on.
    """
    key = _canon_block(block)
    if key in _name_cache:
        return _name_cache[key]

    name = _SUBSTITUENT_NAMES.get(key)
    if name is None:
        parent = remove_star_atoms(re.sub(r"\[\d+\*\]", "[*]", str(block)))
        try:
            compounds = pcp.get_compounds(parent, 'smiles')
            synonyms = compounds[0].synonyms if compounds else []
        except Exception:
            synonyms = []
        readable = next((s for s in (synonyms or []) if _looks_like_a_name(s)), None)
        if readable:
            name = f"{readable.strip().lower()}-derived block"
        else:
            name = f"block {parent}"

    _name_cache[key] = name
    return name


def smiles2blocks(smiles):
    blocks = get_blocks(smiles)[0].replace('.', '^').split('^')
    try:
        return {i: block_name(i) for i in blocks}, blocks
    except Exception:
        return {i: remove_star_atoms(i) for i in blocks}, blocks

def smiles2NLdesc(sm):
    sm2b, blocks = smiles2blocks(sm)
    txt = []
    for k, v in sm2b.items():
        txt.append(f'{v} (SMILES: {k})')
    txt = ', '.join(txt)
    txt_ = f'The molecule is composed of the following sequence of building blocks: {txt}, with [1*] always connects with [2*] in the next block, and [2*] always connect with [1*] in the next block.'
    return txt_, txt, blocks

#################### MOL RECONSTRUCTION ####################
def get_dummy_atom_idx(mol, isotope_label):
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetIsotope() == isotope_label:
            return atom.GetIdx()
    return None

_ZIP_LABEL = 1  # shared atom-map number marking the junction for molzip


def _mark_junction(mol, isotope):
    """Tag the dummy atom carrying `isotope` for molzip; leave the others alone."""
    marked = [atom for atom in mol.GetAtoms()
              if atom.GetAtomicNum() == 0 and atom.GetIsotope() == isotope]
    if len(marked) != 1:
        # Zero means the block lacks the attachment point the chain expects;
        # more than one would make the junction ambiguous. Either way this
        # fragmentation is not reconstructible.
        raise ValueError(
            f"expected exactly one [{isotope}*] attachment point, found {len(marked)}"
        )
    atom = marked[0]
    atom.SetIsotope(0)
    atom.SetAtomMapNum(_ZIP_LABEL)
    return mol


def combine(frag1, wildcard1, frag2, wildcard2):
    """
    Join frag1's [wildcard1*] attachment point to frag2's [wildcard2*].

    Uses RDKit's molzip rather than CombineMols + AddBond + RemoveAtom.
    Deleting the dummy atoms renumbered their neighbours, which permuted the
    neighbour ordering that a chiral tag is defined against, so any stereocentre
    at an attachment point came back inverted. molzip preserves it. It also
    carries the original bond order instead of forcing BondType.SINGLE.
    """
    mol1 = _mark_junction(s2m(frag1), wildcard1)
    mol2 = _mark_junction(s2m(frag2), wildcard2)

    params = Chem.MolzipParams()
    params.label = Chem.MolzipLabel.AtomMapNumber
    mol = Chem.molzip(mol1, mol2, params)
    Chem.SanitizeMol(mol)
    return mol

def reconstruct_mol(sm):
    frags = sm.split('^')
    if len(frags) > 1:
        for i in range(len(frags)-1):
            if i == 0:
                mol = combine(frags[i], 1, frags[i+1], 2)
                sm = m2s(mol)
            else:
                mol = combine(sm, 1, frags[i+1], 2)
                sm = m2s(mol)
    return sm
#################### END MOL RECONSTRUCTION ####################


if __name__ == "__main__":
    """
    Usage (all equivalent; run from the repo root):

        python3 toolset/tokenizer.py
        python3 toolset/tokenizer.py smiles='<SMILES>' verbose=True
        python3 toolset/tokenizer.py smiles='<SMILES>' num_tokens=5
        python3 toolset/tokenizer.py '<SMILES>'

    With no smiles= given, tokenizes imatinib. verbose defaults to False, which
    prints just the block string; pass verbose=True for the full breakdown.
    Spaces around '=' or '==' are tolerated, as are --flag forms.
    """
    IMATINIB = 'Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1'

    _KEYS = r'smiles|verbose|num[_-]tokens'
    _VAL = r'"[^"]*"|\'[^\']*\'|\S+'
    text = ' '.join(sys.argv[1:])

    opts = {}
    for key, val in re.findall(rf'-{{0,2}}({_KEYS})\s*=+\s*({_VAL})', text):
        opts[key.replace('-', '_')] = val.strip('"\'')

    # Anything left over after removing key=value pairs is treated as a bare SMILES.
    leftover = re.sub(rf'-{{0,2}}(?:{_KEYS})\s*=+\s*(?:{_VAL})', ' ', text).strip().strip('"\'')

    smiles = opts.get('smiles') or leftover or IMATINIB
    verbose = str(opts.get('verbose', 'false')).lower() not in ('false', '0', 'no', 'none')
    num_tokens = int(opts['num_tokens']) if opts.get('num_tokens') else None

    result = get_blocks(smiles, num_tokens=num_tokens, verbose=verbose)
    if not verbose:
        print(result[0])