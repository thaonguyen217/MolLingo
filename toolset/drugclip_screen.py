"""
De novo hit finding by DrugCLIP fragment screening (§2.3.1).

The ChEMBL route needs known binders. For a novel or understudied target there
are none, and §2.3.1 falls back to ligand-based virtual screening over a
curated library with DrugCLIP [18], a contrastive model that embeds pockets and
molecules into a shared space so that screening is a nearest-neighbour lookup
rather than docking.

DrugCLIP is third-party and is NOT vendored into this repo: the checkpoint is
1.2 GB and the pre-encoded library 1.4 GB. It also needs a different
environment (`unicore`, which lives in the `unimol` env). This module therefore
drives the installed copy as a subprocess, and only the small pocket file is
written per call. Paths come from toolset/config.py and honour
MOLLINGO_DRUGCLIP_DIR / MOLLINGO_DRUGCLIP_PYTHON.

What a screen does:

  1. fetch a structure for the target (the co-crystal route first, so the box
     can be centred on a real ligand site, else an apo structure)
  2. locate the site: the bound ligand's centroid when there is one, otherwise
     the most buried region, the same heuristic the blind-docking path uses
  3. write the pocket -- every protein atom within POCKET_RADIUS of that
     centre -- as the {pocket, pocket_atoms, pocket_coordinates} LMDB record
     DrugCLIP's retrieval task expects
  4. run unimol/retrieval.py against the pre-encoded library and read back
     `ranked_compounds.txt`, which is `SMILES<TAB>score`, best first

Returns {smiles: score}, highest score first, so the caller can treat it like
the normalized affinities the ChEMBL route produces.
"""

import os
import pickle
import shutil
import subprocess
import tempfile

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser

from toolset.config import DRUGCLIP_DIR, DRUGCLIP_PYTHON

# Atoms within this distance of the site centre form the pocket. DrugCLIP caps
# the pocket at 256 atoms (--max-pocket-atoms), and 10 A around a ligand
# centroid lands in that range for a typical site.
POCKET_RADIUS = 10.0

# Minutes before a screen is abandoned. Embeddings for the library are cached
# in data/emb, so a screen is a matrix product rather than a model run, but the
# first call on a cold cache is slow.
SCREEN_TIMEOUT = 1800


def _structure_for(uniprot_id, protein_name, scratch):
    """
    A structure file for the target, and the bound ligand id if there is one.

    Tries the co-crystal route first: a ligand in the structure marks a real
    binding site, which is a far better pocket centre than any geometric
    guess.
    """
    from toolset.download_protein_complex_rcsb import download_protein_complex_rcsb
    from toolset.download_protein_rcsb import download_protein_rcsb

    for finder, with_ligand in ((download_protein_complex_rcsb, True),
                                (download_protein_rcsb, False)):
        try:
            result = finder(uniprot_id, protein_name)
        except Exception:
            continue
        if not result or not result.get("pdb_id"):
            continue
        from toolset.autodockvina_wo_ligand import download_cif
        path = download_cif(f"{result['pdb_id']}.cif", scratch)
        return path, (result.get("ligand_id") if with_ligand else None)

    # No experimental structure. For a novel target that is the norm rather
    # than the exception, and a predicted structure is enough to define a
    # pocket for a similarity screen.
    # download_alphafold_pdb writes the file and returns None, so the path has
    # to be reconstructed rather than taken from the return value.
    predicted = os.path.join(scratch, f"AF-{uniprot_id}-F1-model_v6.pdb")
    try:
        from toolset.download_alphafold_pdb import download_alphafold_pdb
        download_alphafold_pdb(uniprot_id, save_dir=scratch)
    except Exception:
        pass
    if os.path.isfile(predicted):
        return predicted, None
    return None, None


def _site_centre(structure, ligand_id):
    """The bound ligand's centroid, or the most buried region if there is none."""
    protein, ligand = [], []
    for atom in structure.get_atoms():
        residue = atom.get_parent()
        name = residue.get_resname().strip()
        if residue.get_id()[0] == " ":
            protein.append(atom.get_coord())
        elif ligand_id and name == ligand_id:
            ligand.append(atom.get_coord())

    if not protein:
        raise ValueError("no protein atoms in the structure")
    if ligand:
        return np.asarray(ligand).mean(axis=0), np.asarray(protein)

    from toolset.autodockvina_wo_ligand import _buried_centroid
    protein = np.asarray(protein)
    return _buried_centroid(protein), protein


def write_pocket_lmdb(structure_path, out_path, name, ligand_id=None,
                      radius=POCKET_RADIUS):
    """
    Write the one-record pocket LMDB DrugCLIP's retrieval task reads.

    The record mirrors the format of the installation's own pocket.lmdb:
    PDB atom names (CA, CB, NZ ... which the task maps onto its C/N/O/S/H
    vocabulary) and one coordinate array per atom.
    """
    parser = (MMCIFParser(QUIET=True) if str(structure_path).endswith(".cif")
              else PDBParser(QUIET=True))
    structure = parser.get_structure("target", structure_path)
    centre, _ = _site_centre(structure, ligand_id)

    atoms, coordinates = [], []
    for atom in structure.get_atoms():
        residue = atom.get_parent()
        if residue.get_id()[0] != " ":          # protein residues only
            continue
        coord = atom.get_coord()
        if np.linalg.norm(coord - centre) <= radius:
            atoms.append(atom.get_id())
            coordinates.append(np.asarray(coord, dtype=np.float32))

    if not atoms:
        raise ValueError(f"no protein atoms within {radius} A of the site centre")

    record = {"pocket": name, "pocket_atoms": atoms,
              "pocket_coordinates": coordinates}

    # `lmdb` is a dependency of the DrugCLIP environment, not of MolLingo's.
    # Rather than add it here, the record is pickled and the DrugCLIP
    # interpreter -- which is about to be used anyway -- writes the database.
    record_path = out_path + ".record.pkl"
    with open(record_path, "wb") as handle:
        pickle.dump(record, handle, protocol=4)

    writer = (
        "import lmdb, pickle, sys\n"
        "record = pickle.load(open(sys.argv[1], 'rb'))\n"
        "env = lmdb.open(sys.argv[2], subdir=False, map_size=1 << 24)\n"
        "txn = env.begin(write=True)\n"
        "txn.put(b'0', pickle.dumps(record, protocol=-1))\n"
        "txn.commit(); env.close()\n"
    )
    completed = subprocess.run(
        [DRUGCLIP_PYTHON, "-c", writer, record_path, out_path],
        capture_output=True, timeout=300)
    if completed.returncode != 0 or not os.path.exists(out_path):
        tail = (completed.stderr or b"").decode(errors="replace")[-400:]
        raise RuntimeError(f"could not write the pocket database: {tail.strip()}")

    return out_path, len(atoms), centre


def drugclip_screen(uniprot_id, protein_name=None, top_k=30, device=None):
    """
    Screen the DrugCLIP library against a target's pocket.

    Returns {smiles: score} ordered best first, or {} if the screen could not
    run (no structure, missing install, subprocess failure). Failure is
    reported by the caller rather than raised, so a de novo attempt that is
    not possible on this machine does not abort a pipeline run.
    """
    if not os.path.isdir(DRUGCLIP_DIR) or not os.path.isfile(DRUGCLIP_PYTHON):
        raise FileNotFoundError(
            f"DrugCLIP not found: expected the install at {DRUGCLIP_DIR} and "
            f"its interpreter at {DRUGCLIP_PYTHON}. Set MOLLINGO_DRUGCLIP_DIR "
            f"and MOLLINGO_DRUGCLIP_PYTHON.")

    checkpoint = os.path.join(DRUGCLIP_DIR, "checkpoint_best.pt")
    mols = os.path.join(DRUGCLIP_DIR, "mols.lmdb")
    for path in (checkpoint, mols):
        if not os.path.exists(path):
            raise FileNotFoundError(f"DrugCLIP is missing {path}")

    scratch = tempfile.mkdtemp(prefix="mollingo_drugclip_")
    try:
        structure_path, ligand_id = _structure_for(
            uniprot_id, protein_name or uniprot_id, scratch)
        if not structure_path:
            raise RuntimeError(f"no structure available for {uniprot_id}")

        pocket_path = os.path.join(scratch, "pocket.lmdb")
        _, n_atoms, centre = write_pocket_lmdb(
            structure_path, pocket_path, uniprot_id, ligand_id)

        # retrieval.py writes its ranking into --emb-dir, which is also where
        # the library embedding cache lives. Pointing it at the install's own
        # data/emb would overwrite whatever ranking is already there from a
        # previous screen. Instead a per-call directory is used, with the
        # ~900 MB cache symlinked in so it is still reused rather than
        # recomputed.
        emb_dir = os.path.join(scratch, "emb")
        os.makedirs(emb_dir, exist_ok=True)
        cache_dir = os.path.join(DRUGCLIP_DIR, "data", "emb")
        if os.path.isdir(cache_dir):
            for cached in os.listdir(cache_dir):
                if cached.endswith(".pkl"):
                    os.symlink(os.path.join(cache_dir, cached),
                               os.path.join(emb_dir, cached))
        command = [
            DRUGCLIP_PYTHON, "./unimol/retrieval.py",
            "--user-dir", "./unimol", "./data",
            "--valid-subset", "test",
            "--results-path", scratch,
            "--num-workers", "4", "--ddp-backend=c10d", "--batch-size", "8",
            "--task", "drugclip", "--loss", "in_batch_softmax",
            "--arch", "drugclip", "--max-pocket-atoms", "256",
            "--fp16", "--fp16-init-scale", "4", "--fp16-scale-window", "256",
            "--seed", "1", "--path", checkpoint,
            "--log-interval", "100", "--log-format", "simple",
            "--mol-path", mols,
            "--pocket-path", pocket_path,
            "--emb-dir", emb_dir,
        ]
        # Inherit the caller's GPU selection. Forcing a device here sent the
        # screen to GPU 0 whatever the pipeline was told to use, and on a
        # shared node that GPU was full: the subprocess died inside
        # build_model with an out-of-memory error that looked like a DrugCLIP
        # fault rather than a device mix-up.
        environment = dict(os.environ)
        if device is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(device)
        completed = subprocess.run(
            command, cwd=DRUGCLIP_DIR, env=environment,
            capture_output=True, timeout=SCREEN_TIMEOUT)
        if completed.returncode != 0:
            # Keep enough of the traceback to name the real cause -- an
            # out-of-memory on a busy GPU and a genuine model error surface at
            # the same line otherwise.
            text = (completed.stderr or b"").decode(errors="replace")
            interesting = [line for line in text.splitlines()
                           if ("Error" in line or "error" in line
                               or "memory" in line.lower())]
            detail = " | ".join(interesting[-3:]) or text.strip()[-800:]
            raise RuntimeError(
                f"DrugCLIP retrieval failed (exit {completed.returncode}) on "
                f"CUDA_VISIBLE_DEVICES="
                f"{environment.get('CUDA_VISIBLE_DEVICES', 'unset')}: {detail}")

        ranked_path = os.path.join(emb_dir, "ranked_compounds.txt")
        if not os.path.isfile(ranked_path):
            raise RuntimeError(
                f"DrugCLIP produced no ranking at {ranked_path}")

        from rdkit import Chem
        hits, seen = {}, set()
        with open(ranked_path) as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                smiles, score = parts[0].strip(), parts[1].strip()
                if not smiles or smiles in seen:
                    continue
                if Chem.MolFromSmiles(smiles) is None:
                    continue
                seen.add(smiles)
                try:
                    hits[smiles] = float(score)
                except ValueError:
                    continue
                if len(hits) >= top_k:
                    break
        print(f"DrugCLIP: pocket of {n_atoms} atoms centred at "
              f"{centre[0]:.1f},{centre[1]:.1f},{centre[2]:.1f}; "
              f"{len(hits)} molecules taken from the ranking")
        return hits
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
