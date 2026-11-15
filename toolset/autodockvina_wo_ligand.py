import requests
import os
from Bio.PDB import MMCIFParser, PDBIO, Select, PDBParser, NeighborSearch
import numpy as np
import subprocess
import tempfile
import shutil
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# --- CONFIGURATION ---
# Resolved in toolset/config.py: defaults to the active Python environment
# (sys.prefix), overridable via MOLLINGO_MGLTOOLS_PATH / MOLLINGO_VINA_EXE.
from toolset.config import MGLTOOLS_PATH, VINA_EXE

# Seconds before a structure-database request is abandoned. Without a
# timeout these calls block indefinitely: a run against a target with no
# structure sat idle on an open socket instead of failing and moving on.
NETWORK_TIMEOUT = 30
PY2_EXE = shutil.which('python2') or os.path.join(MGLTOOLS_PATH, 'bin', 'pythonsh')

def download_cif(filename, output_dir='./temp'):
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir, exist_ok=True)

    url = f"https://files.rcsb.org/download/{filename}"
    response = requests.get(url, timeout=NETWORK_TIMEOUT)
    response.raise_for_status() 
    save_path = os.path.join(output_dir, 'protein.cif')
    with open(save_path, "wb") as f:
        f.write(response.content)
    return save_path

class ProteinOnlySelect(Select):
    """Filters out water and non-protein residues (HETATM)."""
    def accept_residue(self, residue):
        resname = residue.get_resname().strip()
        # Exclude water and common ions/solvents if necessary
        if resname in ['HOH', 'DOD', 'WAT']:
            return 0
        # This keeps standard amino acids; Bio.PDB ' ' check identifies protein atoms
        return 1 if residue.get_id()[0] == ' ' else 0

def prep_protein_only(cif_input, padding=10, save_dir='./temp'):
    """Prepares receptor and calculates a 'Blind Docking' box covering the whole protein."""
    # A predicted structure arrives as .pdb rather than .cif.
    parser = (MMCIFParser(QUIET=True) if str(cif_input).lower().endswith('.cif')
              else PDBParser(QUIET=True))
    structure = parser.get_structure('protein', cif_input)
    
    all_coords = []
    for atom in structure.get_atoms():
        # Only include protein atoms for box calculation
        if atom.get_parent().get_id()[0] == ' ':
            all_coords.append(atom.get_coord())
            
    if not all_coords:
        raise ValueError("No protein atoms found in the file.")
    
    all_coords = np.array(all_coords)

    span = (np.max(all_coords, axis=0) - np.min(all_coords, axis=0)) + padding
    if np.all(span <= MAX_BOX_EDGE):
        # Small enough that the whole protein is a sane search space.
        center, size = np.mean(all_coords, axis=0), span
    else:
        # Boxing the entire protein is intractable: xanthine dehydrogenase
        # gives a 229 x 90 x 199 A box, ~4.1 million A^3 against the ~10^4 A^3
        # of a normal site, and Vina then runs for minutes without converging
        # on anything useful. Cap the edge and centre it on the most buried
        # region, which is a cheap stand-in for the binding pocket: burial is
        # counted as protein atoms within BURIAL_RADIUS of each atom, and the
        # box is centred on the centroid of the most buried decile.
        center = _buried_centroid(all_coords)
        size = np.minimum(span, np.full(3, float(MAX_BOX_EDGE)))

    return _finish_receptor(structure, cif_input, save_dir, center, size)


# A blind-docking box edge, in angstroms. Large enough to contain a binding
# site with room for the ligand to translate and rotate, small enough that
# Vina's search converges.
MAX_BOX_EDGE = 30.0
# Seconds before a single Vina search is abandoned.
VINA_TIMEOUT = 300
BURIAL_RADIUS = 10.0
# Angstroms of context kept around the box when trimming the receptor, so the
# residues lining the pocket are not clipped at the boundary.
BOX_RESIDUE_MARGIN = 8.0


def _buried_centroid(coords, sample_limit=4000):
    """Centroid of the most buried atoms, as a proxy for the binding pocket."""
    # Subsample for the O(n^2) neighbour count; burial is a smooth property,
    # so a few thousand atoms locate the protein core just as well.
    if len(coords) > sample_limit:
        step = len(coords) // sample_limit + 1
        sample = coords[::step]
    else:
        sample = coords

    # Chunked to keep the pairwise distance matrix small.
    counts = np.empty(len(sample))
    for start in range(0, len(sample), 512):
        chunk = sample[start:start + 512]
        distances = np.linalg.norm(chunk[:, None, :] - sample[None, :, :], axis=2)
        counts[start:start + len(chunk)] = (distances < BURIAL_RADIUS).sum(axis=1)

    cutoff = np.quantile(counts, 0.9)
    return sample[counts >= cutoff].mean(axis=0)


class BoxResidueSelect(ProteinOnlySelect):
    """
    Protein residues with at least one atom inside the padded docking box.

    Residues far from the box cannot influence the result, but Vina still pays
    for them when it builds its grid maps: passing the whole xanthine
    dehydrogenase dimer produced a 6.5 MB receptor and a ~150 s search even
    after the box itself was capped. The margin keeps the sidechains that line
    the pocket from being clipped at the boundary.
    """

    def __init__(self, center, size, margin=BOX_RESIDUE_MARGIN):
        half = np.asarray(size, dtype=float) / 2.0 + margin
        self.low = np.asarray(center, dtype=float) - half
        self.high = np.asarray(center, dtype=float) + half

    def accept_residue(self, residue):
        if not super().accept_residue(residue):
            return 0
        for atom in residue:
            coord = atom.get_coord()
            if np.all(coord >= self.low) and np.all(coord <= self.high):
                return 1
        return 0


def _finish_receptor(structure, cif_input, save_dir, center, size):

    stem = os.path.splitext(os.path.basename(str(cif_input)))[0]
    receptor_pdb_name = stem + '_receptor.pdb'
    receptor_pdb_path = os.path.join(save_dir, receptor_pdb_name)

    io = PDBIO()
    io.set_structure(structure)
    io.save(receptor_pdb_path, select=BoxResidueSelect(center, size))

    return receptor_pdb_path, center, size

# ---------------- PDB → PDBQT ----------------
def convert_to_pdbqt(input_pdb_path, is_receptor=True):
    save_dir = os.path.dirname(input_pdb_path)
    input_file_name = os.path.basename(input_pdb_path)
    output_file_name = input_file_name.replace('.pdb', '.pdbqt')
    
    script = 'prepare_receptor4.py' if is_receptor else 'prepare_ligand4.py'
    script_path = os.path.join(MGLTOOLS_PATH, 'MGLToolsPckgs', 'AutoDockTools', 'Utilities24', script)

    cmd = [PY2_EXE, script_path, '-o', output_file_name]
    if is_receptor:
        cmd += ['-r', input_file_name, '-A', 'checkhydrogens', '-U', 'nwaters']
    else:
        cmd += ['-l', input_file_name, '-A', 'hydrogens']

    subprocess.run(cmd, check=True, cwd=os.path.abspath(save_dir), capture_output=True)
    return os.path.join(save_dir, output_file_name)

# ---------------- SMILES → Ligand PDBQT ----------------
def prepare_ligand_pdb(smiles, ligand_name='ligand', save_dir='./temp'):
    os.makedirs(save_dir, exist_ok=True)
    pdb_file_name = f"{ligand_name}.pdb"
    pdbqt_file_name = f"{ligand_name}.pdbqt"
    
    pdb_full_path = os.path.join(save_dir, pdb_file_name)
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)

    Chem.MolToPDBFile(mol, pdb_full_path)

    script_path = os.path.join(MGLTOOLS_PATH, 'MGLToolsPckgs', 'AutoDockTools', 'Utilities24', 'prepare_ligand4.py')
    cmd = [PY2_EXE, script_path, '-l', pdb_file_name, '-o', pdbqt_file_name, '-A', 'hydrogens']
    subprocess.run(cmd, check=True, cwd=os.path.abspath(save_dir), capture_output=True)

    return pdb_full_path, os.path.join(save_dir, pdbqt_file_name)


def identify_accurate_growth(receptor_pdb, docked_pdb, contact_threshold=7.0, max_hotspots=5):
    parser = PDBParser(QUIET=True)
    receptor_struct = parser.get_structure('receptor', receptor_pdb)
    ligand_struct = parser.get_structure('ligand', docked_pdb)
    
    # Setup NeighborSearch for residue lookup
    receptor_atoms = [a for a in receptor_struct.get_atoms() if a.element.strip().upper() != 'H']
    ns = NeighborSearch(receptor_atoms)
    
    ligand_atoms = [a for a in ligand_struct.get_atoms() if a.element.strip().upper() != 'H']
    ligand_coords = np.array([a.coord for a in ligand_atoms])
    receptor_coords = np.array([a.coord for a in receptor_atoms])
    
    hotspots = []

    for i, l_atom in enumerate(ligand_atoms):
        l_coord = l_atom.coord
        # Get both pieces of info
        l_index = i
        l_type = l_atom.element.strip().upper()
        
        # 1. Find Nearby Residues
        nearby_res_objs = ns.search(l_coord, contact_threshold, level='R')
        nearby_residues = [f"{r.get_resname()}{r.get_id()[1]}" for r in nearby_res_objs]
        
        # 2. Estimate Available Volume
        grid_size = 0.5  
        box_edge = 5.0
        grid_points = np.mgrid[
            l_coord[0]-box_edge/2 : l_coord[0]+box_edge/2 : grid_size,
            l_coord[1]-box_edge/2 : l_coord[1]+box_edge/2 : grid_size,
            l_coord[2]-box_edge/2 : l_coord[2]+box_edge/2 : grid_size
        ].reshape(3, -1).T
        
        if len(grid_points) > 0:
            prot_dists = np.min(np.linalg.norm(grid_points[:, None] - receptor_coords, axis=2), axis=1)
            lig_dists = np.min(np.linalg.norm(grid_points[:, None] - ligand_coords, axis=2), axis=1)
            
            # Count "Empty" points (standard VdW buffers)
            valid_points = np.sum((prot_dists > 2.2) & (lig_dists > 1.2))
            volume = valid_points * (grid_size ** 3)
        else:
            volume = 0.0

        hotspots.append({
            "ligand_atom_index": l_index,
            "ligand_atom_type": l_type,
            "volume_A3": volume,
            "nearby_residues": list(set(nearby_residues))
        })

    # Sort by volume (largest growing space first)
    hotspots = sorted(hotspots, key=lambda x: x['volume_A3'], reverse=True)
    
    return hotspots[:max_hotspots]

# ---------------- Run Vina ----------------
def _affinity_from_pose(out_pdbqt):
    """
    The best mode's affinity, taken from the pose file Vina wrote.

    The log table used to be the only source. That ties every docking number
    in the paper to the `--log` flag, which AutoDock Vina removed in 1.2, and
    to a file that is not the one the rest of the pipeline reads. The
    `REMARK VINA RESULT` line is written into the output pdbqt by every Vina
    version, and it sits next to the coordinates downstream code uses, so the
    score cannot come from a stale log or from a different pose than the one
    being scored. Modes are written best-first, so the first such line is the
    best mode.
    """
    if not out_pdbqt or not os.path.exists(out_pdbqt):
        return None
    with open(out_pdbqt, 'r') as f:
        for line in f:
            if line.startswith('REMARK VINA RESULT:'):
                parts = line.split()
                try:
                    return float(parts[3])
                except (IndexError, ValueError):
                    return None
    return None


def _affinity_from_log(log_file):
    """The best mode's affinity from Vina's log table ('   1   -5.6  ...')."""
    if not log_file or not os.path.exists(log_file):
        return None
    with open(log_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) > 1 and parts[0] == "1":
                try:
                    return float(parts[1])
                except ValueError:
                    continue
    return None


def run_vina(receptor_pdbqt, ligand_pdbqt, center, size, ligand_name='ligand', save_dir='./temp'):
    out_pdbqt = os.path.join(save_dir, f"docked_{ligand_name}.pdbqt")
    log_file = os.path.join(save_dir, f"log_{ligand_name}.txt")

    cmd = [
        VINA_EXE,
        '--receptor', receptor_pdbqt,
        '--ligand', ligand_pdbqt,
        '--center_x', f"{center[0]:.3f}",
        '--center_y', f"{center[1]:.3f}",
        '--center_z', f"{center[2]:.3f}",
        '--size_x', f"{size[0]:.3f}",
        '--size_y', f"{size[1]:.3f}",
        '--size_z', f"{size[2]:.3f}",
        '--out', out_pdbqt,
        '--log', log_file,
        '--exhaustiveness', '8' # Increase to 16 or 32 for better blind docking results
    ]

    # Without a timeout a single oversized search space stalls the whole
    # pipeline; a capped box normally finishes in seconds.
    subprocess.run(cmd, check=True, capture_output=True, timeout=VINA_TIMEOUT)

    score = _affinity_from_pose(out_pdbqt)
    logged = _affinity_from_log(log_file)
    if score is None:
        score = logged
    elif logged is not None and abs(score - logged) > 0.05:
        # They are two reports of the same number; a disagreement means one of
        # the files is not from this run.
        print(f"vina: the pose file and the log disagree on the affinity for "
              f"{ligand_name} ({score:+.2f} vs {logged:+.2f} kcal/mol); "
              f"using the pose file.")
    # A docking score is a binding free energy: negative is favorable, and
    # drug-like binding runs roughly -4 to -14 kcal/mol. Vina reports a
    # POSITIVE affinity when it could not place the ligand without clashing,
    # which is a failure, not weak binding. Returning it as a number let a
    # failed docking be stored as data: a lead was recorded at +7.8, selected
    # as the best lead, and then used as the baseline for the binding gate,
    # where every candidate looked ~16 kcal/mol better and nothing could be
    # rejected. Refuse it here, at the only place the number is produced.
    if score is not None and score >= 0:
        print(f"vina: discarding a non-negative affinity ({score:+.1f} kcal/mol) "
              f"for {ligand_name}; no pose could be placed in this box.")
        score = None
    return score, out_pdbqt

def docked_pdbqt_to_pdb(pdbqt_path, output_pdb_path):
    """Converts Vina output to PDB to preserve docked coordinates."""
    with open(pdbqt_path, 'r') as f_in, open(output_pdb_path, 'w') as f_out:
        for line in f_in:
            if line.startswith("ENDMDL"): break 
            if line.startswith(("ATOM", "HETATM")):
                f_out.write(line[:66] + "\n")
    return output_pdb_path

# ---------------- Full Pipeline ----------------
def autodockvina_wo_ligand(cif_id, smiles, save_dir=None, sbdd=True,
                           contact_threshold=7.0, max_hotspots=5,
                           structure_path=None):
    # Every intermediate used to land in a fixed './temp' with fixed file
    # names (protein_receptor.pdbqt, ligand.pdbqt, log_ligand.txt). Two
    # docking calls running at once therefore overwrote each other's receptor
    # and ligand and then read each other's log, so a score could come from a
    # different molecule entirely. It was also relative to the working
    # directory. A private directory per call removes both problems.
    own_scratch = save_dir is None
    if own_scratch:
        save_dir = tempfile.mkdtemp(prefix='mollingo_dock_')
    os.makedirs(save_dir, exist_ok=True)
    try:
        return _dock_wo_ligand(cif_id, smiles, save_dir, sbdd,
                               contact_threshold, max_hotspots,
                               structure_path=structure_path)
    finally:
        if own_scratch:
            shutil.rmtree(save_dir, ignore_errors=True)


def _dock_wo_ligand(cif_id, smiles, save_dir, sbdd, contact_threshold,
                    max_hotspots, structure_path=None):
    # 1. Receptor: a local file when one is supplied (a predicted structure),
    #    otherwise fetched from RCSB by id.
    if structure_path:
        cif_path = structure_path
    else:
        cif_file = cif_id if cif_id.endswith('.cif') else f'{cif_id}.cif'
        cif_path = download_cif(cif_file, save_dir)
    receptor_pdb, center, size = prep_protein_only(cif_path, save_dir=save_dir)
    receptor_pdbqt = convert_to_pdbqt(receptor_pdb, is_receptor=True)

    if isinstance(smiles, str):
        # 2. Prep Ligand (SMILES -> PDBQT)
        # Added pdb_full_path to match the return signature of the previous logic
        _, ligand_pdbqt = prepare_ligand_pdb(smiles, 'ligand', save_dir=save_dir)
        
        # 3. Docking
        score, out_pdbqt = run_vina(receptor_pdbqt, ligand_pdbqt, center, size, save_dir=save_dir)
        
        if sbdd:

            # 4. Growth Analysis (Synchronized with autodockvina_w_ligand)
            # Convert docked PDBQT to PDB to keep the new coordinates
            docked_pdb = os.path.join(save_dir, 'docked_aligned.pdb')
            docked_pdbqt_to_pdb(out_pdbqt, docked_pdb)
            growth_analysis = identify_accurate_growth(receptor_pdb, docked_pdb, contact_threshold=contact_threshold, max_hotspots=max_hotspots)
            return {
                "docking_score": score,
                "growth_hotspots": growth_analysis
            }
        else:
            return score
    if isinstance(smiles, list):
        SCORE = []
        for sm in tqdm(smiles, desc='Docking'):
            try:
                _, ligand_pdbqt = prepare_ligand_pdb(sm, 'ligand', save_dir=save_dir)
                score, out_pdbqt = run_vina(receptor_pdbqt, ligand_pdbqt, center, size, save_dir=save_dir)
                SCORE.append([sm, score])
            except:
                pass
        return SCORE

if __name__ == "__main__":
    protein_id = '6XXC' 
    # SM = ['O=C(N/N=C/C1=CC=C2C=CC=CC2=C1)C3=CC=C(O)C=C3', 'O=C(N/N=C/c1ccccc1)c1ccc(F)cc1', 'O=C(N/N=C/c1ccccc1)c1ccc(I)cc1', 'CON=Cc1ccc(C(=O)N/N=C/c2ccccc2)cc1', 'COc1ccc(/C=N/NC(=O)c2ccc(O)cc2)cc1', 'O=C(N/N=C/c1ccc(F)cc1)c1cccc(O)c1', 'O=C(N/N=C/c1ccc(F)cc1)c1ccc(O)c(F)c1', 'O=C(N/N=C/c1ccccc1)c1ccc(O)c(F)c1']
    # test_smiles = SM[0]
    # SM = [""]
    SM = ["O=C(N/N=C/c1ccc(F)cc1)c1ccc(O)c(F)c1"]
    for test_smiles in SM:
        final_score = autodockvina_wo_ligand(protein_id, test_smiles)
        print(f"{test_smiles}: {final_score}")

