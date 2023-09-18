if __name__ == "__main__":

    import sys
    import os.path
    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), 
        os.path.pardir,
        os.path.pardir,
        )))

import os

import re

import numpy as np
import pandas as pd 
from scipy import sparse as sp

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from rdkit.Avalon import pyAvalonTools

try:
    from e3fp.conformer.generate import generate_conformers
    from e3fp.pipeline import fprints_from_mol, confs_from_smiles
except:
    pass 

from utils.io.io_utils import is_valid_smiles, read_smiles

def empty_feature_vector(
    n_bits: int,
    ):
    return sp.csr_matrix(False, shape=(1, n_bits))

def build_rdkit_molecule(
    smi: str, 
    perform_standardisation: bool = False,
    embed_mol: bool = False,
    required_num_conformers: int = 1,
    mol_props: dict = None,
    ):
    """Build an RDKit molecule using SMILES string `smi`.
    Optionally, standardise the molecule using the `standardiser` package.

    Parameters
    ----------
    smi : str
        The SMILES string describing the molecule.
    perform_standardisation : bool, optional
        Flag to standard the molecule with `standardiser`, by default True

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        The molecule generated from the SMILES string.
    """

    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, f"molecule for SMILES {smi} is None"
    
    # add mol_props, if given
    if mol_props is not None:
        for prop_name, prop_value in mol_props.items():
            mol.SetProp(prop_name, prop_value)
   
    # if perform_standardisation:
    #     try:
    #         from standardiser import standardise
    #         mol = standardise.run(mol)
    #     except Exception as e:
    #         # return mol
    #         pass 
   
    if embed_mol:
        # ensure hydrogen
        mol = Chem.AddHs(mol)

        num_conformers_found = 0
        seed = 0
        patience = 10
        patience_count = 0

        # increment seeds until enough conformers have been found 
        # upper limit on number of attempts
        while num_conformers_found < required_num_conformers and seed < max(100, 10 * required_num_conformers): 
            r  = AllChem.EmbedMolecule(
                mol, 
                useBasicKnowledge=True, 
                enforceChirality=True,
                randomSeed=seed,
                clearConfs=False, # keep the existing conformer(s)
            )
            # increment seed
            seed += 1
            patience_count += 1
            # increment conformer count
            if r != -1:
                num_conformers_found += 1
                patience_count = 0
            # check for number of seeds with no conformers
            if patience_count >= patience:
                # patience seeds have passed with no new conformers, break
                break
            
            # temporary print
            # if seed % 10 == 0:
            #     print (smi, "has reached seed", seed, "num conformers:", num_conformers_found, len(mol.GetConformers()))

        # optimise all found conformers
        if num_conformers_found > 0:
            AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=100_000) # seems to optimise kahalalide-F, which has a weight of ~1500Da

    return mol

def compute_fingerprints_multiple_molecules(
    smiles,
    compute_fingerprint_function,
    n_proc,
    verbose: bool = True,
    **compute_fingerprint_function_kwargs,
    ):

    if isinstance(smiles, list):
        num_smiles = len(smiles)
    else: 
        num_smiles = None

    if verbose:
        print ("Computing fingerprints for", num_smiles, "molecule(s) using", n_proc, "process(es)")
        print ("Submitting function with kwargs", compute_fingerprint_function_kwargs)

    if n_proc == 1: # for testing
        all_molecule_fingerprints = [ 
            compute_fingerprint_function(
                smi,
                **compute_fingerprint_function_kwargs,
            )
            for smi in smiles
        ]
    else:

        with ProcessPoolExecutor(max_workers=n_proc) as p:
            running_tasks = []
            for smi in smiles:
                task = p.submit(
                    compute_fingerprint_function,
                    smi,
                    **compute_fingerprint_function_kwargs,
                )
                running_tasks.append(task)
            all_molecule_fingerprints = []
            for running_task in running_tasks: # preseve order
                molecule_fingerprint = running_task.result()
                all_molecule_fingerprints.append(molecule_fingerprint)

    return all_molecule_fingerprints

# 3D fingerprints (e3fp)
def compute_e3fp_fingerprints_single_molecule(
    smi_molecule_id: tuple,
    n_bits: int = 1024, 
    num_conf: int = -1, # maximum number of conformers
    perform_standardisation: bool = False,
    combine_conformer_fingerprints: bool = True,
    ):

    # extra capacity
    n_bits *= 4

    # extract tuple
    smi, molecule_id = smi_molecule_id

    embed_using_rdkit = num_conf != -1

    if embed_using_rdkit:
        mol = build_rdkit_molecule(
            smi, 
            embed_mol=True, # generate the single 3D conformer required
            required_num_conformers=num_conf,
            perform_standardisation=perform_standardisation,
            mol_props={ # for some reason, e3fp requires the "_Name" to be filled 
                "_Name": molecule_id,
            }    
        )

    else: # use e3fp to select number of conformers
        try:
            mol = confs_from_smiles(
                smiles=smi, 
                name=molecule_id, 
                confgen_params={
                    # "seed": 0, # consistency -- this seems to generate the same conformer N times 
                    "num_conf": -1,
                    "pool_multiplier": 2, # from paper  
                    "rmsd_cutoff": 0.5, # from paper (determine different conformers)
                    "max_energy_diff": None, # from paper (difference between lowest energy conformer and other conformers)
                },
            ) 
        except Exception as e:
            print ("Failed to generate conformers using e3fp", e)
            mol = None 

    if mol is None or len(mol.GetConformers()) == 0: # fail if initial conformer cannot be generated
        # return mol # None
        print ("NO CONFORMERS", smi)
        return empty_feature_vector(n_bits)
    
    # generate conformers (seems to fail for some seeds)
    # generated_conformers = False
    # 10 retries, but no big deal if it fails since mol already has 1 3D conformer

    # molecule is constructed with 1 conformer
    # num_multi_conformer_generation_attempts = 10
    # if num_conf > 1 or num_conf == -1:
    #     for seed in range(num_multi_conformer_generation_attempts):
    #         conformers = generate_conformers(mol, num_conf=num_conf, seed=seed) # despite commment below, seeding is important for consistency
    #         # conformers = generate_conformers(mol, num_conf=num_conf, seed=-1) # seed=-1 seems to be more successful at getting multiple conformers
    #         if isinstance(conformers, bool):
    #             # TODO generate_conformers has failed
    #             # raise Exception(molecule_id, smi, num_conf)
    #             pass 
    #         else:
    #             # update mol with (potentially) multiple conformers
    #             mol, molname, num_rotatable_bonds, max_conformers, conf_indices, conf_energies, rmsds = conformers
    #             # generated_conformers = True
    #             break



    # assert len(mol.GetConformers()) > 0, (molecule_id, smi)
    # if len(mol.GetConformers()) == 0:
    #     print ("No conformers for", molecule_id)
    #     # return empty feature
    #     return empty_feature_vector(n_bits)

    try:
        # compute fingerprints (list)
        fingerprints = fprints_from_mol(mol, fprint_params={
            "first":-1,
            "stereo": True,
            "include_disconnected": True,
        })

        folded_fingerprints = sp.csr_matrix([ 
            fp.fold(bits=n_bits).to_rdkit() 
            for fp in fingerprints
        ], dtype=bool)

    except Exception as e:
        print ("e3fp fingerprint exception", e)
        folded_fingerprints = empty_feature_vector(n_bits)

    if combine_conformer_fingerprints:

        # logical or (any operation)
        return sp.csr_matrix(folded_fingerprints.sum(axis=0).astype(bool))
    
    # return fingerprints for all conformers
    return {
        f"{molecule_id}_{i}": folded_fingerprint
        for i, folded_fingerprint in enumerate(folded_fingerprints, start=1)
    }

  
def compute_e3fp_fingerprints_fingerprints(
    smiles: list, 
    molecule_ids: list,
    n_bits: int,
    num_conf: int = 10,
    combine_conformer_fingerprints: bool = True,
    perform_standardisation: bool = False, 
    n_proc: int = 8, 
    ):

    # zip smiles and molecule ids 
    smiles_molecule_ids = zip(smiles, molecule_ids)

    return compute_fingerprints_multiple_molecules(
        smiles_molecule_ids,
        compute_e3fp_fingerprints_single_molecule,
        n_proc=n_proc,
        n_bits=n_bits,
        num_conf=num_conf,
        perform_standardisation=perform_standardisation,
        combine_conformer_fingerprints=combine_conformer_fingerprints,
    )

# GOBBI Pharmacophore fingerprints
def compute_gobbi_2D_pharmacophore_fingerprint_single_molecule(
    smi: str, 
    perform_standardisation: bool = False,
    ):
    """Generate Gobbi fingerprint for a single molecule described by a SMILES string.

    Parameters
    ----------
    smi : str
        The SMILES string describing the molecule.
    perform_standardisation : bool, optional
        Flag to standard the molecule with `standardiser`, by default True

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of Gobbi fingerprints.
    """
    mol = build_rdkit_molecule(
        smi, 
        perform_standardisation=perform_standardisation)
    if mol is None: 
        # return mol
        n_bits = 20_000
        # TODO
        raise NotImplementedError
        return empty_feature_vector(n_bits)

    # assert mol.GetNumAtoms() > 0
    return sp.csr_matrix(  
        Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory), dtype=bool)

def compute_gobbi_2D_pharmacophore_fingerprints(
    smiles: list, 
    perform_standardisation: bool = False, 
    n_proc: int = 8, 
    ):
    """Generate Gobbi fingerprints for a list of molecules using multiprocessing.

    Parameters
    ----------
    smiles : list
        List of molecules.
    perform_standardisation : bool, optional
        Flag to standardise molecule, by default True
    n_proc : int, optional
        Number of processed, by default 8

    NOT CURRENTLY USED

    Returns
    -------
    list
        List of sparse fingerprints.
    """

    print ("computing Gobbi pharmacophore fingerprints for", len(smiles), "molecules")
    return compute_fingerprints_multiple_molecules(
        smiles,
        compute_gobbi_2D_pharmacophore_fingerprint_single_molecule,
        n_proc=n_proc,
        perform_standardisation=perform_standardisation,
        )

# RDKit MACCS implementation fingerprints
def rdk_maccs_fingerprint_single_molecule(
    smi: str, 
    perform_standardisation: bool = False,
    ):
    """Generate MACCS fingerprint for a single molecule described by a SMILES string.

    Parameters
    ----------
    smi : str
        The SMILES string describing the molecule.
    perform_standardisation : bool, optional
        Flag to standard the molecule with `standardiser`, by default True

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of MACCS fingerprints.
    """
    mol = build_rdkit_molecule(smi, perform_standardisation=perform_standardisation)
    
    if mol is None: 
        # return mol
        return empty_feature_vector(n_bits=167)
    # assert mol.GetNumAtoms() > 0
    return sp.csr_matrix(MACCSkeys.GenMACCSKeys(mol), dtype=bool)
    
def compute_rdk_maccs_fingerprint(
    smiles: list, 
    perform_standardisation: bool = False, 
    n_proc: int = 8, 
    ):
    """Generate MACCS fingerprints for a list of molecules using multiprocessing.

    Parameters
    ----------
    smiles : list
        List of molecules.
    perform_standardisation : bool, optional
        Flag to standardise molecule, by default True
    n_proc : int, optional
        Number of processed, by default 8

    Returns
    -------
    list
        List of sparse fingerprints.
    """

    print ("computing MACCS fingerprints for", len(smiles), "molecules")
    return compute_fingerprints_multiple_molecules(
        smiles,
        rdk_maccs_fingerprint_single_molecule,
        n_proc=n_proc,
        perform_standardisation=perform_standardisation,
        )

# atom pairs fingerprints
def atom_pairs_fingerprint_single_molecule(
    smi: str, 
    n_bits: int = 1024, 
    consider_chirality: bool = False,
    perform_standardisation: bool = False,
    ):
    """Generate atom pairs fingerprint for a single molecule described by a SMILES string.

    Parameters
    ----------
    smi : str
        The SMILES string describing the molecule.
    n_bits : int, optional
        Number of bits in fingerprint, by default 1024
    perform_standardisation : bool, optional
        Flag to standardise molecules, by default True

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of atom pairs fingerprints.
    """
    mol = build_rdkit_molecule(smi, perform_standardisation=perform_standardisation)
    if mol is None:
        # return mol
        return empty_feature_vector(n_bits)
    # assert mol.GetNumAtoms() > 0
    
    return sp.csr_matrix(
        rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect( mol , nBits=n_bits, includeChirality=consider_chirality, ), 
        dtype=bool)

def compute_atom_pairs_fingerprints(
    smiles: list, 
    n_bits: int = 1024,
    consider_chirality: bool = False,
    perform_standardisation: bool = False, 
    n_proc: int = 8, 
    ):
    """Generate atom pairs fingerprints for a list of molecules using multiprocessing.

    Parameters
    ----------
    smiles : list
        List of molecules.
    n_bits : int, optional
        Number of bits in fingerprint, by default 1024
    perform_standardisation : bool, optional
        Flag to standardise molecule, by default True
    n_proc : int, optional
        Number of processed, by default 8

    Returns
    -------
    list
        List of sparse fingerprints.
    """

    print ("computing atom pairs fingerprints for", len(smiles), "molecules")
    return compute_fingerprints_multiple_molecules(
        smiles,
        atom_pairs_fingerprint_single_molecule,
        n_proc=n_proc,
        n_bits=n_bits,
        consider_chirality=consider_chirality,
        perform_standardisation=perform_standardisation,
        )

# torsion 
def torsion_fingerprint_single_molecule(
    smi: str, 
    n_bits: int = 1024, 
    consider_chirality: bool = False,
    perform_standardisation: bool = False,
    ):
    """Generate torsion fingerprint for a single molecule described by a SMILES string.

    Parameters
    ----------
    smi : str
        The SMILES string describing the molecule.
    n_bits : int, optional
        Number of bits in fingerprint, by default 1024
    perform_standardisation : bool, optional
        Flag to standardise molecules, by default True

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of torsion fingerprints.
    """
    mol = build_rdkit_molecule(smi, perform_standardisation=perform_standardisation)
    
    if mol is None:
        # return mol
        return empty_feature_vector(n_bits)
    # assert mol.GetNumAtoms() > 0
    
    return sp.csr_matrix(
        rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect( mol , nBits=n_bits, includeChirality=consider_chirality ), 
        dtype=bool)


def compute_torsion_fingerprints(
    smiles: list, 
    n_bits: int = 1024,
    consider_chirality: bool = False,
    perform_standardisation: bool = False, 
    n_proc: int = 8, 
    ):
    """Generate torsion fingerprints for a list of molecules using multiprocessing.

    Parameters
    ----------
    smiles : list
        List of molecules.
    n_bits : int, optional
        Number of bits in fingerprint, by default 1024
    perform_standardisation : bool, optional
        Flag to standardise molecule, by default True
    n_proc : int, optional
        Number of processed, by default 8

    Returns
    -------
    list
        List of sparse fingerprints.
    """

    print ("computing torsion fingerprints for", len(smiles), "molecules")
    return compute_fingerprints_multiple_molecules(
        smiles,
        torsion_fingerprint_single_molecule,
        n_proc=n_proc,
        n_bits=n_bits,
        consider_chirality=consider_chirality,
        perform_standardisation=perform_standardisation,
        )

# avalon 
def avalon_fingerprint_single_molecule(
    smi: str, 
    n_bits: int = 1024, 
    perform_standardisation: bool = False,
    ):
    """Generate Avalon fingerprint for a single molecule described by a SMILES string.

    Parameters
    ----------
    smi : str
        The SMILES string describing the molecule.
    n_bits : int, optional
        Number of bits in fingerprint, by default 1024
    perform_standardisation : bool, optional
        Flag to standardise molecules, by default True

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of Avalon fingerprints.
    """
    mol = build_rdkit_molecule(smi, perform_standardisation=perform_standardisation)
    
    if mol is None:
        # return mol
        return empty_feature_vector(n_bits)
    
    # assert mol.GetNumAtoms() > 0
    
    return sp.csr_matrix(pyAvalonTools.GetAvalonFP( mol , nBits=n_bits, ), dtype=bool)

def calculate_avalon_fingerprints(
    smiles: list, 
    n_bits: int = 1024,
    perform_standardisation: bool = False, 
    n_proc: int = 8, 
    ):
    """Generate Avalon fingerprints for a list of molecules using multiprocessing.

    Parameters
    ----------
    smiles : list
        List of molecules.
    n_bits : int, optional
        Number of bits in fingerprint, by default 1024
    perform_standardisation : bool, optional
        Flag to standardise molecule, by default True
    n_proc : int, optional
        Number of processed, by default 8

    Returns
    -------
    list
        List of sparse fingerprints.
    """

    print ("computing Avalon fingerprints for", len(smiles), "molecules")
    return compute_fingerprints_multiple_molecules(
        smiles,
        avalon_fingerprint_single_molecule,
        n_proc=n_proc,
        n_bits=n_bits,
        perform_standardisation=perform_standardisation,
        )
  
# pattern
def pattern_fingerprint_single_molecule(
    smi: str, 
    n_bits: int = 1024, 
    perform_standardisation: bool = False,
    ):
    """Generate pattern fingerprint for a single molecule described by a SMILES string.

    Parameters
    ----------
    smi : str
        The SMILES string describing the molecule.
    n_bits : int, optional
        Number of bits in fingerprint, by default 1024
    perform_standardisation : bool, optional
        Flag to standardise molecules, by default True

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of pattern fingerprints.
    """
    mol = build_rdkit_molecule(smi, perform_standardisation=perform_standardisation)
    if mol is None:
        # return mol
        return empty_feature_vector(n_bits)
    
    # assert mol.GetNumAtoms() > 0
    
    return sp.csr_matrix(Chem.PatternFingerprint( mol , fpSize=n_bits, ), dtype=bool)

def calculate_pattern_fingerprints(
    smiles: list, 
    n_bits: int = 1024,
    perform_standardisation: bool = False, 
    n_proc: int = 8, 
    ):
    """Generate pattern fingerprints for a list of molecules using multiprocessing.

    Parameters
    ----------
    smiles : list
        List of molecules.
    n_bits : int, optional
        Number of bits in fingerprint, by default 1024
    perform_standardisation : bool, optional
        Flag to standardise molecule, by default True
    n_proc : int, optional
        Number of processed, by default 8

    Returns
    -------
    list
        List of sparse fingerprints.
    """

    print ("computing pattern fingerprints for", len(smiles), "molecules")
    return compute_fingerprints_multiple_molecules(
        smiles,
        pattern_fingerprint_single_molecule,
        n_proc=n_proc,
        n_bits=n_bits,
        perform_standardisation=perform_standardisation,
        )

# RDKit topological fingerprints
def rdk_topological_fingerprint_single_molecule(
    smi: str, 
    n_bits: int = 1024, 
    max_path: int = 5,
    perform_standardisation: bool = False,
    ):
    """Generate RDK fingerprint for a single molecule described by a SMILES string.

    Parameters
    ----------
    smi : str
        The SMILES string describing the molecule.
    n_bits : int, optional
        Number of bits in fingerprint, by default 1024
    max_path : int, optional
        Maximum atom path length to build fingerprint, by default 5
    perform_standardisation : bool, optional
        Flag to standardise molecules, by default True

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of RDK fingerprint.
    """
    mol = build_rdkit_molecule(smi, perform_standardisation=perform_standardisation)
    
    if mol is None:
        # return mol
        return empty_feature_vector(n_bits)
    
    # assert mol.GetNumAtoms() > 0
    return sp.csr_matrix(Chem.RDKFingerprint( mol , fpSize=n_bits, maxPath=max_path), dtype=bool)

def compute_rdk_topological_fingerpint(
    smiles: list, 
    n_bits: int = 1024,
    max_path: int = 5,
    perform_standardisation: bool = False, 
    n_proc: int = 8, 
    ):
    """Generate RDK fingerprints for a list of molecules using multiprocessing.

    Parameters
    ----------
    smiles : list
        List of molecules.
    n_bits : int, optional
        Number of bits in fingerprint, by default 1024
    max_path : int, optional
        Maximum atom path length to build fingerprint, by default 5
    perform_standardisation : bool, optional
        Flag to standardise molecule, by default True
    n_proc : int, optional
        Number of processed, by default 8

    Returns
    -------
    list
        List of sparse fingerprints.
    """
    print ("computing RDKI fingerprints for", len(smiles), "molecules using max_path =", max_path)
    return compute_fingerprints_multiple_molecules(
        smiles,
        rdk_topological_fingerprint_single_molecule,
        n_proc=n_proc,
        n_bits=n_bits,
        max_path=max_path,
        perform_standardisation=perform_standardisation,
        )

def morgan_fingerprint_single_molecule(
    smi: str, 
    radius: int, 
    n_bits: int = 1024,
    use_features: bool = True, # FCFP
    consider_chirality: bool = False,
    perform_standardisation: bool = False,
    as_sparse_matrix: bool = True,
    ):
    """Generate Morgan fingerprint for a single molecule described by a SMILES string.

    Parameters
    ----------
    smi : str
        The SMILES string describing the molecule.
    radius : int
        Radius to consider in fingerprint generation
    n_bits : int, optional
        Number of bits in fingerprint, by default 1024
    perform_standardisation : bool, optional
        Flag to standardise molecules, by default True
    as_sparse_matrix: bool, optional
        Flag to return as scipy matrix, else return as RDKit fingerprint, by default True

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of RDK fingerprint.
    """
    mol = build_rdkit_molecule(smi, perform_standardisation=perform_standardisation)
    
    if mol is None:
        # return mol
        return empty_feature_vector(n_bits)
    # assert mol.GetNumAtoms() > 0
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, 
        radius=radius, 
        nBits=n_bits,
        useFeatures=use_features, 
        useChirality=consider_chirality,
        )
    if not as_sparse_matrix:
        return fp 
    return sp.csr_matrix(fp, dtype=bool)

def compute_morgan_fingerprints(
    smiles: list, 
    radius: int,
    n_bits: int = 1024,
    use_features: bool = True, # FCFP
    consider_chirality: bool = False,
    perform_standardisation: bool = False, 
    n_proc: int = 8, 
    as_sparse_matrix: bool = True,
    ):
    """Generate Morgan fingerprints for a list of molecules using multiprocessing.

    Parameters
    ----------
    smiles : list
        List of molecules.
    n_bits : int, optional
        Number of bits in fingerprint, by default 1024
    max_path : int, optional
        Maximum atom path length to build fingerprint, by default 5
    perform_standardisation : bool, optional
        Flag to standardise molecule, by default True
    n_proc : int, optional
        Number of processed, by default 8
    as_sparse_matrix: bool, optional
        Flag to return as scipy matrix, else return as RDKit fingerprint, by default True
    Returns
    -------
    list
        List of sparse fingerprints.
    """
    return compute_fingerprints_multiple_molecules(
        smiles,
        morgan_fingerprint_single_molecule,
        n_proc=n_proc,
        n_bits=n_bits,
        radius=radius,
        use_features=use_features,
        consider_chirality=consider_chirality, 
        perform_standardisation=perform_standardisation,
        as_sparse_matrix=as_sparse_matrix,
        )

def compute_molecular_fingerprints(
    smiles: list, 
    fps: str, 
    n_bits: int = 1024, # default n_bits
    use_features: bool = True, # ECFP/FCFP
    consider_chirality: bool = False,
    perform_standardisation: bool = False,
    n_proc: int = 8,
    n_bits_delimiter: str = ":",
    smiles_key = "smiles",
    molecule_identifier_key = "molecule_id",
    verbose: bool = True,
    ):
    """Compute molecular fingerints described in fps for a list of molecules described by `smiles`.
    Multiple fingerprints may be concatenated using "," or "+", for example "morg2+rdk_maccs".
    
    Non-default numbers of bits can be specified after `n_bits_delimiter`, for example morg2:500
    will generate a 500-bit morg2 fingerprint.

    Parameters
    ----------
    smiles : list
        List of molecule SMILES.
    fps : str
        String descibing the fingerprint(s) to compute.
    n_bits : int, optional
        Default number of bits for each fingerprint., by default 1024
    n_proc : int, optional
        Number of processes to use in fingerprint generation, by default 8
    n_bits_delimiter : str, optional
        String delimiter to split fingerprint name and number of bits, by default ":"

    Returns
    -------
    scipy.sparse.csr_matrix 
        Sparse matrix containing molecule fingerprints.

    Raises
    ------
    NotImplementedError
        Thrown if fingerprint string is not recognised.
    """

  

    num_smiles = len(smiles)

    if verbose:
        print ("Computing", fps, "fingerprint(s) for", num_smiles, "molecule(s)")

    if len(smiles) == 0:
        print ("No SMILES!")
        return None

    # check for list of dicts input
    if isinstance(smiles[0], dict) and smiles_key in smiles[0] and molecule_identifier_key in smiles[0]:
        molecule_ids = [
            mol[molecule_identifier_key]
            for mol in smiles
        ]
        smiles = [ 
            mol[smiles_key]
            for mol in smiles
        ]
    else: # assume SMILES is a list of strings 
        molecule_ids = [f"molecule_{i}" for i in range(num_smiles)]

    all_computed_fps = []

    for fp in re.split(r"[,\+]", fps):

        if n_bits_delimiter in fp:
            fp, fp_n_bits = fp.split(n_bits_delimiter)
            fp_n_bits = int(fp_n_bits)
        else:
            fp_n_bits = n_bits

        if verbose:
            print ("Computing", fp, "fingerprints for", num_smiles, "SMILES", "using", n_proc, "process(es)")

        # if fp in {"morg1", "morg2", "morg3", }:
        if fp.startswith("morg"):
            radius = int(fp[-1]) # use final character as radius 
            computed_fp = compute_morgan_fingerprints(
                smiles, 
                radius=radius, 
                n_bits=fp_n_bits, 
                use_features=use_features,
                consider_chirality=consider_chirality,
                perform_standardisation=perform_standardisation, 
                n_proc=n_proc,
            )
        # must be before RDK!
        elif fp in {"maccs", "rdk_maccs"}:
            computed_fp = compute_rdk_maccs_fingerprint(
                smiles, 
                perform_standardisation=perform_standardisation, 
                n_proc=n_proc,
            )
        # elif fp in {"rdk5", "rdk6", "rdk7",}:
        elif fp.startswith("rdk"):
            max_path = int(fp[-1])
            computed_fp = compute_rdk_topological_fingerpint(
                smiles, 
                n_bits=fp_n_bits, 
                max_path=max_path,
                perform_standardisation=perform_standardisation, 
                n_proc=n_proc,
            )
        
        elif fp == "atom_pairs":
            computed_fp = compute_atom_pairs_fingerprints(
                smiles,
                n_bits=fp_n_bits,
                consider_chirality=consider_chirality,
                perform_standardisation=perform_standardisation,
                n_proc=n_proc,
            )
        elif fp == "torsion":
            computed_fp = compute_torsion_fingerprints(
                smiles, 
                n_bits=fp_n_bits,
                consider_chirality=consider_chirality,
                perform_standardisation=perform_standardisation,
                n_proc=n_proc,
            )
        elif fp == "avalon":
            computed_fp = calculate_avalon_fingerprints(
                smiles,
                n_bits=fp_n_bits,
                perform_standardisation=perform_standardisation,
                n_proc=n_proc,
            )
        elif fp == "pattern":
            computed_fp = calculate_pattern_fingerprints(
                smiles, 
                n_bits=fp_n_bits,
                perform_standardisation=perform_standardisation,
                n_proc=n_proc,
            )
        elif fp == "gobbi":
            computed_fp = compute_gobbi_2D_pharmacophore_fingerprints(
                smiles, 
                perform_standardisation=perform_standardisation,
                n_proc=n_proc,
            )
        elif fp.startswith("e3fp"): # e3fp*num_conf*
            num_conf = -1 # use e3fp-determined conformer number
            combine_conformer_fingerprints = True # combine into single fingerprint 
            if len(fp) > 4:
                try:
                    num_conf = int(fp[4:])
                except Exception as e:
                    pass 
            # n = 1000
            # smiles = smiles[:n]
            # molecule_ids = molecule_ids[:n]
            computed_fp = compute_e3fp_fingerprints_fingerprints(
                smiles=smiles,
                molecule_ids=molecule_ids, 
                n_bits=n_bits,
                num_conf=num_conf,
                combine_conformer_fingerprints=combine_conformer_fingerprints,
                perform_standardisation=perform_standardisation,
                n_proc=n_proc,
            )
        else:
            raise NotImplementedError

        if isinstance(computed_fp, list): # concatenate list of vectors to 2D sparse matrix
            computed_fp = sp.vstack(computed_fp)

        if verbose:
            print ("Completed computation of fingerprint:", fp,)

        all_computed_fps.append(computed_fp)
    
    return sp.hstack(all_computed_fps).tocsr() # concatenate all fps together


if __name__ == "__main__":

    # from dotenv import load_dotenv
    # load_dotenv()
    
    # from utils.queries.compound_queries import get_properties_of_multiple_natural_products_query

    # smiles = get_properties_of_multiple_natural_products_query(
    #     columns=("smiles",),
    #     name_like="gink",
    #     all_less_than=[ 
    #         ("molecular_weight", 500),
    #     ],
    #     max_records=1000,
    # )


    # clusters = perform_molecule_clustering(
    #     smiles,
    #     max_clusters=None,
    #     n_proc=1)

    # for c in clusters:
    #     print(c, len(clusters[c]))

    # smiles = pd.Series(
    #     [
    #         "CCS(=O)(=O)NC1=CC2=C(C=C1)NC(=O)C2=C(C3=CC=CC=C3)NC4=CC=C(C=C4)CN5CCCCC5",
    #         "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    #         "CC(=O)OC1=CC=CC=C1C(=O)O",
    #     ],
    #     index=[
    #         "Hesperadin",
    #         "Ibuprofen",
    #         "aspirin",
    #     ],
    # )


    # print (compute_e3fp_fingerprints_fingerprints(
    #     smiles=smiles.values,
    #     molecule_ids=smiles.index,
    #     n_bits=1024,
    #     num_conf=10,
    #     perform_standardisation=False,
    #     n_proc=10,
    # ))

    smiles = read_smiles(
        "test_compounds/aspirin.smi",
        assume_clean_input=True,
        return_list=True,
        verbose=True,
    )

    # print (compute_atom_pairs_fingerprints(
    #     smiles=smiles,
    #     n_bits=1024,
    #     consider_chirality=True,
    # ))
    # print (compute_atom_pairs_fingerprints(
    #     smiles=smiles,
    #     n_bits=1024,
    #     consider_chirality=False,
    # ))

    # # fps = get_morgan_fingerprints(
    # #     smiles=smiles,
    # #     n_proc=5)


    # # print (compute_molecular_fingerprints(smiles, fps="gobbi", n_proc=1))

    computed_fps = compute_molecular_fingerprints(
        smiles, 
        # fps="rdk_maccs",
        # fps="atom_pairs:100",
        # fps="rdk7:1000",
        fps="e3fp10",
        # fps="rdk_maccs,morg2,gobbi",
        # fps="rdk_maccs,rdk5:10000+morg3:100",
        # fps="morg3+rdk_maccs+torsion",
        # fps="morg1+rdk_maccs+atom_pairs+torsion+avalon",
        # fps="atom_pairs",
        # fps="rdk5",
        # fps="rdk5:2048",
        perform_standardisation=False,
    )

    print (computed_fps.shape)

    # mol = build_rdkit_molecule(
    #     smi="CC(=O)OC1=CC=CC=C1C(=O)O",
    #     mol_props={
    #         "_Name": "aspirin",
    #     }
    # )

    # print (mol.GetProp("_Name"))
