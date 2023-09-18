if __name__ == "__main__":

    import sys
    import os.path

    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import os

import numpy as np
import pandas as pd

from scipy.sparse import load_npz, save_npz, csr_matrix, lil_matrix
from scipy.stats import rankdata

from bioactivity_predictor.models import BioactivityPredictor, load_model, canonise_fp

from utils.molecules.chembl_utils import (
    ALL_CHEMBL_ORGANISM_GROUPS, 
    ALL_CHEMBL_TARGET_TYPES, 
    ALL_CHEMBL_TARGETS,
    get_chembl_organism_group, 
    )
from utils.molecules.rdkit_utils import BRICS_decompose_smiles_using_RDKit
from utils.io.io_utils import (
    read_smiles, 
    sanitise_filename,
    write_json, 
    load_json, 
    )

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

ALL_CHEMBL_TARGET_TO_ID_FILENAME = "data/databases/chembl/target_to_id.json"
ALL_CHEMBL_TARGET_TO_ID = load_json(ALL_CHEMBL_TARGET_TO_ID_FILENAME)

ALL_ENTREZ_TARGETS_FILENAME = "data/databases/entrez/all_entrez_targets.json"
ALL_ENTREZ_TARGETS = load_json(ALL_ENTREZ_TARGETS_FILENAME, verbose=False)
# convert to map from target_chembl_id to target data
ALL_ENTREZ_TARGETS = {
    record["target_chembl_id"]: record
    for record in ALL_ENTREZ_TARGETS
    if record["target_chembl_id"] is not None
}

DEFAULT_MODEL_ROOT = "C:/KNIME/basic/bioactivity_predictor/models_chembl_31"

PRELOADED_MODELS = {}

preload_bioactivity_prediction_models = os.environ.get("PRELOAD_MODELS", default="false") == "true"
if preload_bioactivity_prediction_models:

    print ("Preloading bioactivity prediction model(s)")

    num_loaded = 0

    for chirality in (
        "no_chirality",
        "chirality",
    ):

        chirality_sanitised = sanitise_filename(chirality)
        if chirality not in PRELOADED_MODELS:
            PRELOADED_MODELS[chirality] = {}

        for target_type in {
            "SINGLE PROTEIN", 
            # "CELL-LINE",
            }:
            target_type_sanitised = sanitise_filename(target_type)
            if target_type not in PRELOADED_MODELS[chirality]:
                PRELOADED_MODELS[chirality][target_type] = {}

            for organism_group in {
                "Homo sapiens",
                "Selected animals",
            }:
                organism_group_sanitised = sanitise_filename(organism_group)
                if organism_group not in PRELOADED_MODELS[chirality][target_type]:
                    PRELOADED_MODELS[chirality][target_type][organism_group] = {}

                # iterate over seeds
                for seed in range(1):

                    if seed not in PRELOADED_MODELS[chirality][target_type][organism_group]:
                        PRELOADED_MODELS[chirality][target_type][organism_group][seed] = {}

                    for max_actives in {
                        1000,
                        }:
                        if max_actives not in PRELOADED_MODELS[chirality][target_type][organism_group][seed]:
                            PRELOADED_MODELS[chirality][target_type][organism_group][seed][max_actives] = {}
                        
                        for fp in {
                            "morg3+rdk_maccs+torsion",
                            }:
                            fp = canonise_fp(fp)
                            model_filename = os.path.join(
                                DEFAULT_MODEL_ROOT,
                                chirality_sanitised,
                                target_type_sanitised,
                                organism_group_sanitised,
                                f"{seed:02d}",
                                f"max_actives={max_actives}",
                                f"{fp}-nn.pkl.gz",
                                )
                            if os.path.exists(model_filename):
                                PRELOADED_MODELS[chirality][target_type][organism_group][seed][max_actives][fp] = load_model(
                                    model_filename=model_filename,
                                    verbose=True,
                                    )
                                num_loaded += 1

    print ("Preloaded", num_loaded, "bioactivity prediction model(s)")

def parse_SMILES_as_input_for_bioactivity_prediction_model(
    supplied_mols,
    assume_clean_input: bool,
    smiles_key: str,
    molecule_identifier_key: str,
    brics_decompose: bool = False,
    verbose: bool = False,
    ):
    """Helper function to convert molecules to a Series

    Parameters
    ----------
    supplied_mols : str
        May be a string filepath, list of dictionaries or an existing Series 
    assume_clean_input : bool
        Flag to assume input is well-formed with a delimiter
    smiles_key : str
        Key of SMILES values
    molecule_identifier_key : str
        Key of molecule id
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    pd.Series
        An instance of Pandas Series containing the SMILES, indexed with molecule IDs
    """

    if verbose:
        print ("Parsing molecules as a Pandas Series")
   
    if isinstance(supplied_mols, pd.Series):
        return supplied_mols

    # load SMILES
    if isinstance(supplied_mols, str): # read from file
    
        # read (and filter) smiles
        # convert to list of dicts
        supplied_mols = read_smiles(
            supplied_mols,
            remove_invalid_molecules=True, 
            return_list=True,
            assume_clean_input=assume_clean_input,
            molecule_identifier_key=molecule_identifier_key,
            smiles_key=smiles_key,
            verbose=verbose,
            )

    if verbose:
        print ("Parsing molecules from list of dictionaries")
        print ("Number of molecules:", len(supplied_mols))
    
    
    if brics_decompose:

        supplied_mols = BRICS_decompose_smiles_using_RDKit(
            smiles=supplied_mols,
            keep_original_molecule_in_output=True,
            keep_non_leaf_nodes=True,
            smiles_key=smiles_key,
            molecule_identifier_key=molecule_identifier_key,
            verbose=verbose,
        )

    # JSON object - convert to dict and then Series
    # supplied_mols = {
    #     supplied_mol[molecule_identifier_key]: supplied_mol[smiles_key]
    #     for supplied_mol in supplied_mols
    # }

    # convert to series, now not required
    # supplied_mols = pd.Series(supplied_mols)

    return supplied_mols

def load_and_configure_model(
    model_root_dir: str = DEFAULT_MODEL_ROOT,
    consider_chirality: bool = False,
    target_type: str = None,
    organism_group: str = None,
    seed: int = 0,
    max_actives: int = None,
    fp: str = None,
    model: BioactivityPredictor = None,
    model_name: str = "nn+var(nb)",
    k: int = 500,
    alpha: float = 1e-0,
    cv: float = np.inf,
    num_similar_mols_to_show: bool = 10,
    n_proc: int = 1,
    verbose: bool=True,
    ):

    if verbose:
        print ("Loading and configuring model")

    # handle default model root
    if model_root_dir is None:
        model_root_dir = DEFAULT_MODEL_ROOT

    # canonise fp (if provided)
    fp = canonise_fp(fp)

    if consider_chirality:
        consider_chirality = "chirality"
    else: # consider_chirality = False
        consider_chirality = "no_chirality"

    if model is None:

        # flag to check for preloaded models
        use_preloaded = model_root_dir == DEFAULT_MODEL_ROOT

        # fix seed 
        if use_preloaded:
            seed = 0

        # handle model loading 
        if consider_chirality is None:
            return None 
        chirality_sanitisied = sanitise_filename(consider_chirality)
        if target_type is None:
            return None 
        target_type_sanitised = sanitise_filename(target_type)
        if organism_group is None:
            return None 
        organism_group_sanitised = sanitise_filename(organism_group) 

        # check if model is pre-loaded, else load it
        if use_preloaded and consider_chirality in PRELOADED_MODELS \
            and target_type in PRELOADED_MODELS[consider_chirality] \
            and organism_group in PRELOADED_MODELS[consider_chirality][target_type] \
            and seed in PRELOADED_MODELS[consider_chirality][target_type][organism_group] \
            and max_actives in PRELOADED_MODELS[consider_chirality][target_type][organism_group][seed] \
            and fp in PRELOADED_MODELS[consider_chirality][target_type][organism_group][seed][max_actives]:

            if verbose:
                print ("Model is preloaded")

            model = PRELOADED_MODELS[consider_chirality][target_type][organism_group][seed][max_actives][fp]
        
        else:

            model_filename = os.path.join(
                model_root_dir,
                chirality_sanitisied,
                target_type_sanitised,
                organism_group_sanitised, 
                f"{seed:02d}",
                f"max_actives={max_actives}",
                f"{fp}-nn.pkl.gz")

            if verbose:
                print ("Model is not preloaded, attempting to load from", model_filename)

            if not os.path.exists(model_filename): # likely because there were no molecules in the training set
                if verbose:
                    print (model_filename, "does not exist, skipping prediction")
                return None

            model = load_model(
                model_filename=model_filename,
                verbose=verbose,
                )

    
    elif verbose:
        print ("Model is already loaded")
    
    if verbose:
        print ("Begin model configuration")

    if hasattr(model, "n_proc"):
        if verbose: 
            print ("Setting n_proc to", n_proc)
        model.set_n_proc(n_proc)

    if verbose:
        print ("Setting model_name to", model_name)
    return model
    model.model_name = model_name
    
    if verbose:
        print("Setting k to", k)
    model.set_k(k)

    if verbose:
        print ("Setting num_similar_mols_to_show to", num_similar_mols_to_show)
    model.num_similar_mols_to_show = num_similar_mols_to_show

    if verbose:
        print ("Setting alpha to", alpha)
    model.alpha = alpha 

    if verbose:
        print ("Setting cv to", cv)
    model.cv = cv

    return model

def perform_bioactivity_prediction_with_single_model(
    supplied_mols: str,
    
    model_root_dir: str = None, # default
    consider_chirality: bool = False,
    target_type: str = None,
    organism_group: str = None,
    seed: int = 0,
    max_actives: int = None,
    fp: str = None,

    model: BioactivityPredictor = None, # pass in preloaded model
    model_name: str = "nn+var(nb)",
    k: int = 500,
    alpha: float = 1e-0,
    cv: float = np.inf,

    max_target_rank: int = 100,
    min_probability: float = 0,

    return_similar_mols: bool = False,
    num_similar_mols_to_show: bool = 10,
    targets_of_interest: list = None,
    assume_clean_input: bool = False,
    return_chembl_matches: bool = True,
    return_full_target_data: bool = True,
    smiles_key: str = "smiles",
    molecule_identifier_key: str = "molecule_id",
    brics_decompose: bool = False,
    precomputed_query_fingerprint_path: str = None,
    pickled_nearest_neighbours_path: str = None,
    n_proc: int = 1,

    output_format = "json",

    verbose: bool = True,
    ):

    output_format = output_format.lower()

    if output_format not in {"json", "npz", "parquet"}:
        output_format = "json"

    # load model and set model parameters
    model = load_and_configure_model(
        # args to load default model if no preloaded model is given
        model_root_dir=model_root_dir,
        consider_chirality=consider_chirality,
        target_type=target_type,
        organism_group=organism_group,
        seed=seed,
        max_actives=max_actives,
        fp=fp,      
        # allow preloaded model  
        model=model,
        # configure args
        model_name=model_name,
        k=k,
        alpha=alpha,
        cv=cv,
        num_similar_mols_to_show=num_similar_mols_to_show,
        n_proc=n_proc,
        verbose=verbose,
    )
    if model is None:
        print ("Model is None, returning None")
        return None 

    # parse supplied mols
    supplied_mols = parse_SMILES_as_input_for_bioactivity_prediction_model(
        supplied_mols,
        assume_clean_input=assume_clean_input,
        brics_decompose=brics_decompose,
        smiles_key=smiles_key,
        molecule_identifier_key=molecule_identifier_key,
        verbose=verbose,
        )
    
    # supplied_mols = supplied_mols[:10000]
    
    if verbose:
        print ("Performing bioactivity prediction", )
        print ("Predicting for", len(supplied_mols), "molecules")


    predicted_probabilities = model.predict_proba(
        supplied_mols, 
        return_similar_mols=return_similar_mols,
        precomputed_query_fp_path=precomputed_query_fingerprint_path,
        precomputed_nn_path=pickled_nearest_neighbours_path,
        return_chembl_matches=return_chembl_matches,
        verbose=verbose,
    )

    if isinstance(predicted_probabilities, tuple): # return_similar_mols split tuple
        assert return_similar_mols
        predicted_probabilities, target_type_most_similar_mols = predicted_probabilities

    # predicted_probabilites is csr_matrix

    assert hasattr(model, "id_to_target") # map from id to full target data
    id_to_target = model.id_to_target
    n_targets = len(id_to_target)

    if verbose:
        print ("Returning predictions in", output_format, "format")

    # handle sparse matrix predictions (for evaluation only)
    if output_format == "npz":

        # reformat for all targets 
        predicted_probabilities_all_targets = lil_matrix((len(supplied_mols), len(ALL_CHEMBL_TARGET_TO_ID)))

        if verbose:
            print ("Returning as sparse matrix of shape", predicted_probabilities_all_targets.shape)
        
        # add columns for predicted targets
        for i, target_data in id_to_target.items():

            target_chembl_id = target_data["target_chembl_id"]
            
            predicted_probabilities_all_targets_index = ALL_CHEMBL_TARGET_TO_ID[target_chembl_id]

            predicted_probabilities_all_targets[:, predicted_probabilities_all_targets_index] = predicted_probabilities[:,int(i)]

        # return as sparse probability matrix (n_compounds x ALL n_targets)
        # return predicted_probabilities
        return csr_matrix(predicted_probabilities_all_targets)
    
    elif output_format == "parquet":
        
        columns = [id_to_target[str(i)]["target_chembl_id"] for i in range(n_targets)]
        index = [supplied_mol[molecule_identifier_key] for supplied_mol in supplied_mols]

        # cannot write this to file annoyingly
        # return pd.DataFrame.sparse.from_spmatrix(predicted_probabilities, index=index, columns=columns)
        return pd.DataFrame(predicted_probabilities.A, index=index, columns=columns)

    elif output_format == "json":

        if verbose:
            print ("Constructing a list of dictionaries of hits of rank <=", max_target_rank, "probability >", min_probability)

        # indexed by molecule identifier
        all_molecule_predicted_probabilities_dict = dict()

        # iterate over molecule_indexes, molecule_ids, and probabily vectors
        for mol_num, (molecule_data, molecule_predicted_probabilities) in enumerate(
            zip(supplied_mols, predicted_probabilities)
            ):

            # extract molecule_id from molecule_data
            molecule_id = molecule_data[molecule_identifier_key]

            if not isinstance(molecule_predicted_probabilities, np.ndarray):
                molecule_predicted_probabilities = molecule_predicted_probabilities.A.flatten()

            # conpute confidence score (relative to all organisms)
            ranks_all_targets = rankdata(-molecule_predicted_probabilities, method="dense").astype(int) # largest to smallest
            molecule_max_rank = ranks_all_targets.max()
            # convert rank into confidence score with linspace
            confidence_scores = ( (molecule_max_rank - ranks_all_targets) / (molecule_max_rank - 1) * 1000).astype(int)

            # iterate over all targets 
            molecule_predicted_probabilities_as_list = []
            for target_id in range(n_targets):

                target_chembl_id = id_to_target[str(target_id)]["target_chembl_id"]

                rank = int(ranks_all_targets[target_id])
                probability = float(molecule_predicted_probabilities[target_id])
                confidence_score = int(confidence_scores[target_id])

                # skip target based on rank
                if max_target_rank is not None and rank > max_target_rank:
                    continue
                # skip target based on probability 
                if probability == 0:
                    continue # not predicted
                if min_probability is not None and probability < min_probability: 
                    continue
                # skip target if not in targets of interest 
                if targets_of_interest is not None and target_chembl_id not in targets_of_interest:
                    continue

                current_target_data = {
                    "target_chembl_id": target_chembl_id,
                }
                
                # add full target data if required
                if return_full_target_data and target_chembl_id in ALL_ENTREZ_TARGETS:

                    current_target_data.update(
                        ALL_ENTREZ_TARGETS[target_chembl_id]
                    )
                    
                # add rank
                current_target_data.update(
                    {
                        "probability": probability,
                        "rank": rank,
                        "confidence_score": confidence_score,
                    }
                )

                molecule_predicted_probabilities_as_list.append(current_target_data)

            # sort by rank
            molecule_predicted_probabilities_as_list = sorted(
                molecule_predicted_probabilities_as_list,
                key=lambda record: record["rank"])


            all_molecule_predicted_probabilities_dict[molecule_id] = molecule_predicted_probabilities_as_list

        return all_molecule_predicted_probabilities_dict
    
    else:
        raise NotImplementedError


def test():
    model_root_dir = "C:/KNIME/basic/bioactivity_predictor/models_chembl_31"

    consider_chirality = False
    # consider_chirality = True

    target_type = "SINGLE PROTEIN"
    # target_type = "PROTEIN-PROTEIN INTERACTION"
    # target_type = "PROTEIN COMPLEX"

    organism_group = "Homo sapiens"

    n_proc = 20

    molecules = [
        {
            "molecule_id": "aspirin",
            "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        },
        {
            "molecule_id": "ibuprofen",
            "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        }
    ]

    results = perform_bioactivity_prediction_with_single_model(
        supplied_mols=molecules,
        model_root_dir=model_root_dir,
        consider_chirality=consider_chirality,
        target_type=target_type,
        organism_group=organism_group,
        seed=0, # fixed - must be 0
        max_actives=1000, # fixed - must be 1000,
        fp="rdk_maccs+torsion+morg3", # fixed
        model_name="nn+var(nb",
        k=2000, # fixed
        alpha=1, # fixed
        cv=float("inf"), # fixed
        max_target_rank=100, # can be changed 
        n_proc=n_proc,
        verbose=True,
    )
    output_dir = "c:/results"
    os.makedirs(output_dir, exist_ok=True)


    # to dataframe 
    for molecule_id, molecule_results in results.items():

        molecule_output_filename = os.path.join(output_dir, f"{molecule_id}.csv")
        print ("Writing results for molecule", molecule_id, "to file", molecule_output_filename)
        molecule_results = pd.DataFrame(molecule_results)

        molecule_results.to_csv(molecule_output_filename)
  
