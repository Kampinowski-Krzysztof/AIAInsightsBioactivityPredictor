if __name__ == "__main__":

    import sys
    import os.path
    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), 
        os.path.pardir,
        os.path.pardir,
        )))

import os

from scipy.sparse import csr_matrix, save_npz
from scipy.stats import rankdata

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt

from utils.io.io_utils import load_json, write_json, write_smiles
from utils.molecules.rdkit_utils import diversity_molecule_selection

# will be empty if file does not exist
ALL_CHEMBL_TARGETS_FILENAME = "data/databases/chembl/all_targets.json"
ALL_CHEMBL_TARGETS = load_json(ALL_CHEMBL_TARGETS_FILENAME, verbose=False)


ALL_CHEMBL_TARGET_TYPES = ( # commented out target_types have no actives
    "SINGLE PROTEIN",  
    # "ORGANISM",               
    # "METAL",                       
    "PROTEIN-PROTEIN INTERACTION",
    "SELECTIVITY GROUP",           
    # "SMALL MOLECULE",              
    # "MACROMOLECULE",               
    "PROTEIN COMPLEX GROUP",       
    "CHIMERIC PROTEIN",            
    "PROTEIN COMPLEX",             
    # "OLIGOSACCHARIDE",             
    "PROTEIN FAMILY",   
    "CELL-LINE",         
)

ALL_CHEMBL_ORGANISM_GROUPS = (
    "Homo sapiens",
    "Selected animals",
    "Other organisms"
)

CHUNKSIZE = 100

DESIRED_MOLECULE_FIELDS = {
    "atc_classifications",
    "availability_type",
    "biotherapeutic",
    "black_box_warning",
    "chebi_par_id",
    "chirality",
    # "cross_references",
    "dosed_ingredient",
    "first_approval",
    "first_in_class",
    "helm_notation",
    "indication_class",
    "inorganic_flag",
    "max_phase",
    "molecule_chembl_id",
    "molecule_hierarchy",
    "molecule_properties",
    "molecule_structures",
    "molecule_synonyms",
    "molecule_type",
    "natural_product",
    "oral",
    "parenteral",
    "polymer_flag",
    "pref_name",
    "prodrug",
    "structure_type",
    "therapeutic_flag",
    "topical",
    "usan_stem",
    "usan_stem_definition",
    "usan_substem",
    "usan_year",
    "withdrawn_class",
    "withdrawn_country",
    "withdrawn_flag",
    "withdrawn_reason",
    "withdrawn_year",
}
DESIRED_ASSAY_FIELDS = (
    'target_chembl_id',
    'molecule_chembl_id', 
    'canonical_smiles', 
    'activity_comment', 
    # 'activity_id', 
    # 'activity_properties', 
    'assay_chembl_id', 
    # 'assay_description', 
    'assay_type', 
    # 'assay_variant_accession', 
    # 'assay_variant_mutation', 
    # 'bao_endpoint', 
    # 'bao_format', 
    # 'bao_label', 
    # 'data_validity_comment', 
    # 'data_validity_description', 
    'document_chembl_id', 
    # 'document_journal', 
    # 'document_year', 
    'ligand_efficiency', 
    # 'molecule_pref_name', 
    # 'parent_molecule_chembl_id', 
    'pchembl_value', 
    # 'potential_duplicate', 
    # 'qudt_units', 
    # 'record_id', 
    'relation',
    # 'src_id', 
    # 'standard_flag', 
    'standard_relation', 
    'standard_text_value', 
    'standard_type', 
    'standard_units', 
    'standard_upper_value', 
    'standard_value', 
    # 'target_organism', 
    # 'target_pref_name', 
    # 'target_tax_id', 
    # 'text_value', 
    # 'toid', 
    'type', 
    # 'units', 
    # 'uo_units', 
    # 'upper_value', 
    # 'value',
)
DESIRED_DOCUMENT_FIELDS = (
    'abstract', 
    'authors', 
    'doc_type', 
    # 'document_chembl_id', 
    # 'doi', 
    # 'doi_chembl', 
    # 'first_page', 
    # 'issue', 
    'journal', 
    'journal_full_title', 
    # 'last_page', 
    # 'patent_id', 
    'pubmed_id',
    # 'src_id', 
    'title', 
    'volume', 
    'year'
)

selected_animals_filename = os.path.join("data", "databases", "chembl", "selected_animals.txt")
SELECTED_ANIMALS = set()
if os.path.exists(selected_animals_filename):
    with open(selected_animals_filename, "r") as f:
        SELECTED_ANIMALS = set(map(str.strip, f.readlines()))

def get_chembl_organism_group(
    organism: str,
    ):
    if organism == "Homo sapiens":
        return ALL_CHEMBL_ORGANISM_GROUPS[0] # Homo sapiens
    elif organism=="Animals" or organism in SELECTED_ANIMALS: #compatability with animals 
        return ALL_CHEMBL_ORGANISM_GROUPS[1] # Selected animals
    else:
        return ALL_CHEMBL_ORGANISM_GROUPS[-1] # Other organisms

def match_chembl_molecules_using_similar_smiles(
    smiles: str, 
    min_similarity: int = 85,
    ):
    """Search CHEMBL for molecules with a Tanimoto similarity value greater than or equal to `min_similarity`
    to the supplied `smiles`.

    Parameters
    ----------
    smiles : str
        The SMILES string of the query molecule
    min_similarity : int, optional
        The minumim Tanimoto similarity to the supplied query molecule, by default 85

    Returns
    -------
    list
        The matching molecules from CHEMBL
    """
    try:
        from chembl_webresource_client.new_client import new_client
    except Exception as e:
        ret = []
        print ("chembl_webresource exception", e, "returning", ret)
        return ret
    similarity = new_client.similarity
    res = similarity.filter(
        smiles=smiles, 
        similarity=min_similarity)
    try:
        return list(res)
    except Exception as e:
        ret = []
        print ("chembl_webresource exception", e, "returning", ret)
        return ret

def match_chembl_molecules_by_inchikey(
    inchikeys: list,
    verbose: bool = True,
    ):
    """Search CHEMBL for molecules that match the provided `inchikeys`.

    Parameters
    ----------
    inchikey : list
        The INCHIKEY(s) to match

    Returns
    -------
    list
        Matching record(s) from CHEMBL, if any are found
    """
    try:
        from chembl_webresource_client.new_client import new_client
    except Exception as e:
        ret = []
        print ("chembl_webresource exception", e, "returning", ret)
        return ret
    molecule = new_client.molecule
    if isinstance(inchikeys, set):
        inchikeys = list(inchikeys)
    elif isinstance(inchikeys, str):
        inchikeys = [inchikeys]
    num_inchikeys = len(inchikeys)
    n_chunks = num_inchikeys // CHUNKSIZE +1

    # function return 
    all_retrieved_mols = []
    for chunk_num in range(n_chunks):
        chunk_inchikeys = inchikeys[chunk_num*CHUNKSIZE: (chunk_num+1)*CHUNKSIZE]
        if len(chunk_inchikeys) == 0:
            continue
        elif len(chunk_inchikeys) == 1:
            chunk_inchikeys = chunk_inchikeys[0] # convert to str
        try:
            if verbose:
                print ("Retreiving data about molecules from CHEMBL:", chunk_inchikeys)
            chunk_retrieved_mols = molecule.get(chunk_inchikeys)
            if isinstance(chunk_retrieved_mols, dict):
                chunk_retrieved_mols = [chunk_retrieved_mols]
            else:
                # force error
                chunk_retrieved_mols = list(chunk_retrieved_mols)
            all_retrieved_mols.extend(chunk_retrieved_mols)
        except Exception as e:
            print ("chembl_webresource exception", e, )

    return all_retrieved_mols

def match_chembl_smiles_by_smiles(
    smiles: str, 
    ):
    """Search CHEMBL for molecules using a SMILES string.

    Parameters
    ----------
    smiles : str
        The SMILES string to search with

    Returns
    -------
    list
        List of matching molecules, if any
    """
    try:
        from chembl_webresource_client.new_client import new_client
    except Exception as e:
        ret = []
        print ("chembl_webresource exception", e, "returning", ret)
        return ret
    molecule = new_client.molecule
    try:
        matching_molcules =  molecule.filter(molecule_structures__canonical_smiles__flexmatch=smiles)
        return list(matching_molcules)
    except Exception as e:
        ret = []
        print ("Match molecules using CHEMBL SMILES exception", e, "returning", ret)
        return ret


def query_chembl_for_publication_data(
    document_chembl_ids: list,
    verbose: bool = True,
    ):
    """Search CHEMBL for documents whose CHEMBL IDs are in `document_chembl_ids`.

    Parameters
    ----------
    document_chembl_ids : list
        The document_chembl_ids(s) to search for

    Returns
    -------
    list
        Matching record(s) from CHEMBL, if any are found
    """
    try:
        from chembl_webresource_client.new_client import new_client
    except Exception as e:
        ret = []
        print ("chembl_webresource exception", e, "returning", ret)
        return ret
    documents = new_client.document

    

    if isinstance(document_chembl_ids, set):
        document_chembl_ids = list(document_chembl_ids)
    if isinstance(document_chembl_ids, str):
        document_chembl_ids = [document_chembl_ids]
    # if isinstance(document_chembl_ids, list) and len(document_chembl_ids) == 1:
    #     document_chembl_ids = document_chembl_ids[0]


    # function return
    all_found_documents = []
    
    # shuld be a list now
    num_document_chembl_ids = len(document_chembl_ids)

    n_chunks = num_document_chembl_ids // CHUNKSIZE + 1

    for chunk_num in range(n_chunks):
        chunk_document_chembl_ids = document_chembl_ids[chunk_num*CHUNKSIZE:(chunk_num+1)*CHUNKSIZE]
        if len(chunk_document_chembl_ids) == 0:
            continue 
        elif len(chunk_document_chembl_ids) == 1:
            chunk_document_chembl_ids = chunk_document_chembl_ids[0] # convert to string
        try:
            if verbose:
                print ("Obtaining data about CHEMBL documents:", chunk_document_chembl_ids)
            chunk_found_documents = documents.get(chunk_document_chembl_ids)
            if verbose:
                print ("Found", len(chunk_found_documents), "chunk documents")
            if isinstance(chunk_found_documents, dict):
                chunk_found_documents = [chunk_found_documents]
            else:
                chunk_found_documents = list(chunk_found_documents)
            all_found_documents.extend(chunk_found_documents)
        except Exception as e:
            print ("CHEMBL document live-query exception", e, )

    if verbose:
        print ("In total, found", len(all_found_documents), "documents")

    return all_found_documents

def search_chembl_for_activity_evidence(
    # molecule_chembl_ids,
    # target_chembl_ids,
    target_molecule_pairs: list,
    filter_by_assay: bool = True,
    min_assay_confidence_score: int = 8,
    assay_types: list = ["B", "F"], 
    activity_type: str = "active",
    allow_measured_standard_types: bool = True,
    allowed_measured_standard_types: list = ["EC50", "IC50", "AC50", "GC50", "GI50", "LD50", "Ki", "Kd"],
    allow_non_measured_standard_types: bool = True,
    allowed_non_measured_standard_types: list = ["Activity", "Inhibition", "Potency"],
    verbose: bool = False,
    ):
    """Use `chembl_webresource_client` to live query CHEMBL for activity evidence for a provided set of molecules and targets.

    Parameters
    ----------
    molecule_chembl_ids : list
        List of molecule_chembl_ids to search for
    target_chembl_ids : list
        List of target_chembl_ids to search for
    verbose : bool
        Flag to print updates to the console

    Returns
    -------
    list
        List of activity records for given molecules and targets
    """

    all_activity_evidence = []


    try:
        from chembl_webresource_client.new_client import new_client
    except Exception as e:
        print ("chembl_webresource exception", e, "returning", all_activity_evidence)
        return all_activity_evidence


    float_fields = ["value", "pchembl_value", "standard_value"]

    activities = new_client.activity

    base_filter_criteria = {
        # "pchembl_value__isnull": False,
    } 
    assay_filter_criteria = {
        "assay_type__in": assay_types,
        "confidence_score__gt": min_assay_confidence_score,
    }

    for target_chembl_id, molecule_chembl_id in target_molecule_pairs:

        # update filter critera
        assay_filter_criteria["target_chembl_id"] = target_chembl_id
        base_filter_criteria["target_chembl_id"] = target_chembl_id
        base_filter_criteria["molecule_chembl_id"] = molecule_chembl_id

        if filter_by_assay:

            if verbose:
                print ("filtering by assay:", assay_filter_criteria)

            try:
                from chembl_webresource_client.new_client import new_client
                
                assays = new_client.assay.filter(
                    **assay_filter_criteria,
                ).only(["assay_chembl_id"])
                assays = list(assays) # force exception
            except Exception as e:
                print ("Filtering my assay exception", e)
                assays = []
            if len(assays) == 0:
                continue

            assays = [assay["assay_chembl_id"] for assay in assays]

            if verbose:
                print ("discovered", len(assays), "assay(s)")
            base_filter_criteria["assay_chembl_id__in"] = assays

        try:
            activity_evidence = activities.filter(**base_filter_criteria, ).only(DESIRED_ASSAY_FIELDS)
            activity_evidence = list(activity_evidence)
            # clean up float records
            for evidence_record in activity_evidence:
                for float_field in float_fields:
                    if float_field not in evidence_record:
                        continue
                    float_value = evidence_record[float_field]
                    if float_value is None:
                        continue
                    try:
                        float_value = float(float_value)
                        evidence_record[float_field] = float_value
                    except Exception as e:
                        continue
                all_activity_evidence.append(evidence_record)

            # all_activity_evidence.extend( activity_evidence )
        except Exception as e:
            print ("CHEMBL live-query exception", e,)


    # if molecule_chembl_ids is not None:
    #     if isinstance(molecule_chembl_ids, set):
    #         molecule_chembl_ids = list(molecule_chembl_ids)
    #     if isinstance(molecule_chembl_ids, list) and len(molecule_chembl_ids) == 1:
    #         molecule_chembl_ids = molecule_chembl_ids[0]
    #     if verbose:
    #         print ("Searching for molecule(s)", molecule_chembl_ids)
    #     if isinstance(molecule_chembl_ids, str):
    #         base_filter_criteria["molecule_chembl_id"] = molecule_chembl_ids
    #     else:
    #         base_filter_criteria["molecule_chembl_id__in"] = molecule_chembl_ids

    # if target_chembl_ids is not None:
    #     if isinstance(target_chembl_ids, set):
    #         target_chembl_ids = list(target_chembl_ids)
    #     if isinstance(target_chembl_ids, list) and len(target_chembl_ids) == 1:
    #         target_chembl_ids = target_chembl_ids[0]
    #     if verbose:
    #         print ("Searching for targets(s)", target_chembl_ids)
        
    #     if isinstance(target_chembl_ids, str):
    #         base_filter_criteria["target_chembl_id"] = target_chembl_ids
    #         assay_filter_criteria["target_chembl_id"] = target_chembl_ids

    #     else:
    #         base_filter_criteria["target_chembl_id__in"] = target_chembl_ids
    #         assay_filter_criteria["target_chembl_id__in"] = target_chembl_ids

    # if filter_by_assay: # use min_assay_confidence_score to filter assays
    #     if verbose:
    #         print ("filtering by assay:", assay_filter_criteria)

    #     try:
    #         from chembl_webresource_client.new_client import new_client
            
    #         assays = new_client.assay.filter(
    #             **assay_filter_criteria,
    #         ).only(["assay_chembl_id"])
    #         assays = list(assays) # force exception
    #     except Exception as e:
    #         print ("Filtering my assay exception", e)
    #         assays = []

    #     assays = [assay["assay_chembl_id"] for assay in assays]
    #     if verbose:
    #         print ("discovered", len(assays), "assay(s)")
    #     base_filter_criteria["assay_chembl_id__in"] = assays



    # try:
    #     activity_evidence = activities.filter(**base_filter_criteria, ).only(DESIRED_ASSAY_FIELDS)
    #     activity_evidence = list(activity_evidence)
    #     # clean up float records
    #     for evidence_record in activity_evidence:
    #         for float_field in float_fields:
    #             if float_field not in evidence_record:
    #                 continue
    #             float_value = evidence_record[float_field]
    #             if float_value is None:
    #                 continue
    #             try:
    #                 float_value = float(float_value)
    #                 evidence_record[float_field] = float_value
    #             except Exception as e:
    #                 continue
    #         all_activity_evidence.append(evidence_record)

    #     # all_activity_evidence.extend( activity_evidence )
    # except Exception as e:
    #     print ("CHEMBL live-query exception", e,)

    return all_activity_evidence

def get_assay_type_description(
    assay_type: str,
    ):
    """Map single character `assay_type` to the full CHEMBL description of the assay.

    Parameters
    ----------
    assay_type : str
        Single character assay_type code. Must be one of {"B", "F", "A", "T", "P", "U"}

    Returns
    -------
    dict
        Full description of the assay type in dict form

    Raises
    ------
    ValueError
        Raised if invalid value is given to `assay_type`
    """
    assay_type = assay_type.capitalize()
    
    assert assay_type in {"B", "F", "A", "T", "P", "U"}
    
    
    if assay_type == "B":
        assay_type_full = "Binding"
        assay_type_description = "Data measuring binding of compound to a molecular target, e.g. Ki, IC50, Kd."
    elif assay_type == "F":
        assay_type_full = "Functional"
        assay_type_description = "Data measuring the biological effect of a compound, e.g. %cell death in a cell line, rat weight."
    elif assay_type == "A":
        assay_type_full = "ADMET"
        assay_type_description = "ADME data e.g. t1/2, oral bioavailability."
    elif assay_type == "T":
        assay_type_full = "Toxicity"
        assay_type_description = "Data measuring toxicity of a compound, e.g., cytotoxicity."
    elif assay_type == "P":
        assay_type_full = "Physicochemical"
        assay_type_description = "Assays measuring physicochemical properties of the compounds in the absence of biological material e.g., chemical stability, solubility."
    elif assay_type == "U":
        assay_type_full = "Unclassified"
        assay_type_description = "A small proportion of assays cannot be classified into one of the above categories e.g., ratio of binding vs efficacy."
    else:
        raise ValueError
    return {
        "assay_type_full": assay_type_full,
        "assay_type_description": assay_type_description,
    }


def get_data_about_chembl_targets(
    output_dir: str,
    target_types = ALL_CHEMBL_TARGET_TYPES,
    verbose: bool = False,
    ):
    """Query CHEMBL for the set of all targets for specified target types.
    The targets for each target type will be saved in JSON format in `output_dir`.
    All targets will be saved in `output_dir`'/all_targets.json'. 

    Parameters
    ----------
    output_dir : str
        A directory to save the targets retrieved in JSON format
    target_types : set, optional
        A set of CHEMBL target types, by default ALL_CHEMBL_TARGET_TYPES

    Returns
    -------
    dict
        Dictionary containing the data for all targets belonging to the target types 
        specified in `target_types`.
    """
    os.makedirs(output_dir, exist_ok=True,)

    all_targets = {}

    if isinstance(target_types, str):
        target_types = [target_types]

    for target_type in map(lambda s: s.replace(" ", "_").upper(), target_types):

        target_type_output_filename = os.path.join(
            output_dir, 
            f"{target_type}.json")
        if os.path.exists(target_type_output_filename):
            targets_of_type = load_json(target_type_output_filename, verbose=verbose)
            
        else:
            targets_of_type = {}

            if verbose:
                print ("identfying targets from CHEMBL of type", target_type)

            try:
                from chembl_webresource_client.new_client import new_client
                targets = new_client.target\
                    .filter(
                        target_type=target_type.replace("_", " "),
                        target_chembl_id__isnull=False,
                        )
                targets = list(targets) # force exception if it is going to happen
            except Exception as e:
                print ("Get CHEMBL targets error", e)
                targets = []
            n_targets = len(targets)
            if verbose:
                print ("Discovered", n_targets, "targets of type", target_type)

            # iterate over cursor
            for i, target in enumerate(targets, start=1):
                target_chembl_id = target["target_chembl_id"]
                targets_of_type[target_chembl_id] = target
                if verbose:
                    print("Completed target", target_chembl_id, ":", 
                        i, "/", n_targets, ":",
                        "{:.02f}% complete".format(i*100/n_targets))

            write_json(targets_of_type, target_type_output_filename, verbose=verbose)
        
        all_targets.update(targets_of_type)
        if verbose:
            print ("completed target type", target_type)

    # write all targets
    all_targets = {
        target_id: {
            "pref_name": data["pref_name"],
            "target_type": data["target_type"],
            "organism": data["organism"],
            "accessions": [component["accession"] for component in data["target_components"]],  
        }
        for target_id, data in all_targets.items()
    }

    all_targets_filename = os.path.join(output_dir, "all_targets.json")
    write_json(all_targets, all_targets_filename, verbose=verbose )

    return all_targets

def group_chembl_target_activities_by_molecule(
    target_activities: list,
    molecule_grouped_activities: dict,
    min_mol_weight: float,
    max_mol_weight: float,
    min_num_heavy_atoms: int,
    max_num_heavy_atoms: int,
    activity_type: str,
    verbose: bool = True,
    ):
    """Iterate over a list of records obtained from CHEMBL, given in `target_activities`
    and update the dict `molecule_grouped_activities` mapping molecule_chembl_id to a list
    of assays for that molecule. 

    Parameters
    ----------
    target_activities : list
        List of dict records from CHEMBL
    molecule_grouped_activities : dict
        Dictionary mapping molecule_chembl_id to assays
    min_mol_weight : float
        Minimum weight of molecule to consider, can be None to skip this check
    max_mol_weight : float
        Maximum weight of molecule to consider, can be None to skip this check
    min_num_heavy_atoms : int
        Minimum number of heavy atoms in the molecule to consider, can be None to skip this check
    max_num_heavy_atoms : int
        Maximum number of heavy atoms in the molecule to consider, can be None to skip this check
    activity_type : str
        Must be one of {'active', 'not active'}
    verbose : bool, optional
        Flag to print updates to the console, by default True

    Returns
    -------
    dict
        Updated `molecule_grouped_activities` with relevant records from `target_activities` added
    """
    n_target_activities = len(target_activities)
    for i, target_activity in enumerate(target_activities, start=1):

        relation = target_activity["standard_relation"]
        # filter by relation (= acceptable for both active and non-active)
        if isinstance(relation, str):
            if activity_type == "active" and ">" in relation:
                continue
            elif activity_type == "not active" and "<" in relation:
                continue
        # filter by comment
        activity_comment = target_activity["activity_comment"]
        if isinstance(activity_comment, str):
            if activity_type == "active" and activity_comment.lower() in {"inactive", "not active"}:
                continue
            elif activity_type == "not active" and activity_comment.lower() in {"active"}:
                continue

        molecule_smiles = target_activity["canonical_smiles"]
        rdkit_mol = Chem.MolFromSmiles(molecule_smiles)
        # skip invalid molecule
        if rdkit_mol is None:
            continue
        # filter by molecule weight
        mol_weight = CalcExactMolWt(rdkit_mol)
        if min_mol_weight is not None and mol_weight < min_mol_weight:
            continue
        if max_mol_weight is not None and mol_weight > max_mol_weight:
            continue
        # filter by number of heavy atoms
        num_heavy_atoms = rdkit_mol.GetNumHeavyAtoms()
        if min_num_heavy_atoms is not None and num_heavy_atoms < min_num_heavy_atoms:
            continue
        if max_num_heavy_atoms is not None and num_heavy_atoms > max_num_heavy_atoms:
            continue

        # get molecule_chembl_id and add to molecule_grouped_activities
        molecule_id = target_activity["molecule_chembl_id"]
        if molecule_id not in molecule_grouped_activities:
            molecule_grouped_activities[molecule_id] = []

        # clean up floats
        for float_key in (
            "pchembl_value",
            "standard_value",
            "value",
        ):
            if float_key in target_activity and target_activity[float_key] is not None:
                try:
                    target_activity[float_key] = float(target_activity[float_key])
                except:
                    pass

        molecule_grouped_activities[molecule_id].append(target_activity)

        if verbose:
            percent_complete = i * 100 / n_target_activities
            print("Processed activity",
                i, "/", n_target_activities,     
                f"{percent_complete:.02f}% complete")

    return molecule_grouped_activities

# uses DB
def get_activities_for_chembl_targets(
    targets: list, 
    target_activities_output_dir: str,
    activity_threshold_in_uM: float = 10,
    min_mol_weight: float = 100,
    max_mol_weight: float = 1000,
    min_num_heavy_atoms: int = None,
    max_num_heavy_atoms: int = None,
    filter_by_assay: bool = True,
    min_assay_confidence_score: int = 5,
    assay_types: list = ["B", "F"], 
    activity_type: str = "active",
    allow_measured_standard_types: bool = True,
    allowed_measured_standard_types: list = ["EC50", "IC50", "AC50", "GC50", "GI50", "LD50", "Ki", "Kd"],
    allow_non_measured_standard_types: bool = True,
    allowed_non_measured_standard_types: list = ["Activity", "Inhibition", "Potency"],
    allowed_relations: list = None,
    require_pchembl: bool = False,
    use_local_chembl_db: bool = False, 
    local_database_name: str = "chembl_26",
    verbose: bool = True,
    ):
    """Query CHEMBL and build a set of dictionaries, one for each target supplied in `targets`,
    mapping unique molecule_chembl_ids to relevant assays.

    Parameters
    ----------
    targets : list
        The list of targets to query activities for
    target_activities_output_dir : str
        The directory to output all activities, in JSON format
    activity_threshold_in_uM : float, optional
        The threshold in micro-Molar to determine active/inactive, by default 10
    min_mol_weight : float, optional
        Minimum weight (in Daltons) for molecules, by default 100
    max_mol_weight : float, optional
        Maximum weight (in Daltons) for molecules, by default 1000
    min_num_heavy_atoms : int, optional
        Minimum number of heavy atoms allowed for actives/inactives, by default None
    max_num_heavy_atoms : int, optional
        Maximum number of heavy atoms allowed for actives/inactives, by default None
    filter_by_assay : bool, optional
        Flag to consider assay confidence score to filter assays, by default True
    min_assay_confidence_score : int, optional
        Minimum assay confidence score, by default 5
    assay_types : list, optional
        Allowed types of assays, by default ["B", "F"] (Binding and Functional)
    activity_type : str, optional
        The type of activity to retrieve, by default "active"
    allow_measured_standard_types : bool, optional
        Flag to allow all measurable standard types, by default True
    allowed_measured_standard_types : list, optional
        Measurable standard types to consider, by default ["EC50", "IC50", "AC50", "GC50", "GI50", "LD50", "Ki", "Kd"]
    allow_non_measured_standard_types : bool, optional
        Flag to allow non-measurable standard types, by default True
    allowed_non_measured_standard_types : list, optional
        Non-measurable standard types to consider, by default ["Activity", "Inhibition", "Potency"]
    use_local_chembl_db : bool, optional
        Flag to use a local instance of a CHEMBL DB, otherwise use `chembl_webresource_client` to query up-to-date records, by default False
    local_database_name : str, optional
        Name of the local CHEMBL DB instance, by default "chembl_26"
    verbose : bool, optional
        Flag to print updates to the console, by default True

    Returns
    -------
    str
        The output directory containing the activities for each target in JSON files
    """


    os.makedirs(target_activities_output_dir, exist_ok=True)
    if verbose:
        print ("Downloading", activity_type, "activities for targets into directory", target_activities_output_dir)
    assert activity_type in {"active", "not active"}

    targets = sorted(targets)
    num_targets = len(targets)

    for target_num, target_chembl_id in enumerate(targets, start=1):

        if verbose:
            print ("Processing target", target_chembl_id)

        molecule_grouped_activities_filename = os.path.join(
            target_activities_output_dir, 
            f"{target_chembl_id}.json")

        molecule_grouped_activities = dict()

        if os.path.exists(molecule_grouped_activities_filename):

            continue #  skip target if already processed 
            # molecule_grouped_activities = load_json(molecule_grouped_activities_filename, verbose=verbose)
            # if len(molecule_grouped_activities) > 0:
            #     continue

        if use_local_chembl_db:

            from utils.queries.chembl_target_queries import get_activities_for_target_from_local_chembl_db_query
            from dotenv import load_dotenv
            load_dotenv()

            if verbose:
                print ("Querying local ChEMBL DB", local_database_name)

            target_activities = get_activities_for_target_from_local_chembl_db_query(
                target_chembl_ids=target_chembl_id,
                max_uM=activity_threshold_in_uM,
                assay_types=assay_types,
                min_assay_confidence_score=min_assay_confidence_score,
                allowed_measurements=allowed_measured_standard_types,
                allowed_potencies=allowed_non_measured_standard_types,
                database=local_database_name,
                verbose=verbose,
            )

            # update molecule_grouped_activities
            molecule_grouped_activities = group_chembl_target_activities_by_molecule(
                target_activities=target_activities,
                molecule_grouped_activities=molecule_grouped_activities,
                min_mol_weight=min_mol_weight,
                max_mol_weight=max_mol_weight,
                min_num_heavy_atoms=min_num_heavy_atoms,
                max_num_heavy_atoms=max_num_heavy_atoms,
                activity_type=activity_type,
                verbose=verbose,
            )

        else:

            if verbose:
                print ("Using chembl_webresource_client to search for up-to-date records")

            base_filter_criteria = {
                "target_chembl_id": target_chembl_id,

                "canonical_smiles__isnull": False,
                "molecule_chembl_id__isnull": False,
            }

            # filter by relation
            if allowed_relations is not None:
                if len(allowed_relations) == 1:
                    base_filter_criteria["standard_relation"] = allowed_relations[0]
                else:
                    base_filter_criteria["standard_relation__in"] = allowed_relations
            
            # require pchembl value 
            if require_pchembl: 
                base_filter_criteria["pchembl_value__isnull"] = False

            if filter_by_assay: # use min_assay_confidence_score to filter assays
                if verbose:
                    print ("filtering by assay")

                try:
                    from chembl_webresource_client.new_client import new_client
                    assays = new_client.assay.filter(
                        target_chembl_id=target_chembl_id,
                        assay_type__in=assay_types, 
                        confidence_score__gte=min_assay_confidence_score,
                    ).only(["assay_chembl_id"])
                    assays = list(assays) # force exception
                except Exception as e:
                    print ("Filtering my assay exception", e)
                    assays = []
                
                if verbose:
                    print ("discovered", len(assays), "assay(s)")
                
                if len(assays) == 0:
                    print ("Skipping target", target_chembl_id)
                    molecule_grouped_activities = {}
                    write_json(molecule_grouped_activities, molecule_grouped_activities_filename, verbose=verbose)
                    continue

                assays = [assay["assay_chembl_id"] for assay in assays]
                base_filter_criteria["assay_chembl_id__in"] = assays

            # construct a list of different filter criteria
            all_filter_criteria = []

            if allow_measured_standard_types:
                # select relation_type based on activity_type
                if activity_type == "active":
                    relation_type = "lte"
                else:
                    relation_type = "gt"

                nM_criteria = {   # nM criteria
                    **base_filter_criteria,
                    "standard_units": "nM",
                }
                if activity_threshold_in_uM is not None:
                    nM_criteria[f"standard_value__{relation_type}"] = activity_threshold_in_uM * 1000, # scale from uM to nM
                
                uM_criteria = {   # µM criteria
                    **base_filter_criteria,
                    "standard_units__in": ["µM", "uM"],
                }
                
                if activity_threshold_in_uM is not None:
                    uM_criteria[f"standard_value__{relation_type}"] = activity_threshold_in_uM

                if len(allowed_measured_standard_types) == 1:
                    nM_criteria["standard_type"] = allowed_measured_standard_types[0]
                    uM_criteria["standard_type"] = allowed_measured_standard_types[0]
                else:
                    nM_criteria["standard_type__in"] = allowed_measured_standard_types
                    uM_criteria["standard_type__in"] = allowed_measured_standard_types
                
                all_filter_criteria.extend([nM_criteria, uM_criteria, ])

            # non-measured
            if allow_non_measured_standard_types: # active/inactive comments
                non_measured_criteria = {
                    **base_filter_criteria,
                    "standard_type__in": allowed_non_measured_standard_types,
                    "activity_comment__iexact": activity_type,
                }
                all_filter_criteria.append(non_measured_criteria)

            # iterate over all filter criteria
            for filter_criteria in all_filter_criteria:

                if verbose:
                    print ("Filtering using criteria:", filter_criteria)

                # use chembl_webresource_client filter to obtain records
                try:
                    from chembl_webresource_client.new_client import new_client
                    target_activities = new_client.activity.filter(
                        **filter_criteria,
                        ).only(DESIRED_ASSAY_FIELDS)
                    target_activities = list(target_activities)
                except Exception as e:
                    print ("Get CHEMBL activities exception", e)
                    target_activities = []
                # process results from this set of filter criteria and update molecule_grouped_activities
                molecule_grouped_activities = group_chembl_target_activities_by_molecule(
                    target_activities=target_activities,
                    molecule_grouped_activities=molecule_grouped_activities,
                    min_mol_weight=min_mol_weight,
                    max_mol_weight=max_mol_weight,
                    min_num_heavy_atoms=min_num_heavy_atoms,
                    max_num_heavy_atoms=max_num_heavy_atoms,
                    activity_type=activity_type,
                    verbose=verbose,
                )

        write_json(molecule_grouped_activities, molecule_grouped_activities_filename, verbose=verbose) 
            
        if verbose:
            percent_complete = target_num / num_targets * 100
            print ("completed_target", target_num, "/", num_targets, 
                f": {percent_complete:.02f}% complete")
            print ("Number of", activity_type, "molecules found:", len(molecule_grouped_activities))
            print ()

    return target_activities_output_dir

def filter_target_actives(
    target_actives: dict,
    activity_threshold_in_uM: float,
    activity_type: str,
    max_mol_weight: int = None,
    min_mol_weight: int = None,
    min_num_heavy_atoms: int = None,
    max_num_heavy_atoms: int = None,
    molecule_chembl_ids_to_skip: set = None, 
    verbose: bool = True,
    ):

    # clean up target_actives by filtering for max uM, inactive comments, in mols_to_skip, etc.
    target_actives_filtered = {}
    for molecule_chembl_id, molecule_activities_for_target in target_actives.items():

        # ensure molecule_chembl_id is uppercase
        molecule_chembl_id = molecule_chembl_id.upper()

        # skip mol if it is in mols_to_skip
        if molecule_chembl_ids_to_skip is not None and molecule_chembl_id in molecule_chembl_ids_to_skip:
            if verbose:
                print (molecule_chembl_id, "is in set of molecules to skip, skipping it")
            continue
        # build list of activities for current molecule that pass the filter criteria 
        mol_activities_filtered = []
        for molecule_activity_for_target in molecule_activities_for_target:

            # sanity filter based on experimental comment
            activity_comment = molecule_activity_for_target["activity_comment"]
            if isinstance(activity_comment, str):
                activity_comment = activity_comment.lower()
                if activity_type == "active" and activity_comment in {"inactive", "not active"}:
                    if verbose:
                        print("skipping molecule_id", molecule_chembl_id, "comment:", activity_comment)
                    continue
                elif activity_type == "not active" and activity_comment in {"active",}:
                    if verbose:
                        print("skipping molecule_id", molecule_chembl_id, "comment:", activity_comment) 
                    continue
            elif activity_threshold_in_uM is not None:
                # sanity check on units (if no explicit experimental comment for record)
                standard_unit = molecule_activity_for_target["standard_units"]
                standard_value = molecule_activity_for_target["standard_value"]
                
                if standard_value is None:
                    if verbose:
                            print ("Skipping mol_id", molecule_chembl_id, "nM", "standard value is null", standard_value, "comment:", activity_comment)
                    continue # required if activity_threshold_in_uM is provided
                
                if isinstance(standard_value, str):
                    standard_value = float(standard_value)
                
                assert isinstance(standard_value, float), type(standard_value)

                if activity_type == "active":
                    if standard_unit in {"nM"} and standard_value > activity_threshold_in_uM * 1000:
                        if verbose:
                            print ("Skipping mol_id", molecule_chembl_id, "nM", standard_value, "comment:", activity_comment)
                        continue
                    if standard_unit in {"uM", "µM"} and standard_value > activity_threshold_in_uM:
                        if verbose:
                            print ("skipping mol_id", molecule_chembl_id, "uM", standard_value, "comment:", activity_comment)
                        continue
                elif activity_type == "not active":
                    if standard_unit in {"nM"} and standard_value < activity_threshold_in_uM * 1000:
                        continue
                    if standard_unit in {"uM", "µM"} and standard_value < activity_threshold_in_uM:
                        continue
                else:
                    raise NotImplementedError(activity_type)

            # require Ki / IC50?
            # standard_type = molecule_activity_for_target["standard_type"]
            # if standard_type not in {"Ki", "IC50"}:
            #     continue

            # require pchembl
            # pchembl_value = molecule_activity_for_target["pchembl_value"]
            # if pchembl_value is None:
            #     continue

            # filter by molweight/num atoms etc.
            canonical_smiles = molecule_activity_for_target["canonical_smiles"]
            mol = Chem.MolFromSmiles(canonical_smiles)
            if mol is None:
                continue # skip invalid molecule

            mol_weight = CalcExactMolWt(mol)
            if min_mol_weight is not None and mol_weight < min_mol_weight:
                continue
            if max_mol_weight is not None and mol_weight > max_mol_weight:
                continue
             # filter by number of heavy atoms
            num_heavy_atoms = mol.GetNumHeavyAtoms()
            if min_num_heavy_atoms is not None and num_heavy_atoms < min_num_heavy_atoms:
                continue
            if max_num_heavy_atoms is not None and num_heavy_atoms > max_num_heavy_atoms:
                continue

            mol_activities_filtered.append(molecule_activity_for_target)
        
        # add molecule to current set of actives for target
        if len(mol_activities_filtered) > 0:
            target_actives_filtered[molecule_chembl_id] = mol_activities_filtered
    
    return target_actives_filtered

def build_training_data(
    targets: list, 
    training_data_output_dir: str,
    target_activities_dir: str,
    activity_type: str,
    activity_threshold_in_uM=10,
    min_actives=10,
    max_actives=None,
    max_mol_weight: int = None,
    min_mol_weight: int = None,
    min_num_heavy_atoms: int = None,
    max_num_heavy_atoms: int = None,
    molecule_chembl_ids_to_skip: set = None, 
    seed: int = 0,
    verbose: bool = False,
    ):
    """Use downloaded activities, located in target_activities_dir, to build training data.
    Training data consists of a SMILES file and a sparse labels matrix. 

    Parameters
    ----------
    targets : list
        List of targets to build the training data for
    training_data_output_dir : str
        The directory to write the training data to
    target_activities_dir : str
        The directory containings the activities in JSON format
    activity_type : str
        The type of activity to build the training data on. Must be "active"
    activity_threshold_in_uM : int, optional
        The threshold in micro-Molar to determine active/inactive, by default 10
    min_actives : int, optional
        The minimum number of unique actives required to keep a target, by default 10
    max_actives : _type_, optional
        The maximum number of actives allowed for each target, by default None.
        If supplied, molecules will be selected based on chemical diversity.
    molecule_chembl_ids_to_skip : set, optional
        A set of molecule_chembl_ids to skip when constructing the training data, by default None
    verbose : bool, optional
        Flag to print updates to the console, by default False

    Returns
    -------
    str
        The directory containing all of the generated training data files

    Raises
    ------
    NotImplementedError
        Raised if activity_type not in {"active", "not active"}
    """
    assert activity_type == "active"

    training_data_output_dir = os.path.join(
        training_data_output_dir, 
        f"min_actives={min_actives}",
        f"max_actives={max_actives}",
        )
    os.makedirs(training_data_output_dir, exist_ok=True)

    mol_smiles_filename = os.path.join(
        training_data_output_dir, 
        "chembl_molecules.smi")
    
    # final file to write 
    uniprot_accession_to_target_chembl_id_filename = os.path.join(
        training_data_output_dir,
        "uniprot_accession_to_target_chembl_id.json")

    if verbose:
        print ("Using activities located in", target_activities_dir, "to build training data",
            "for", len(targets), "targets")
        print ("Writing training data to directory", training_data_output_dir)
        print ("Writing selected molecules to", mol_smiles_filename)
    
    if os.path.exists(uniprot_accession_to_target_chembl_id_filename):
        print (uniprot_accession_to_target_chembl_id_filename, "already exists, terminating")
        return training_data_output_dir

    training_molecule_smiles = {}
    target_to_molecule = {}

    for target_chembl_id in targets:
        target_actives_filename = os.path.join(
            target_activities_dir,
            f"{target_chembl_id}.json")
        assert os.path.exists(target_actives_filename), target_actives_filename
        target_actives = load_json(target_actives_filename, verbose=verbose)
        
        # clean up target_actives by filtering for max uM, inactive comments, in mols_to_skip, etc.
        target_actives = filter_target_actives(
            target_actives,
            activity_threshold_in_uM=activity_threshold_in_uM,
            activity_type=activity_type,
            max_mol_weight=max_mol_weight,
            min_mol_weight=min_mol_weight,
            min_num_heavy_atoms=min_num_heavy_atoms,
            max_num_heavy_atoms=max_num_heavy_atoms,
            molecule_chembl_ids_to_skip=molecule_chembl_ids_to_skip,
            verbose=verbose,
        )
        
        # determine number of actives remaining for current target
        number_of_actives = len(target_actives)

        if verbose:
            print ("Target:", target_chembl_id, "number of unique molecules retrieved from CHEMBL (before diversity selection):", number_of_actives)

        if number_of_actives < min_actives:
            if verbose:
                print ("Too few active molecules for target ID", target_chembl_id, )
            continue

        if target_chembl_id not in target_to_molecule:
            target_to_molecule[target_chembl_id] = list()

        # sort active molecule_chembl_ids lexicographically
        selected_molecules = sorted(target_actives) 

        # select `max_actives` actives
        if max_actives is not None and len(selected_molecules) > max_actives:

            # select by diversity 
            # extract smiles into list (preserving order of selected_mols)
            active_mol_smiles = [ 
                target_actives[mol_id][0]["canonical_smiles"]
                for mol_id in selected_molecules
            ]
            selected_indexes = diversity_molecule_selection(
                smiles=active_mol_smiles,
                sample_size=max_actives,
                seed=seed,
                verbose=verbose,
            )
            # use indexes to get mol IDs from selected_mols list
            selected_molecules = [ 
                selected_molecules[i]
                    for i in selected_indexes
            ]

            # select by pchembl value?
            # target_actives_pchembl_values = {}

            # key = "pchembl_value"

            # for molecule_chembl_id, molecule_target_activities in target_actives.items(): 
            #     max_val = None 
            #     for molecule_target_activity in molecule_target_activities:
            #         if key not in molecule_target_activity:
            #             continue
            #         val = molecule_target_activity[key]
            #         if max_val is None or val > max_val:
            #             max_val = val
            #     if max_val is None:
            #         continue
            #     target_actives_pchembl_values[molecule_chembl_id] = max_val

            # # sort molecules by max pchembl_values
            # molecules_sorted_by_value = sorted(
            #     target_actives_pchembl_values, 
            #     key=target_actives_pchembl_values.get, 
            #     reverse=True) # biggest to smallest

            # # take top max_actives molecules
            # selected_molecules = molecules_sorted_by_value[:max_actives]

            # if len(selected_molecules) < min_actives:

            #     selected_molecules = sorted(target_actives)

            #     # select by diversity 
            #     # extract smiles into list (preserving order of selected_mols)
            #     active_mol_smiles = [ 
            #         target_actives[mol_id][0]["canonical_smiles"]
            #         for mol_id in selected_molecules
            #     ]
            #     selected_indexes = diversity_molecule_selection(
            #         smiles=active_mol_smiles,
            #         sample_size=max_actives,
            #     verbose=verbose
            #     )
            #     # use indexes to get mol IDs from selected_mols list
            #     selected_molecules = [ 
            #         selected_molecules[i]
            #             for i in selected_indexes
            #     ]

        # iterate over selected_molecules
        for molecule_chembl_id in selected_molecules:

            if molecule_chembl_id not in training_molecule_smiles: # add canonical_smiles to set of all training_molecule_smiles
                training_molecule_smiles[molecule_chembl_id] = target_actives[molecule_chembl_id][0]["canonical_smiles"]
            
            # add molecule_id to target_to_molecule for current target
            target_to_molecule[target_chembl_id].append(molecule_chembl_id)

    # processing of targets has completed, begin training data construction
    n_training_molecules = len(training_molecule_smiles)
    n_targets = len(target_to_molecule)

    target_to_molecule_json_filename = os.path.join(
        training_data_output_dir,
        "target_to_molecule.json")
    write_json(target_to_molecule, target_to_molecule_json_filename, verbose=verbose)

    # map molecule_chembl_id to integer ID using string sorting 
    selected_molecules = sorted(training_molecule_smiles)
    molecule_to_id = {m: i 
        for i, m in enumerate(selected_molecules)}
    molecule_to_id_filename = os.path.join(
        training_data_output_dir, 
        "molecule_to_id.json")
    write_json(molecule_to_id, molecule_to_id_filename, verbose=verbose)

    # map target_chembl_id to integer ID using string sorting 
    sorted_targets = sorted(target_to_molecule)
    target_to_id = {t: i 
        for i, t in enumerate(sorted_targets)}
    target_to_id_filename = os.path.join(
        training_data_output_dir,
        "target_to_id.json")
    write_json(target_to_id, target_to_id_filename, verbose=verbose)

    training_molecule_smiles = [
        (mol_id, training_molecule_smiles[mol_id])
        for mol_id in selected_molecules]

    write_smiles(training_molecule_smiles, mol_smiles_filename, verbose=verbose)

    # build sparse labels matrix and save in .npz format
    # rows are molecules
    # columns are targets
    row_indices = []
    col_indices = []
    for target in target_to_molecule:
        target_id = target_to_id[target]
        for mol in target_to_molecule[target]:
            molecule_chembl_id = molecule_to_id[mol]
            row_indices.append(molecule_chembl_id)
            col_indices.append(target_id)
    data = [True] * len(row_indices)
    shape = (n_training_molecules, n_targets)

    labels = csr_matrix(
        (data, (row_indices, col_indices)), 
        shape=shape, 
        dtype=bool)
    
    for target in target_to_molecule:
        target_chembl_id = target_to_id[target]
        for mol in target_to_molecule[target]:
            molecule_chembl_id = molecule_to_id[mol]
            assert labels[molecule_chembl_id, target_chembl_id]

    # ensure min_actives and max_actives is respected 
    num_actives = labels.sum(0).A[0]
    assert (num_actives >= min_actives).all()
    if max_actives is not None:
        assert (num_actives <= max_actives).all()
    num_active_ranks = rankdata(-num_actives, method="dense") # rank targets by the number of actives in the training set

    labels_filename = os.path.join(training_data_output_dir, "labels.npz")
    save_npz(labels_filename, labels)

    # write data about targets in training set
    all_targets_in_training_set_filename = os.path.join(
        training_data_output_dir,
        "all_targets_in_training_set.json")
    all_targets_in_training_set = {
        target_chembl_id: {
            "target_chembl_id": target_chembl_id,
            **target_data,
            "num_actives": int(num_actives[target_to_id[target_chembl_id]]),
            "num_actives_rank": int(num_active_ranks[target_to_id[target_chembl_id]]),
        } 
        for target_chembl_id, target_data in targets.items() 
        if target_chembl_id in target_to_id
    }
    write_json(all_targets_in_training_set, all_targets_in_training_set_filename, verbose=verbose)

    # id to target (including data)
    id_to_target = {
        i: all_targets_in_training_set[target_chembl_id]
        for target_chembl_id, i in target_to_id.items()
    }
    id_to_target_filename = os.path.join(
        training_data_output_dir, 
        "id_to_target.json")
    write_json(id_to_target, id_to_target_filename, verbose=verbose)

    # print target types in training set
    uniprot_accession_to_target_chembl_id = {}
    target_types = {}

    for target_chembl_id, data in all_targets_in_training_set.items():
        target_type = data["target_type"]
        if target_type not in target_types:
            target_types[target_type] = []
        target_types[target_type].append(target_chembl_id)
        components = data["accessions"]
        for acc in components:
            if acc not in uniprot_accession_to_target_chembl_id:
                uniprot_accession_to_target_chembl_id[acc] = []
            uniprot_accession_to_target_chembl_id[acc].append(target_chembl_id)

    # write unique accs to file

    write_json(uniprot_accession_to_target_chembl_id, uniprot_accession_to_target_chembl_id_filename, verbose=verbose)

    if verbose:
        print ("Completed generation of training data")
        print ("Summary:")
        print ("Number of unique molecules:", len(molecule_to_id))
        print ("Number of unique CHEMBL targets:", len(target_to_id))
        print ("Number of unique UniProt accessions:", len(uniprot_accession_to_target_chembl_id))
        print ("Total number of activities:", labels.sum())
        print ("Number of polypharm ligands:", (labels.sum(axis=-1) > 1).sum())

    return training_data_output_dir


if __name__ == "__main__":

    # get molecule chembl IDs for DB

    from dotenv import load_dotenv
    load_dotenv()

    from utils.mysql_utils import mysql_query, chunked_load_data_from_local_file

    # query = '''
    # SELECT 
    #     inchikey
    # FROM small_molecule
    # WHERE molecule_chembl_id IS NULL
    # '''

    # records = mysql_query(query, verbose=True,)

    # # records = [{
    # #     "inchikey": "ABJKWBDEJIDSJZ-UHFFFAOYSA-N"
    # # }]


    # chunksize = 1_000

    # n_records = len(records)
    # n_chunks = n_records // chunksize + 1

    # for chunk_num in range(n_chunks):
    #     chunk_records = records[chunk_num * chunksize :  (chunk_num+1) * chunksize]
    #     if len(chunk_records) == 0:
    #         continue
        
    #     chunk_inchikeys = [
    #         r["inchikey"]
    #         for r in chunk_records
    #     ]

    #     chembl_molecules = match_chembl_molecules_by_inchikey(
    #         inchikeys=chunk_inchikeys,
    #         verbose=True,
    #     )

    #     if len(chembl_molecules) == 0:
    #         continue

    #     rows = []

    #     for chembl_molecule in chembl_molecules:
    #         molecule_chembl_id = chembl_molecule["molecule_chembl_id"]
    #         inchikey = chembl_molecule["molecule_structures"]["standard_inchi_key"]

    #         rows.append((molecule_chembl_id, inchikey))

    #     chunked_load_data_from_local_file(
    #         rows=rows,
    #         fields=["molecule_chembl_id", "inchikey"],
    #         table_name="small_molecule_temp",
    #         verbose=True,
    #     )
    # raise Exception


    # evidence = search_chembl_for_activity_evidence(
    #     # molecule_chembl_ids=["CHEMBL452383"],
    #     # target_chembl_ids=["CHEMBL3594"],
    #     target_molecule_pairs=[("CHEMBL3594", "CHEMBL452383"), ("CHEMBL3594", "CHEMBL25")],
    #     verbose=True,
    # )

    # write_json(evidence, "evidence.json")

    from concurrent.futures import ProcessPoolExecutor

    # build training data
    root_output_dir = "/media/david/Elements/data/chembl"

    target_types = ["SINGLE PROTEIN"]
    # target_types = ["PROTEIN-PROTEIN INTERACTION"]
    # target_types = ALL_CHEMBL_TARGET_TYPES

    organism_groups = ["Homo sapiens"]
    # organism_groups = ALL_CHEMBL_ORGANISM_GROUPS

    # use_local_chembl_db = False
    use_local_chembl_db = True

    # local_db_name = "chembl_26"
    local_db_name = "chembl_28"
    # local_db_name = "chembl_31"

    # for training data
    activity_threshold_in_uM = 10
    min_actives = 10
    min_mol_weight = 100
    max_mol_weight = 1000 # Excape
    min_num_heavy_atoms = 12 # Excape
    max_num_heavy_atoms = None

    with ProcessPoolExecutor(max_workers=22) as p:

        running_tasks = []

        for target_type in target_types:

            all_target_type_targets = get_data_about_chembl_targets(
                output_dir=root_output_dir,
                target_types=[target_type],
                verbose=True,
            )

            # assay_root_output_dir = os.path.join(root_output_dir, "assays")
            assay_root_output_dir = os.path.join(root_output_dir, "assays_6")

            # append DB name if using local
            if use_local_chembl_db:
                assay_root_output_dir += f"_{local_db_name}"

            os.makedirs(assay_root_output_dir, exist_ok=True)

            filter_by_assay = target_type == "SINGLE PROTEIN"
            if filter_by_assay:
                min_assay_confidence_score = 6 # PIDGIN
                # min_assay_confidence_score = 9 # exactly one direct assay target
            else:
                min_assay_confidence_score = 0

            for organism_group in organism_groups:

                organism_targets = {
                    target : target_data
                    for target, target_data in all_target_type_targets.items()
                    if get_chembl_organism_group(target_data["organism"]) == organism_group
                }

                # download molecule -> target activities from ChEMBL
                get_activities_for_chembl_targets(
                    targets=organism_targets,
                    target_activities_output_dir=assay_root_output_dir,
                    activity_threshold_in_uM=None,
                    min_mol_weight=None,
                    max_mol_weight=None, # keep general, filter when selecting training data
                    min_num_heavy_atoms=None,
                    max_num_heavy_atoms=None,
                    filter_by_assay=filter_by_assay,
                    min_assay_confidence_score=min_assay_confidence_score, # for single protein only (otherwise, do not filter by assay)
                    activity_type="active",
                    require_pchembl=False,
                    use_local_chembl_db=use_local_chembl_db,
                    local_database_name=local_db_name,
                    verbose=True,
                )

                training_data_root_output_dir = os.path.join(
                    root_output_dir, 
                    # "training_data",
                    "training_data_6",
                )

                # append local db name if using it
                if use_local_chembl_db:
                    training_data_root_output_dir += f"_{local_db_name}"

                training_data_root_output_dir = os.path.join(
                    training_data_root_output_dir,
                    target_type.replace(" ", "_"),
                    organism_group.replace(" ", "_"),
                    )
                os.makedirs(training_data_root_output_dir, exist_ok=True,)

                for max_actives in (
                    None,
                    1000,
                ):

                    # multiple seeds for diversity selection
                    num_seeds = 1 if max_actives == None else 30
                    # num_seeds = 1

                    for seed in range(num_seeds):

                        seed_training_data_root_output_dir = os.path.join(
                            training_data_root_output_dir,
                            f"{seed:02d}",
                        )
                        os.makedirs(seed_training_data_root_output_dir, exist_ok=True,)

                        task = p.submit(
                            build_training_data,
                            targets=organism_targets,
                            training_data_output_dir=seed_training_data_root_output_dir, 
                            target_activities_dir=assay_root_output_dir,
                            activity_type="active",
                            activity_threshold_in_uM=activity_threshold_in_uM,
                            min_actives=min_actives,
                            max_actives=max_actives, 
                            min_mol_weight=min_mol_weight,
                            max_mol_weight=max_mol_weight,
                            min_num_heavy_atoms=min_num_heavy_atoms,
                            max_num_heavy_atoms=max_num_heavy_atoms,
                            molecule_chembl_ids_to_skip=None,
                            seed=seed,
                            verbose=False,
                        )

                        running_tasks.append(task)

        # await all task completion
        for running_task in running_tasks:
            running_task.result()