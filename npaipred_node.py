import logging
import knime.extension as knext
import knime.types.chemistry as ktchem
from bioactivity_predictor import predict
import os
import pandas as pd
import rdkit
from rdkit import Chem
import requests
import json 
import random
import pandas as pd
LOGGER = logging.getLogger(__name__)

class NewError(Exception):
        def __init__(self, message):
            self.message = message

my_category = knext.category(
    path="/community",
    level_id="AIA_Insigths",
    name="AIA Insigths",
    description="Extensions developed by AIA Insigths LTD",
    icon="icon.png",
)

@knext.node(
    name="Bioactivity Predictor", 
    node_type=knext.NodeType.PREDICTOR, 
    icon_path="icon.png", 
    category=my_category
    )

@knext.input_table(
    name="Input Data", 
    description="The input should be a table containing a SMILES column and an ID column. This can be achieved by passing .smi files or directly"+ 
    "inputting the molecule data into the AIA smiles reader extension plugin. Custom table inputs are supported, but when writng to ")

@knext.output_table(
    name="Output Data", 
    description="Whatever the node has produced")
class ModelNode:
    """Natural product bioactivity predictor
    This plugin allows for prediction of the bioactivity of input molecules with respect to a number of targets. To access the API key, required to
    connect to AIA servers, please email cto@aiainsights.com .
    """

    api_key_param = knext.StringParameter("The API key for model access", "Please email cto@aiainsights.com for a trial AIA Insights account and an API token", ""  )
    #consider_chirality_param = knext.BoolParameter("Consider chirality", "Should the model consider chirality of the compounds when making predictions?")
    #target_type_param = knext.StringParameter("Target type", "The target category", enum=["SINGLE PROTEIN", "PROTEIN-PROTEIN INTERACTION", "PROTEIN COMPLEX", "CELL LINE", "CHIMERIC PROTEIN", "PROTEIN COMPLEX GROUP", "PROTEIN FAMILY", "SELECTIVITY GROUP"], default_value = "SINGLE PROTEIN")
    #organism_group_param = knext.StringParameter("Organism group", "Targets Organism of origin", enum=["Homo Sapiens", "Other organisms", "Selected animals"], default_value = "Homo Sapiens")
    output_path_param = knext.StringParameter("CSV output file folder", "Provide a folder path if you'd like to save the predictions in a .csv format", "")


    def process_input_mols(self, input_table):
        inputs = []
        input_1_pandas = input_table.to_pandas()
        if "SMILES" in input_1_pandas:
            smiles_list = input_1_pandas["SMILES"]
        else:
            pass
        if "ID" in input_1_pandas:
            id_list = input_1_pandas["ID"]
        else:
            pass

        for index,molecule in enumerate(smiles_list):
            input_list = []
            try:
                input_list.append(molecule)
                input_list.append(id_list[index])
                inputs.append(input_list)
            except IndexError:
                pass
        return inputs

    def configure(self, configure_context, input_schema_1):
        input_schema_1 = input_schema_1.append(knext.Column(knext.string(), "Rank 1 predictions"))
        input_schema_1 = input_schema_1.append(knext.Column(knext.string(), "Rank 2 predictions"))
        input_schema_1 = input_schema_1.append(knext.Column(knext.string(), "Rank 3 predictions"))
        return input_schema_1

        json_object = json.dumps(relevant_data, indent=4)

        with open("a:/test/sample.json", "w") as outfile:
            outfile.write(json_object)

        return None

    def save_to_csv(self, molecule_output_filename, mol_entry):
        column_headers = ['target_CHEMBL',
                        'rank',
                        "Num_of_most_similar_mols_as_evidence",
                        "gene",
                        "protein",
                        "accession",
                        "entrez_gene_id",
                        "opentargets_id",
                        "protein_family",
                        "uniprot_identifier",
                        "organism_scientific"]
        df = pd.DataFrame(self.prep_csv_entry(mol_entry), columns=column_headers) 
        df.to_csv(molecule_output_filename, index=False)

    def prep_csv_entry(self, mol_entry):
        csv_entry = []
        for index, rank_data in enumerate(mol_entry):
            for item in rank_data:
                csv_entry.append([item["target_chembl_id"],
                                    index+1,
                                    item["num_target_activity_evidence_for_most_similar_mols"],
                                    item["gene"],
                                    item["protein"],
                                    item["accession"],
                                    item["entrez_gene_id"],
                                    item["opentargets_id"],
                                    item["protein_family"],
                                    item["uniprot_identifier"],
                                    item["organism_scientific"]])
        return csv_entry 

    def online_prediction(self, input_mols):
        url = "https://app.npaiengine.com/activity_prediction/submit/"
        input_string = ""
        for index,input in enumerate(input_mols):
            if len(input) != 2:
                generated_id = f"{input[0]}_{random.randint(0, 1000)}"
                input_mols[index] = [input[0] , generated_id]
            input_string += f"{input_mols[index][0]} {input_mols[index][1]} "
        response = requests.post(url, data={"user": self.api_key_param,
            "ligand-upload-option": "paste",
            "smiles": input_string,
            "model_root_dir": "chembl_31",
            "target_types": "SINGLE PROTEIN",
            "organisms": "Homo sapiens",
            "maximum-target-rank": 10})
        """Add in a check for response, if it is the wrong type, throw up a custom exception"""
        """"response = open('a:/test/sample.json')
        relevant_data= json.load(response)"""

        js = json.loads(response.text)
        relevant_data = js["targets"]["SINGLE PROTEIN"]["Homo sapiens"]
        results = []
        for molecule in input_mols:
            rank_data = {1: [], 2: [], 3: []}

            for target in relevant_data.get(molecule[1], []):
                rank = target["target_rank"]
                if rank <= 3:
                    rank_entry = {"target_chembl_id": target["target_chembl_id"],
                    "num_target_activity_evidence_for_most_similar_mols":target["num_target_activity_evidence_for_most_similar_mols"],
                    "gene":target["gene"],
                    "protein":target["protein"],
                    "accession":target["accession"],
                    "entrez_gene_id":target["entrez_gene_id"].replace(';', ''),
                    "opentargets_id":target["opentargets_id"].replace(';', ''),
                    "protein_family":target["protein_family"],
                    "uniprot_identifier":target["uniprot_identifier"],
                    "organism_scientific":target["organism_scientific"]}

                    rank_data[rank].append(rank_entry)

            mol_entry = [rank_data[1], rank_data[2], rank_data[3]]
            mol_dict = {f"rank_{i}": [x["accession"] for x in rank_data[i]] for i in range(1, 4)}
            results.append(mol_dict)
            if self.output_path_param is not "":
                output_dir = self.output_path_param
                os.makedirs(output_dir, exist_ok=True)
                molecule_output_filename = os.path.join(output_dir, f"{molecule[1]}.csv")
                self.save_to_csv(molecule_output_filename, mol_entry)
        return results
    
    def execute(self, exec_context, input_1):
        inputs = [["CC(=O)OC1=CC=CC=C1C(=O)O", "aspirin"], ["CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  "Ibuprofen"], ["O=C1c3c(O)c(O)c(O)cc3OC(=C1)c2ccccc2", "baicalein"]]
        molecules = self.process_input_mols(input_1)
        results = self.online_prediction(molecules)

        input_1_pandas = input_1.to_pandas()
        rank_1_pred, rank_2_pred, rank_3_pred = [], [], []


        for molecule in results:
            rank_1_pred.append(",".join(molecule["rank_1"]))
            rank_2_pred.append(",".join(molecule["rank_2"]))
            rank_3_pred.append(",".join(molecule["rank_3"]))
        input_1_pandas["Rank 1 predictions"] = rank_1_pred
        input_1_pandas["Rank 2 predictions"] = rank_2_pred
        input_1_pandas["Rank 3 predictions"] = rank_3_pred
        return knext.Table.from_pandas(input_1_pandas)


@knext.node(
    name="Bioactivity Predictor SMILE reader/formatter", 
    node_type=knext.NodeType.SOURCE, 
    icon_path="icon.png", 
    category=my_category
    )


@knext.output_table(
    name="SMILES table", 
    description="A table of molecule SMILES strings, with IDs in a format readable for bioactivity predictor")

class SmilesReaderNode():
        """SMILES reader node, used to generate tables in appropriate format for the NPAIPredictor node. The node accepts .smi files as input, folders
        containing single .smi files or manual input of SMILES + names."""
    smiles_input_param = knext.StringParameter("Input molecules or molecule source", "The input molecules may be provided as a string (format: SMILE_1, ID_1; SMILE2_2, ID_2...SMILE_N, ID_N) or path to a .smi file or folder containing .smi files. The smiles files may contain single smiles (if providing a directory only) or multiple SMILE-ID pairs, separated by a separator chosen in the 'Separator for smile file(s) option'", "CC(=O)OC1=CC=CC=C1C(=O)O, aspirin; CC(C)CC1=CC=C(C=C1)C(C)C(=O)O, ibuprofen")
    delimiter_param = knext.StringParameter("Separator for smile file(s)", "If providing a file path to a .smi file, select the separator", enum=["space",",",";"], default_value="space")
    def is_smiles(col):
        """
        Check if the provided knext.Column contains smiles values. Due to
        technical reasons, the type of the column can be either SmilesValue or 
        SmilesAdapterValue, so here we check for both.
        
        The function knext.logical turns the value types into KNIME column types,
        and is called "logical" as opposed to "primitive" like plain numbers.
        """
        return col.ktype == knext.logical(ktchem.SmilesValue) or col.ktype == knext.logical(ktchem.SmilesAdapterValue)

    def read_smiles(self, smile_file):
        input_dict = {}
        smiles_list = []
        id_list = []
        with open(smile_file) as f:
            for line in f:
                line = line.strip("\n")
                if self.delimiter_param == "space":
                    input = line.split(" ")
                else:
                    input = line.split(self.delimiter_param)
                smiles_list.append(input[0].strip(" "))
                id_list.append(input[1].strip(" "))
        swap_lists = False
        for index,smile in enumerate(smiles_list):
            mol = None
            try:
                mol =Chem.MolFromSmiles(smile)
            except Chem.rdchem.KekulizeException:
                pass
            if mol is None:
                try:
                    mol = Chem.MolFromSmiles(id_list[index])
                    swap_lists = True
                except Chem.rdchem.KekulizeException:
                    pass
            if mol is None:
                    raise NotImplementedError(f"SMILE: {smile}, {id_list[index]} failed to convert to an RDkit mol object")
        if swap_lists:
            temp_list = smiles_list
            smiles_list = id_list
            id_list = temp_list
        input_dict["molecule_id"] = id_list
        input_dict["smiles"] = smiles_list
        return input_dict

    def read_smiles_folder(self, smile_folder):
        input_dict = {"molecule_id":[], "smiles":[]}
        file_list = os.listdir(smile_folder)
        smile_list = []
        id_list = []
        for file_name in file_list:
            file_path = os.path.join(smile_folder, file_name)
            if file_name.endswith(".smi"):
                try:
                    with open(file_path, 'r') as file:
                        mol = None
                        smile = file.read().strip(" ").strip("\n")
                        try:
                            mol = Chem.MolFromSmiles(smile)
                        except Chem.rdchem.KekulizeException:
                            pass
                        if mol is not None:
                            base_name, extension = os.path.splitext(file_name)
                            id = base_name
                            id_list.append(id)
                            smile_list.append(smile)
                        else:
                            smiles_dict = self.read_smiles(file_path)
                            input_dict["molecule_id"] = input_dict["molecule_id"] + smiles_dict["molecule_id"]
                            input_dict["smiles"] = input_dict["smiles"] + smiles_dict["smiles"]
                except Exception as e:
                    print(f"An error occurred while processing {file_name}: {str(e)}")
        input_dict["molecule_id"] = input_dict["molecule_id"] + id_list
        input_dict["smiles"] = input_dict["smiles"] + smile_list
        return input_dict
    
    def process_input_mols(self):
        input_dict = {}
        id_list = []
        smiles_list = []
        if os.path.isfile(self.smiles_input_param):
            input_dict = self.read_smiles(self.smiles_input_param)
        elif os.path.isdir(self.smiles_input_param):
            input_dict = self.read_smiles_folder(self.smiles_input_param)
        else:
            mols = self.smiles_input_param.split(";")
            for molecule in mols:
                mol = molecule.split(",")
                try:
                    id_list.append(mol[1].strip())
                    smiles_list.append(mol[0].strip())
                except IndexError:
                    pass
            input_dict["molecule_id"] = id_list
            input_dict["smiles"] = smiles_list
        return input_dict
    
    def configure(self, config_context):  
        pass
        

    def execute(self, exec_context):
        input_dict = self.process_input_mols()
        data = pd.DataFrame({"ID": input_dict["molecule_id"], "SMILES": input_dict["smiles"]})
        return knext.Table.from_pandas(data)
