

if __name__ == "__main__":
    import sys
    import os.path

    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import os 
import re

import numpy as np
import pandas as pd
from scipy import sparse as sp


from joblib import parallel_backend

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, ComplementNB, CategoricalNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifierCV
from sklearn.ensemble import (
    BaggingClassifier, ExtraTreesClassifier, 
    AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier

from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import NotFittedError
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, chi2, f_classif

# from skmultilearn.model_selection import IterativeStratification

# from xgboost import XGBClassifier

from concurrent.futures import ProcessPoolExecutor

from functools import partial

from utils.io.io_utils import load_compressed_pickle, load_json, write_compressed_pickle
from utils.molecules.fingerprints import compute_molecular_fingerprints

dense_input = {"nn", "lda", }
support_multi_label = {"nn", "etc", }

# def build_model(args):
#     model = args.model
#     assert isinstance(model, list)
#     n_proc = args.n_proc

#     print ("model is", model[0])
#     if model[0] == "stack":
#         return StackedBioactivityPredictor(
#             models=model[1:],
#             n_proc=n_proc,
#         )
#     else:
#         return BioactivityPredictor(model=model[0], n_proc=n_proc)

def canonise_fp(
    fp: str,
    ):
    # returns None if fp is None 
    if isinstance(fp, str) and "+" in fp:
        fp = fp.split("+")
    if isinstance(fp, list) or isinstance(fp, tuple):
        fp = sorted(set(fp))
        fp = "+".join(fp)

    return fp

def get_model_filename(
    model_filename,
    verbose: bool = True,
    ):
    if verbose:
        print ("Getting model filename for", model_filename)
    if not model_filename.endswith(".pkl.gz"):
        model_filename += ".pkl.gz"
    model_filename = re.sub(r"-nn.+\.pkl", "-nn.pkl", model_filename)
    if verbose:
        print ("Model filename is", model_filename)
    return model_filename

def save_model(
    model, 
    model_filename: str,
    verbose: bool = True,
    ):
    if verbose:
        print ("Saving model to", model_filename)
    model_filename = get_model_filename(model_filename, verbose=verbose)
    assert model_filename.endswith(".pkl.gz")
    model_output_dir = os.path.dirname(model_filename)
    os.makedirs(model_output_dir, exist_ok=True)
    return write_compressed_pickle(model, model_filename, verbose=verbose)

def load_model(
    model_filename,
    verbose: bool = False,
    ):
    if verbose:
        print ("Loading model from", model_filename)
    assert model_filename.endswith(".pkl.gz")
    # remove classifier from name in nn+ models
    if not os.path.exists(model_filename):
        model_filename = get_model_filename(model_filename, verbose=verbose)
    if not os.path.exists(model_filename):
        raise IOError("Model does not exist at", model_filename)
    return load_compressed_pickle(model_filename, default=None, verbose=verbose)

def get_unique_column_indexes_sparse_matrix(data):
    unique_col_indices = []
    unique_rows = set()
    # iterate over columns
    for col_idx, col in enumerate(data.T):
        indices = tuple(col.indices.tolist())
        if indices not in unique_rows:
            unique_rows.add(indices)
            unique_col_indices.append(col_idx)
    return unique_col_indices

def make_local_classifier_prediction_single_query(
    query,
    query_neighbour_distances,
    X_neighbour,
    y_neighbour,
    k: int,
    classifier,
    remove_zero_variance: bool,
    use_unique_columns_only: bool,
    return_chembl_matches: bool = True,
    ):
    if len(query.shape) == 1:
        # make query 2D
        query = query[None,:]

    # X_neighbour = X[query_neighbour_idx]
    # y_neighbour = y[query_neighbour_idx]

    if remove_zero_variance:

        X_neighbour_feature_sums = X_neighbour.sum(0).A.flatten()
        non_zero_variance_idx = np.logical_and(X_neighbour_feature_sums > 0, X_neighbour_feature_sums < k)

        if non_zero_variance_idx.any(): # if all features have zero variance then do nothing 
            X_neighbour = X_neighbour[:,non_zero_variance_idx]
            query = query[:,non_zero_variance_idx]

        # del X_neighbour_feature_sums, non_zero_variance_idx

    if use_unique_columns_only:
        unique_column_idx = get_unique_column_indexes_sparse_matrix(X_neighbour)
        if unique_column_idx: # if unique_column_idx contains any elements
            X_neighbour = X_neighbour[:,unique_column_idx]
            query = query[:,unique_column_idx]
        
        # del unique_column_idx

    n_targets = y_neighbour.shape[1]

    # initialise predictions matrix for query molecule
    query_target_probabilities = sp.lil_matrix((1, n_targets), )

    target_counts = y_neighbour.sum(axis=0).A[0]

    # predict 0 if all neighbours do not target
    predict_zeros_idx = target_counts == 0
    # predict 1 if all (k) neighbours do target
    predict_ones_idx = target_counts == k

    # predict ones for distance=0
    if return_chembl_matches:
        zero_distance_mask = (query_neighbour_distances == 0)[0]
        if zero_distance_mask.any():
            zero_distance_labels = y_neighbour[zero_distance_mask].A.any(axis=0)
            predict_ones_idx = np.logical_or(predict_ones_idx, zero_distance_labels)

    # set prediction for classes where only positive class is seen
    query_target_probabilities[0, predict_ones_idx] = 1.0

    # only fit classifier for classes with examples of both positive and negative samples
    predict_with_classifier_idx = ~np.logical_or(predict_zeros_idx, predict_ones_idx)

    # count number of targets to predict with a classifier
    num_targets_to_predict = predict_with_classifier_idx.sum()

    if num_targets_to_predict > 0:
        # select labels of classifier targets
        y_neighbour_predictable_targets = y_neighbour[:,predict_with_classifier_idx]
        if num_targets_to_predict == 1:
            y_neighbour_predictable_targets = y_neighbour_predictable_targets.A # make dense for some reason
        # fit classifier 
        classifier.fit(X_neighbour, y_neighbour_predictable_targets, )
        # use classifier to predict probabilities for all predictable targets
        classifier_query_target_probabilities = classifier.predict_proba(query) 

        if num_targets_to_predict == 1:
            # index in using classes_ attribute to select positive class
            classifier_query_target_probabilities = classifier_query_target_probabilities[:,classifier.classes_==True]
    
        # update return array
        query_target_probabilities[0, predict_with_classifier_idx] = classifier_query_target_probabilities

    return query_target_probabilities

def make_all_query_predictions(
    queries,
    neighbour_distances,
    neighbour_indices,
    X,
    y,
    k: int,
    classifier,
    remove_zero_variance: bool,
    use_unique_columns_only: bool,
    return_chembl_matches: bool = True,
    chunk_size: int = 1_000,
    n_proc: int = 1,
    verbose: bool = True,
    ):

    n_queries = queries.shape[0]

    n_chunks = n_queries // chunk_size + 1

    if verbose:
        print ("Building custom classifier for", n_queries, "queries using", n_proc, "process(es)" )
        print ("Chunking to", n_chunks, "chunk(s) of size", chunk_size)

    with ProcessPoolExecutor(max_workers=n_proc) as p:

        all_query_target_probabilities = []

        for chunk_num in range(n_chunks):

            chunk_queries = queries[chunk_num * chunk_size : (chunk_num+1) * chunk_size]
            chunk_neighbour_distances = neighbour_distances[chunk_num * chunk_size : (chunk_num+1) * chunk_size]
            chunk_neighbour_indices = neighbour_indices[chunk_num * chunk_size : (chunk_num+1) * chunk_size]
            chunk_neighbour_features =  (X[ni] for ni in chunk_neighbour_indices)
            chunk_neighbour_labels =  (y[ni] for ni in chunk_neighbour_indices)
            
            if len(chunk_queries) == 0:
                continue
            
            chunk_query_predictions = p.map(
                partial(
                    make_local_classifier_prediction_single_query,
                    k=k,
                    classifier=classifier,           
                    remove_zero_variance=remove_zero_variance,  
                    use_unique_columns_only=use_unique_columns_only,
                    return_chembl_matches=return_chembl_matches,
                ),
                # iterate over multiple lists of args
                chunk_queries, 
                chunk_neighbour_distances, 
                chunk_neighbour_features, # generator
                chunk_neighbour_labels, # generator
            )
            if not isinstance(chunk_query_predictions, list):
                chunk_query_predictions = list(chunk_query_predictions)

            all_query_target_probabilities.extend(chunk_query_predictions)

            if verbose:
                print ("Completed predictions for query chunk", chunk_num+1, "/", n_chunks)

    if isinstance(all_query_target_probabilities, list):
        all_query_target_probabilities = sp.vstack(all_query_target_probabilities).tocsr()

    return all_query_target_probabilities

class BioactivityPredictor(BaseEstimator, ClassifierMixin):  
    """BioactivityPredictor model"""
    
    def __init__(
        self, 
        fp: str,
        model_name: str,
        id_to_target_filename: str,
        use_features: bool = True,
        consider_chirality: bool = False,
        n_bits: int = 1024,
        k: int = 2000,
        cv: float = float("inf"), # minimum targets for calibration
        alpha: float = 1, # smoothing co-efficient for NB
        num_similar_mols_to_show: int = 5, # number of most similar mols to show
        n_proc: int = 1,
        ):

        self.fp =fp
        self.model_name = model_name
        assert self.model_name in {"dum", "nn", "nb", 
            "bag", "lr", "svc", "etc", "ridge", "ada", "gb", "lda",
            "xgc"} or self.model_name.startswith("nn+")
        self.use_features = use_features
        self.consider_chirality = consider_chirality
        
        self.n_bits = n_bits
        self.k = k
        self.cv = cv
        self.alpha = alpha
        self.n_proc = n_proc

        # store int value of top mols to show
        self.num_similar_mols_to_show = num_similar_mols_to_show

        # load id_to_target
        self.id_to_target = load_json(id_to_target_filename)

        # initialise internal classifier
        model_name = self.model_name  
        if model_name == "dum":
            self.model = DummyClassifier(
                strategy="stratified",
            )
        elif model_name == "nn":
            self.model = KNeighborsClassifier(
                n_neighbors=self.k,
                metric="jaccard", 
                algorithm="brute", 
                n_jobs=self.n_proc,
                )
        elif model_name == "nb":
            self.model = BernoulliNB(
                alpha=alpha,
            )
        elif model_name.startswith("nn+"):
            self.model = NearestNeighbors(
                n_neighbors=self.k,
                metric="jaccard",
                n_jobs=self.n_proc,
            )
        elif model_name == "svc":
            self.model = SVC(
                probability=True,
            )
        elif model_name == "bag":
            self.model = BaggingClassifier(
                n_jobs=self.n_proc,
            )
        elif model_name == "lr":
            self.model = LogisticRegressionCV(
                max_iter=1000,
                n_jobs=self.n_proc,
            )
        elif model_name == "ada":
            self.model = AdaBoostClassifier()
        elif model_name == "gb":
            self.model = GradientBoostingClassifier()
        elif model_name == "lda":
            self.model = LinearDiscriminantAnalysis()
        elif model_name == "etc":
            self.model = ExtraTreesClassifier( # capable of multilabel classification out of the box
                n_estimators=500,
                bootstrap=True, 
                max_features="log2",
                min_samples_split=10,
                max_depth=5,
                min_samples_leaf=3,
                verbose=True,
                n_jobs=n_proc,
            ) 
        elif model_name == "ridge":
            self.model = RidgeClassifierCV()
        elif model_name == "xgc":
            raise NotImplementedError
            self.model = XGBClassifier(
                n_jobs=self.n_proc,
                verbosity=0,
            )
        else:
            raise NotImplementedError(f"{model_name} not yet implemented!")
        
    def fit(self, training_molecules, training_molecule_labels):
        """
        """
        # assert isinstance(X, pd.Series) # WHY DID I DO THIS?!
        assert isinstance(training_molecules, list)
        num_training_molecules = len(training_molecules)
        # assert X.shape[0] == y.shape[0]
        assert num_training_molecules == training_molecule_labels.shape[0]

        # save training set molecule ids
        self.training_mol_ids = np.array([
            
            training_molecule["molecule_id"]
            for training_molecule in training_molecules
        ])


        print ("Fitting Bioactivity Prediction model", 
            "({}-{})".format(self.fp, self.model_name),
            "to", num_training_molecules, "SMILES")

        if  len(training_molecule_labels.shape) == 1:
            print ("Fitting in the single target setting")
            self.multi_label = False
        else:
            n_targets = training_molecule_labels.shape[1]
            print ("Fitting in the multi-target setting for", n_targets, "target(s)")
            self.multi_label = True

        if self.multi_label:
            if self.model_name not in support_multi_label:
                pass
            elif self.model_name.startswith("nn+"):
                pass 
            else:
                print ("Wrapping classifier in OneVsRestClassifier")
                self.model = OneVsRestClassifier( # wrap classifier in OneVsRestClassifier for multi-label case
                    self.model,
                    n_jobs=self.n_proc,
                )
                # create instance of model for each target to train in series (using all threads for each model)
                # self.model = [self.model() for _ in range(n_targets)]

        if hasattr(self, "use_features"):
            use_features = self.use_features
        else:
            use_features = True
        if hasattr(self, "consider_chirality"):
            consider_chirality = self.consider_chirality
        else:
            consider_chirality = False

        # convert X to fingerprint
        training_molecules = compute_molecular_fingerprints(
            training_molecules, 
            fps=self.fp, 
            n_bits=self.n_bits,
            use_features=use_features,
            consider_chirality=consider_chirality, 
            n_proc=self.n_proc,
            )

        assert isinstance(training_molecules, sp.csr_matrix), type(training_molecules)
        assert training_molecules.shape[0] == training_molecule_labels.shape[0]

        if self.model_name.startswith("nn+"): # keep training data references for local classifier fitting
            print ("Model", self.model_name, "uses local classifier, saving reference to training data in sparse form")
            self.X = training_molecules # save as sparse
            self.y = training_molecule_labels

        if self.model_name in dense_input: # cannot handle sparse input
            print ("Model", self.model_name, "requires dense input, converting training data to dense")
            training_molecules = training_molecules.A # convert to dense for training

        if self.model_name.startswith("nn+"):
            print ("Model", self.model_name, "requires an nearest neighbours model, fitting it")
            # assert isinstance(X, np.ndarray)
            # ensure dense
            if not isinstance(training_molecules, np.ndarray):
                training_molecules = training_molecules.A
            self.model.fit(training_molecules) # fit nearest neighbours

        elif self.model is not None:
            print ("Fitting", self.model_name, "model to", 
                training_molecules.shape[0], "'", self.fp, "' fingerprints", 
                "of shape", training_molecules.shape, 
                "for", n_targets, "target",
                "using", self.n_proc, "process(es)")

            self.model.fit(training_molecules, training_molecule_labels)

            # if self.multi_label:
            #     print ("MULTI-LABEL SETTING: FITTING EACH MODEL IN SERIES")
            #     assert isinstance(self.model, list)
            #     for i in range(n_targets):
            #         print ("FITTING MODEL TO TARGET NUMBER", i+1, "/", n_targets)
            #         self.model[i].fit(X, y[:,i].A)
            #         print ("COMPLETED FITTING MODEL TO TARGET NUMBER", i+1, "/", n_targets)
            #         print()
            # else:
            #     print ("FITTING IN SINGLE TARGET SETTING")
            #     self.model.fit(X, y)

        return self

    def local_nn_prediction(
        self, 
        queries, 
        mode: str = "predict_proba",
        return_similar_mols: bool = False,
        precomputed_nn_path: str = None,
        return_chembl_matches: bool = True,
        verbose: bool = True,
        ):
        assert mode == "predict_proba"
        if isinstance(queries, sp.csr_matrix):
            queries = queries.A

        X = self.X
        y = self.y
        n_training_molecules = X.shape[0]

        model_name = self.model_name

        k = int(self.k)
        k = min(k, n_training_molecules)
        n_queries = queries.shape[0]

        if verbose:
            print ("Fitting local classifier", model_name, "for", n_queries, "query molecule(s)")
            print ("Using the", k, "most similar molecules to build classifier")

        if precomputed_nn_path is not None and os.path.exists(precomputed_nn_path):
            if verbose:
                print ("Loading precomputed nearest neighbours from", precomputed_nn_path)
            # neighbour_distances, neighbour_indices = load_compressed_pickle(pickled_nn_path)
            neighbour_distances, neighbour_indices = np.load(precomputed_nn_path)

        else:

            nn_nproc = self.n_proc
            # nn_nproc = 24
            # set to 2000 to reuse for other settings of k in evaluation
            nn_n_neighbours = 2000
            # reduce k if there is not enough training molecules
            nn_n_neighbours = min(nn_n_neighbours, n_training_molecules)

            if verbose:
                print ("Determining", nn_n_neighbours ,"nearest neighbours from", 
                    n_training_molecules, "molecules for", n_queries, 
                    "query molecule(s) using", nn_nproc, "process(es)")

            # update model n_jobs
            self.model.n_jobs = nn_nproc

            neighbour_distances, neighbour_indices = self.model.kneighbors(
                queries, 
                n_neighbors=nn_n_neighbours, # compute k=nn_n_neighbours NN (less may be selected below) 
                return_distance=True,
                )

            if precomputed_nn_path is not None:
                if verbose:
                    print ("Saving computed nearest neighbours to", precomputed_nn_path)
                np.save(precomputed_nn_path, (neighbour_distances, neighbour_indices))
        
        # select k (may have loaded greater k value)
        neighbour_distances = neighbour_distances[:,:k] 
        neighbour_indices = neighbour_indices[:,:k].astype(int)

        assert neighbour_distances.shape[0] == n_queries
        assert neighbour_distances.shape[1] == k 
        assert neighbour_indices.shape[0] == n_queries
        assert neighbour_indices.shape[1] == k 

        if verbose:
            print ("Determined neighbours of shape:", neighbour_indices.shape, )
            print ("Now building local classifiers for each query molecule")

        if hasattr(self, "alpha"):
            alpha = self.alpha
        else:
            alpha = 1e-0

        # if hasattr(self, "cv"):
        #     cv = self.cv 
        # else:
        #     cv = np.inf

        # base classifier
        if "nb" in model_name:
            if verbose:
                print ("Using NaiveBayes as base classifier")
            base_classifier = BernoulliNB(
                alpha=alpha, 
                # fit_prior=True, # seems to just divide probabilities ~ 100 but does not affect ranking very much
                fit_prior=False, 
                binarize=None, 
            ) 
        elif "tree" in model_name:
            if verbose:
                print ("Using decision tree as as classifier")
            base_classifier = DecisionTreeClassifier()
        else:
            raise NotImplementedError("Base classifier is not implemented")

        remove_zero_variance = "var" in model_name
        if verbose and remove_zero_variance:
            print ("Removing features with zero variance")

        use_unique_columns_only = "unique" in model_name
        if verbose and use_unique_columns_only:
            print ("Removing perfectly correlated features")

        # statistical feature selection
        if "chi2(" in model_name:
            if verbose:
                print ("Selecting features based on chi2 statistics")
            base_classifier = Pipeline([
                ("feature_selection", SelectPercentile(
                        percentile=10,
                        score_func=chi2,
                    )
                ),
                ("classifier", base_classifier),
            ])

        # ensemble
        n_estimators = 10

        if "bag(" in model_name:
            if verbose:
                print ("Ensembling using bagging")
                print ("Using", n_estimators, "estimators")
            base_classifier = BaggingClassifier(
                base_estimator=base_classifier,
                n_estimators=n_estimators, 
                bootstrap=True, # samples
                bootstrap_features=True, # features
                n_jobs=1,
                )
        elif "ada(" in model_name:
            if verbose:
                print ("Ensembling using (ada)boosting")
                print ("Using", n_estimators, "estimators")
            base_classifier = AdaBoostClassifier(
                base_estimator=base_classifier,
                n_estimators=n_estimators, 
                )

        # calibration
        # non_calibrated_classifier = base_classifier
        # calibrated_classifier = CalibratedClassifierCV(
        #     base_classifier, 
        #     method="sigmoid", 
        #     cv=cv, 
        #     ensemble=True)

        if self.multi_label:
            base_classifier = OneVsRestClassifier(base_classifier, n_jobs=1)

        #     non_calibrated_classifier = OneVsRestClassifier(
        #         non_calibrated_classifier, 
        #         n_jobs=1, 
        #         )
        #     calibrated_classifier = OneVsRestClassifier(
        #         calibrated_classifier, 
        #         n_jobs=1,
        #         )

      
        # import pickle as pkl
        # input_filename = "input.pkl"
        # print ("writing", input_filename)
        # with open(input_filename, "wb") as f:
        #     pkl.dump(
        #         (queries, neighbour_distances, neighbour_indices, X, y, k, base_classifier, remove_zero_variance, use_unique_columns_only, return_chembl_matches),
        #         f, 
        #         pkl.HIGHEST_PROTOCOL)
        # raise Exception

        prediction_n_proc = self.n_proc
        # prediction_n_proc = min(self.n_proc, 10)

        all_query_predictions = make_all_query_predictions(
            queries=queries,
            neighbour_distances=neighbour_distances,
            neighbour_indices=neighbour_indices,
            X=X,
            y=y,
            k=k,
            classifier=base_classifier,
            remove_zero_variance=remove_zero_variance,
            use_unique_columns_only=use_unique_columns_only,
            return_chembl_matches=return_chembl_matches,
            chunk_size=1000,
            n_proc=prediction_n_proc,
            verbose=verbose,
        )

        # predictions = np.vstack(predictions)
        assert all_query_predictions.shape[0] == n_queries

        if self.multi_label:
            assert all_query_predictions.shape[1] == self.y.shape[1]

        if not return_similar_mols:
            return all_query_predictions

        assert hasattr(self, "training_mol_ids")
        training_mol_ids = self.training_mol_ids
        assert hasattr(self, "num_similar_mols_to_show")
        num_similar_mols_to_show = int(self.num_similar_mols_to_show)

        if verbose:
            print ("Returning top", num_similar_mols_to_show, "similar molecules to query molecule(s)")
        # convert from distance to similarity 
        neighbour_similarities = 1 - neighbour_distances
        del neighbour_distances
        most_similar_mols = [ # return as list of dicts
            [ 
                {
                    "molecule_chembl_id": training_mol_ids[neighbour_indices[query_no, similar_mol_id]],
                    "similarity": neighbour_similarities[query_no, similar_mol_id],
                }
                for similar_mol_id in range(min(num_similar_mols_to_show, k))
            ]
            for query_no in range(n_queries)
        ]

        return all_query_predictions, most_similar_mols
    
    def predict(self, X):
        raise NotImplementedError
        # NOT FULLY IMPLEMENTED
        print ("predicting for", X.shape[0], 
            "query molecule(s)")

        if hasattr(self, "use_features"):
            use_features = self.use_features
        else:
            use_features = True
        if hasattr(self, "consider_chirality"):
            consider_chirality = self.consider_chirality
        else:
            consider_chirality = False
        
        X = compute_molecular_fingerprints(
            X, 
            fp=self.fp, 
            n_bits=self.n_bits,
            use_features=use_features,
            consider_chirality=consider_chirality,
            n_proc=self.n_proc, 
        )  
        # remove none molecules from prediction
        X = sp.vstack([x for x in X if x is not None])

        print ("performing prediction",
            "using", self.n_proc, "processes")

        if self.model_name == "nn+nb":

            return self._local_nn_prediction(
                X,
                mode="predict")
        else:
            if self.model_name in dense_input \
                and not isinstance(X, np.ndarray):
                X = X.A
            assert hasattr(self.model, "predict")

            with parallel_backend('threading', n_jobs=self.n_proc):
                return self.model.predict(X)

    def predict_proba(  
        self, 
        X, 
        return_similar_mols=False,
        precomputed_query_fp_path=None,
        precomputed_nn_path=None,
        return_chembl_matches=True,
        verbose: bool = True,
        ):
        # assert isinstance(X, pd.Series)
        # num_queries = X.shape[0]
        assert isinstance(X, list)
        num_queries = len(X)

        if verbose:
            print ("Predicting probabilities for", num_queries, "query molecule(s)")

        if precomputed_query_fp_path is not None and os.path.exists(precomputed_query_fp_path):
            if verbose:
                print ("Loading precomputed fingerprints from", precomputed_query_fp_path)
            X = sp.load_npz(precomputed_query_fp_path)

        else:
            if verbose:
                print ("Computing", self.fp, "fingerprints for", num_queries, "query molecule(s)")
            if hasattr(self, "use_features"):
                use_features = self.use_features
            else:
                use_features = True
            if hasattr(self, "consider_chirality"):
                consider_chirality = self.consider_chirality
            else:
                consider_chirality = False
        
            X = compute_molecular_fingerprints(
                X, 
                fps=self.fp, 
                n_bits=self.n_bits, 
                use_features=use_features,
                consider_chirality=consider_chirality,
                n_proc=self.n_proc,
                )
            if precomputed_query_fp_path is not None:
                if verbose:
                    print ("Saving computed fingerprints to", precomputed_query_fp_path)
                sp.save_npz(precomputed_query_fp_path, X)

        if verbose:
            print ("Performing probability prediction using", self.n_proc, "process(es)")
            if return_similar_mols:
                print ("Returning most similar molecules")
        
        # remove None molecules from prediction (mols_did not sanitise)
        # X = sp.vstack([x for x in X if x is not None])
        assert isinstance(X, sp.csr_matrix), type(X)

        if self.model_name.startswith("nn+"):
            # nn+XXX variant
            if verbose:
                print ("Using local classifier")
            return self.local_nn_prediction(
                X,
                mode="predict_proba",
                return_similar_mols=return_similar_mols,
                precomputed_nn_path=precomputed_nn_path,
                return_chembl_matches=return_chembl_matches,
                verbose=verbose,
                )

        else:
            raise NotImplementedError

        if self.model_name in dense_input \
            and not isinstance(X, np.ndarray):
            X = X.A

        if self.model_name in support_multi_label:
            # with parallel_backend('threading', n_jobs=self.n_proc):
            probs = self.model.predict_proba(X) # handle missing classes correctly
            classes = self.model.classes_
            return np.hstack([p[:,idx] if idx.any() else 1-p
                for p, idx in zip(probs, classes)]) # check for existence of positive class
        
        elif not isinstance(self.model, OneVsRestClassifier):
            assert isinstance(self.model, list)
            print ("MODEL IS A LIST")
            predictions = []
            for i, m in enumerate(self.model):
                probs = m.predict_proba(X)
                classes = m.classes_ 
                predictions.append(probs[:,classes])
                print ("completed prediction for target", i+1, "/", len(self.model))
            predictions = np.hstack(predictions)
            assert predictions.shape[0] == X.shape[0]
            assert predictions.shape[1] == len(self.model)
            return predictions

        else: 
            assert isinstance(self.model, OneVsRestClassifier)
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(X)
            elif hasattr(self.model, "decision_function"):
                print ("predicting with decision function")
                return self.model.decision_function(X)
            else:
                raise Exception

    def decision_function(self, X):
        raise NotImplementedError
        # not fully implemented
        print ("computing decision function for", X.shape[0], 
            "query molecules")

        if hasattr(self, "use_features"):
            use_features = self.use_features
        else:
            use_features = True
        if hasattr(self, "consider_chirality"):
            consider_chirality = self.consider_chirality
        else:
            consider_chirality = False
    
        X = compute_molecular_fingerprints(
            X, 
            fp=self.fp,
            n_bits=self.n_bits,
            use_features=use_features,
            consider_chirality=consider_chirality,
            n_proc=self.n_proc)
        
        # remove none molecules from prediction
        X = sp.vstack([x for x in X if x is not None])
        
        print ("determining decision function",
            "using", self.n_proc, "processes")
        if self.model_name == "nn+nb":
           
            return self._local_nn_prediction(
                X,
                mode="predict_proba") # NB does not have a decision function

        if self.model_name in dense_input \
            and not isinstance(X, np.ndarray):
            X = X.A

        if self.model_name in support_multi_label: # k neigbours has no decision function
            with parallel_backend('threading', n_jobs=self.n_proc):
                probs = self.model.predict_proba(X) # handle missing classes correctly
            classes = self.model.classes_
            return np.hstack([probs[:,idx] if idx.any() else 1-probs
                for probs, idx in zip(probs, classes)]) # check for existence of positive class

        else:
            assert isinstance(self.model, OneVsRestClassifier)

            if hasattr(self.model, "decision_function"):
                with parallel_backend('threading', n_jobs=self.n_proc):
                    return self.model.decision_function(X)
            elif hasattr(self.model, "predict_proba"):
                print ("predicting using probability")
                with parallel_backend('threading', n_jobs=self.n_proc):
                    return self.model.predict_proba(X)
            else:
                raise Exception

    def check_is_fitted(self):
        if self.model is None:
            return True
        try:
            check_is_fitted(self.model)
            return True
        except NotFittedError:
            return False

    def __str__(self):
        return f"NPAIPredictor({self.fp}-{self.model_name})"

    def set_n_proc(self, n_proc):
        print ("Setting n_proc =", n_proc)
        self.n_proc = n_proc
        if self.model is not None:
            if isinstance(self.model, list):
                for m in self.model:
                    m.n_jobs = n_proc
            else:
                self.model.n_jobs = n_proc

    def set_k(self, k):
        self.k = k 
        if isinstance(self.model, KNeighborsClassifier):
            self.model.n_neighbors = k

if __name__ == "__main__":

    # import glob 

    # target_to_id = load_json("/media/david/Elements/data/chembl/training_data/max_actives=None/target_to_id.json")
    # id_to_target = {v:k for k, v in target_to_id.items()}
    # del target_to_id

    # for model_filename in glob.iglob("models/max_actives=None/*.pkl.gz"):
    #     print (model_filename)
    #     model = load_model(model_filename)

    #     if hasattr(model, "id_to_target"):
    #         continue
    
    #     model.id_to_target = id_to_target
    #     save_model(model, model_filename)

    print (canonise_fp("rdk_maccs+torsion+morg3"))
    print (canonise_fp("torsion+morg3+rdk_maccs"))
    print (canonise_fp(None))