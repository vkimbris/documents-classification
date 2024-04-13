import gensim
import numpy as np

from gensim.corpora.dictionary import Dictionary

from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.coherence_metrics import Coherence

from typing import List, Tuple, Dict

from .regs import BaseRegularizer


class PLSI:

    def __init__(self, 
                 dataset: Dataset,
                 num_topics: int = 10,
                 regularizers: Dict[str, List[BaseRegularizer]]| None = None,
                 seed: int = 42) -> None:
        
        self.dataset = dataset

        self.corpus = dataset.get_corpus()
        self.vocabulary = Dictionary(documents=self.corpus)
        self.num_topics = num_topics
        self.regularizers = regularizers
        self.seed = seed

        self.n_documents = len(self.corpus)
        self.n_words = len(self.vocabulary)

        self.theta, self.phi = self.__initialize_parameteres()

        self.bow_matrix = self.__transform_corpus_to_bow_matrix()

    def train(self, max_iter: int, threshold: float = 1e-3, verbose: bool = True):
        for k in range(max_iter):
            previous_theta, previous_phi = self.theta, self.phi

            self._m_step(self._e_step())

            residue = np.sum((previous_theta - self.theta) ** 2) + np.sum((previous_phi - self.phi) ** 2)

            if verbose:
                print(f"Iteration: {k}, Residue: {residue}\n")

            if residue < threshold:
                break

        return self.__create_model_output()
    
    def get_topics(self, topk: int = 10):
        topics_indices = np.argpartition(self.phi, -topk, axis=1)[:, -topk:]

        id_to_word_func = np.vectorize(lambda i: self.vocabulary[i])

        topics = id_to_word_func(topics_indices)
        topics = list(map(list, topics))

        return topics
    
    def get_representative_docs(self, path_to_original_corpus: str, top_n_docs: int = 5):
        path_to_original_indexes = self.dataset.dataset_path + "/indexes.txt"

        with open(path_to_original_indexes, "r") as f:
            original_indexes = f.readlines()

        with open(path_to_original_corpus, "r") as f:
            original_corpus = f.readlines()
        
        original_indexes = list(map(int, original_indexes))
        original_indexes = np.array(original_indexes)

        representative_docs_indexes = np.argsort(self.theta, axis=0)[-top_n_docs:, :]
        representative_docs_indexes = original_indexes[representative_docs_indexes]
        
        representative_docs = {}
        for topic_id in range(self.num_topics):
            representative_docs[topic_id] = []

            for i in representative_docs_indexes[:, topic_id]:
                representative_docs[topic_id].append(original_corpus[i])

        return representative_docs

    def _e_step(self) -> np.ndarray:
        denominator = np.matmul(self.theta, self.phi)
        
        theta = self.theta.T.reshape(self.num_topics, self.n_documents, 1)
        phi = self.phi.reshape(self.num_topics, 1, self.n_words)

        nominator = np.matmul(theta, phi)

        return self.bow_matrix * nominator / np.where(denominator < 1e-7, 1e-7, denominator)

    def _m_step(self, documents_words_topics_counter: np.ndarray):
        theta_regs_components, phi_regs_components = self._get_regularizers_components()
        
        new_theta = documents_words_topics_counter.sum(axis=2)
        new_theta = new_theta + theta_regs_components
        new_theta = new_theta / documents_words_topics_counter.sum(axis=2).sum(axis=0)
        new_theta = new_theta.T

        new_theta = np.where(new_theta < 0, 0, new_theta)

        new_phi = documents_words_topics_counter.sum(axis=1)
        new_phi = new_phi + phi_regs_components
        new_phi = new_phi / documents_words_topics_counter.sum(axis=1).sum(axis=1).reshape(-1, 1)

        new_phi = np.where(new_phi < 0, 0, new_phi)

        self.theta = new_theta
        self.phi = new_phi

    def _get_regularizers_components(self):
        if self.regularizers is None:
            return 0, 0
        
        theta_regs, phi_regs = self.regularizers["theta"], self.regularizers["phi"]

        theta_regs_components = 0
        for theta_reg in theta_regs:
            theta_regs_components = theta_regs_components + theta_reg.get_component(self.phi, self.theta)

        phi_regs_components = 0
        for phi_reg in phi_regs:
            phi_regs_components = phi_regs_components + phi_reg.get_component(self.phi, self.theta)

        return theta_regs_components, phi_regs_components

    def __initialize_parameteres(self) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(self.seed)

        theta = np.random.uniform(low=0, high=1, size=(self.n_documents, self.num_topics))
        theta = theta / theta.sum(axis=1).reshape(-1, 1)

        phi = np.random.uniform(low=0, high=1, size=(self.num_topics, self.n_words))
        phi = phi / phi.sum(axis=1).reshape(-1, 1)

        return theta, phi
    
    def __transform_corpus_to_bow_matrix(self) -> List[List[Tuple[int, int]]]:
        bag_of_words = [self.vocabulary.doc2bow(doc) for doc in self.corpus]

        bag_of_words_matrix = np.zeros((self.n_documents, self.n_words))

        for k, document in enumerate(bag_of_words):
            for word_id, word_amount in document:
                bag_of_words_matrix[k][word_id] = word_amount

        return bag_of_words_matrix
    
    def __create_model_output(self):
        model_output = {
            "topic-word-matrix": self.phi,
            "document-topic-matrix": self.theta,
            "topics": self.get_topics()
        }

        return model_output