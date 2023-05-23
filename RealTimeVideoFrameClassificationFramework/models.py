'''
Copyright by Okyaz Eminaga 2023
'''
from abc import ABC
import numpy as np
import onnxruntime as rt
from typing import Any, Dict, List, Tuple
import utils
#from plexusnet.architecture import LoadModel

class ModelBasic(ABC):
    def __init__(self, filename: str, preprocess : Any, cfg: Dict, positive_label: List=[], negative_label: List=[], labels: Dict ={}, threshold: float=1) -> None:
        """_summary_

        Args:
            filename (str): the file name of the model
            preprocess (a method): a function for preprocessing the image
            cfg (dict): the model configuration
            positive_label (list, optional): the name list of positive labels. Defaults to [""].
            negative_label (list, optional): the name list of negative labels. Defaults to [""].
            labels (dict, optional): the label and the corresponding index (e.g. {"cancer" : 1, "benign":0}) . Defaults to {}.
            threshold (int, optional): the decision threshold to determine the positive label. Defaults to 1.
        """
        self.filename = filename
        self.model = None#LoadModel(filename)
        self.preprocess=preprocess
        self.positive_label=positive_label
        self.negative_label=negative_label
        self.labels = labels
        self.threshold=threshold
        self.cfg = cfg
        super().__init__()
    def Prediction(self, img):
        '''
        if self.preprocess is not None:
            img=self.preprocess(img)
        
        pred_ = self.model.predict_on_batch(np.array([img]))
        
        positive_score = 0
        for p_lbl in self.positive_label:
            positive_score += pred_[0][self.labels[p_lbl]]
        
        negative_score = 0

        for n_lbl in self.negative_label:
            negative_score += pred_[0][self.labels[n_lbl]]

        OR = positive_score/(negative_score+1e-8)
        ROI_status = OR>self.threshold
        '''
        return [None], [None], [None]#[OR],[ROI_status], [None]
class ModelOnnx(ModelBasic):
    def __init__(self, filename, preprocess, cfg: Dict, positive_label=[""], negative_label=[""], labels={}, threshold=1, GPU=True) -> None:
        if GPU:
            self.model = rt.InferenceSession(filename, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])#, providers=providers)
        else:
            self.model = rt.InferenceSession(filename, providers=["CPUExecutionProvider"])
        self.preprocess=preprocess
        self.positive_label=positive_label
        self.negative_label=negative_label
        self.labels = labels
        self.threshold=threshold
        self.cfg=cfg
    def Prediction(self, img):
        if self.preprocess is not None:
            img=self.preprocess(img)
        pred_ = self.model.run([self.model.get_outputs()[0].name], {self.model.get_inputs()[0].name: np.array([img], dtype=np.float32)})
        positive_score = 0
        for p_lbl in self.positive_label:
            positive_score += pred_[0][0][self.labels[p_lbl]]

        negative_score = 0
        for n_lbl in self.negative_label:
            negative_score += pred_[0][0][self.labels[n_lbl]]
        
        OR = positive_score/(negative_score+1e-8)
        ROI_status = OR>self.threshold
        return [OR],[ROI_status], [None]

class ModelOnnxWithHeatmap(ModelBasic):
    def __init__(self, filename, preprocess, cfg: Dict, positive_label=[""], negative_label=[""], labels={}, threshold=1, GPU=True) -> None:
        if GPU:
            self.model = rt.InferenceSession(filename, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])#, providers=providers)
        else:
            self.model = rt.InferenceSession(filename, providers=["CPUExecutionProvider"])
        self.preprocess=preprocess
        self.positive_label=positive_label
        self.negative_label=negative_label
        self.labels = labels
        self.threshold=threshold
        self.cfg=cfg
    def Prediction(self, img):
        if self.preprocess is not None:
            img=self.preprocess(img)
        heatmap, pred_ = self.model.run([self.model.get_outputs()[0].name], {self.model.get_inputs()[0].name: np.array([img], dtype=np.float32)})
        positive_score = 0
        for p_lbl in self.positive_label:
            positive_score += pred_[0][0][self.labels[p_lbl]]

        negative_score = 0
        for n_lbl in self.negative_label:
            negative_score += pred_[0][0][self.labels[n_lbl]]
        
        OR = positive_score/(negative_score+1e-8)
        ROI_status = OR>self.threshold
        return [OR],[ROI_status], [heatmap]