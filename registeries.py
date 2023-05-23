from models import *
from helper import EminagaAlgorthimPreprocess
model_registry = {
    "ModelBasic": ModelOnnx,
    "ModelOnnx": ModelOnnx}

preprocess_registry = {
    "None":None,
    None:None,
    "EminagaAlgorthimPreprocess":EminagaAlgorthimPreprocess
}