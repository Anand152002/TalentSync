import deeplake
import os
from config import ACTIVELOOP_TOKEN, DEEPLAKE_DATASET_PATH

os.environ['ACTIVELOOP_TOKEN'] = ACTIVELOOP_TOKEN

def store_in_deeplake(embeddings, metadatas, dataset_path, overwrite=True):
    if overwrite:
        # Try to create a new empty dataset or clear the existing one
        try:
            ds = deeplake.empty(dataset_path)
            print(f"Created a new dataset at {dataset_path}.")
        except deeplake.util.exceptions.DatasetHandlerError:
            # If the dataset already exists, handle accordingly
            print(f"A dataset already exists at {dataset_path}.")
            return  # Exit or handle as needed
    else:
        # Load the existing dataset if not overwriting
        try:
            ds = deeplake.load(dataset_path)
            print(f"Loaded existing dataset at {dataset_path}.")
        except deeplake.util.exceptions.DatasetHandlerError:
            print(f"Dataset not found at {dataset_path}.")
            return


