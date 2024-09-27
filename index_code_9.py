from llments.datastore.pyserini_datastore import PyseriniDatastore
import pandas as pd
import numpy as np
import torch

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")          # GPU available, select GPU as device
        print("CUDA is available! Using GPU.")
    else:
        device = torch.device("cpu")           # No GPU available, fall back to CPU
        print("CUDA is not available. Using CPU.")

    datastore = PyseriniDatastore(index_path='/data/tir/projects/tir7/user_data/mihirban/NQ/colbert/NQ_index_passage-8',
                                  document_path='/data/tir/projects/tir6/general/afreens/dbqa/data/corpus_files/wiki_par.jsonl',
                                  index_encoder='colbert-ir/colbertv2.0',
                                  fields=['contents'],
                                  docid_field="id",
                                  pooling='mean',
                                  to_faiss=True,
                                  device=device,
                                  shard_id=8,
                                  shard_num=10)
    # Your further code utilizing 'datastore' object can be added here

if __name__ == "__main__":
    main()
