from llments.datastore.datastore import Datastore
from llments.lm.lm import LanguageModel


class RAGLanguageModel(LanguageModel):
    def __init__(self, base: LanguageModel, datastore: Datastore):
        """Apply retrieval-augmented generation over a datastore.

        Args:
            base: The language model to be modified.
            

        Returns:
            LanguageModel: The enhanced language model.
        """
        
        

