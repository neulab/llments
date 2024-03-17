from pyserini.encode import JsonlRepresentationWriter, FaissRepresentationWriter, JsonlCollectionIterator
from pyserini.encode import DprDocumentEncoder, TctColBertDocumentEncoder, AnceDocumentEncoder, AggretrieverDocumentEncoder, AutoDocumentEncoder, CosDprDocumentEncoder
from pyserini.encode import UniCoilDocumentEncoder
from pyserini.encode import OpenAIDocumentEncoder, OPENAI_API_RETRY_DELAY

class Datastore:
    def __init__(self, input_jsonl, output_dir, encoder, fields, device):
        self.input_jsonl = input_jsonl
        self.output_dir = output_dir
        self.encoder = encoder
        self.device = device
        self.fields = fields

    def encode(self, delimiter="\n", docid_field=None, batch_size=64, max_length=256, dimension=768, 
                prefix=None, pooling='cls', l2_norm=False, to_faiss=False, use_openai=False, rate_limit=3500):
        encoder_class_map = {
            "dpr": DprDocumentEncoder,
            "tct_colbert": TctColBertDocumentEncoder,
            "aggretriever": AggretrieverDocumentEncoder,
            "ance": AnceDocumentEncoder,
            "sentence-transformers": AutoDocumentEncoder,
            "unicoil": UniCoilDocumentEncoder,
            "openai-api": OpenAIDocumentEncoder,
            "cosdpr": CosDprDocumentEncoder,
            "auto": AutoDocumentEncoder,
        }

        encoder_class = None

        for class_keyword in encoder_class_map:
            if class_keyword in self.encoder.lower():
                encoder_class = encoder_class_map[class_keyword]
                break

        # if none of the class keyword was matched, use the AutoDocumentEncoder
        if encoder_class is None:
            encoder_class = AutoDocumentEncoder
        
        if "sentence-transformers" in self.encoder:
            pooling = 'mean'
            l2_norm = True
        elif "contriever" in self.encoder:
            pooling = 'mean'
            l2_norm = False
        elif "auto" in self.encoder:
            pooling = 'cls'
            l2_norm = False

        print("Initializing the document encoder ...")

        if encoder_class == AutoDocumentEncoder:
            encoder_instance = encoder_class(model_name=self.encoder, device=self.device, pooling=pooling, l2_norm=l2_norm, prefix=prefix)
        else:
            encoder_instance = encoder_class(model_name = self.encoder, device = self.device)

        if to_faiss:
            embedding_writer = FaissRepresentationWriter(self.output_dir, dimension=dimension)
        else:
            embedding_writer = JsonlRepresentationWriter(self.output_dir)

        collection_iterator = JsonlCollectionIterator(self.input_jsonl, self.fields, docid_field, delimiter)

        if use_openai:
            batch_size = int(rate_limit / (60 / OPENAI_API_RETRY_DELAY))

        print("Building the index ...")

        with embedding_writer:
            for batch_info in collection_iterator(batch_size):
                texts = batch_info['text']
                titles = batch_info['title'] if 'title' in self.fields else None
                expands = batch_info['expand'] if 'expand' in self.fields else None
                fp16 = False
                max_length = max_length
                add_sep = False
                
                embeddings = encoder_instance.encode(texts=texts, titles=titles, expands=expands, fp16=fp16, max_length=max_length, add_sep=add_sep)
                batch_info['vector'] = embeddings
                embedding_writer.write(batch_info, self.fields)

        print("\nIndex creation completed sucessfully!")
