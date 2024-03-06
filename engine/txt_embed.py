import numpy as np
import math
import time
import functools
#import scann
# no scann available for windows.

from vertexai.language_models import TextEmbeddingModel
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

def generate_batches(sentences, batch_size = 5):
    for i in range(0, len(sentences), batch_size):
        yield sentences[i : i + batch_size]

def encode_texts_to_embeddings(sentences):
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    try:
        embeddings = model.get_embeddings(sentences)
        return [embedding.values for embedding in embeddings]
    except Exception:
        return [None for _ in range(len(sentences))]

def encode_text_to_embedding_batched(sentences, api_calls_per_second=0.33, batch_size=5):
    # Generates batches and calls embedding API

    embeddings_list = []

    # Prepare the batches using a generator
    batches = generate_batches(sentences, batch_size)

    seconds_per_job = 1 / api_calls_per_second

    with ThreadPoolExecutor() as executor:
        futures = []
        for batch in tqdm(
                batches, total=math.ceil(len(sentences) / batch_size), position=0
        ):
            futures.append(
                executor.submit(functools.partial(encode_texts_to_embeddings), batch)
            )
            time.sleep(seconds_per_job)

        for future in futures:
            embeddings_list.extend(future.result())

    is_successful = [
        embedding is not None for sentence, embedding in zip(sentences, embeddings_list)
    ]
    embeddings_list_successful = np.squeeze(
        np.stack([embedding for embedding in embeddings_list if embedding is not None])
    )
    return embeddings_list_successful


'''
# configure ScaNN as a tree - asymmetric hash hybrid with reordering
# anisotropic quantization as described in the paper; see README
def create_index(embedded_dataset,
                 num_leaves,
                 num_leaves_to_search,
                 training_sample_size):
    # normalize data to use cosine sim as explained in the paper
    normalized_dataset = embedded_dataset / np.linalg.norm(embedded_dataset, axis=1)[:, np.newaxis]

    searcher = (
        scann.scann_ops_pybind.builder(normalized_dataset, 10, "dot_product")
        .tree(
            num_leaves=num_leaves,
            num_leaves_to_search=num_leaves_to_search,
            training_sample_size=training_sample_size,
        )
        .score_ah(2, anisotropic_quantization_threshold=0.2)
        .reorder(100)
        .build()
    )
    return searcher
'''