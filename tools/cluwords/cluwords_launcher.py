from tools.cluwords.script_functions import create_embedding_models, generate_topics
import os

"""
This is the main launcher method to run CluWords through a notebook
It needs to be predefined to include your local CluWords setup, see below
Afterwards CluWords can be run easily from within notebooks by calling generate_cluwords()
"""

MAIN_PATH='tools/cluwords'
EMBEDDING_RESULTS = 'multi_embedding'
PATH_TO_SAVE_RESULTS = '{}/cluwords/{}/results'.format(MAIN_PATH, EMBEDDING_RESULTS)
PATH_TO_SAVE_MODEL = '{}/cluwords/{}/datasets/gn_w2v_models'.format(MAIN_PATH, EMBEDDING_RESULTS)
DATASETS_PATH= '{}/cluwords/{}/datasets'.format(MAIN_PATH, EMBEDDING_RESULTS)
CLASS_PATH = '{}/cluwords/{}/acm_so_score_Pre'.format(MAIN_PATH, EMBEDDING_RESULTS)
HAS_CLASS = False

try:
    os.mkdir('{}/cluwords'.format(MAIN_PATH))
    os.mkdir('{}/cluwords/{}'.format(MAIN_PATH, EMBEDDING_RESULTS))
    os.mkdir('{}/cluwords/{}/results'.format(MAIN_PATH, EMBEDDING_RESULTS))
    os.mkdir('{}/cluwords/{}/datasets'.format(MAIN_PATH, EMBEDDING_RESULTS))
    os.mkdir('{}/cluwords/{}/datasets/gn_w2v_models'.format(MAIN_PATH, EMBEDDING_RESULTS))
except FileExistsError:
    pass

# Run CluWords with the specified settings.
def generate_cluwords(filename, embedding_path, embedding_bin=False, threads=4, components=20, algorithm='knn_cosine'):
    DATASET = filename
    EMBEDDINGS_FILE_PATH = embedding_path
    EMBEDDINGS_BIN_TYPE = embedding_bin
    N_THREADS = threads
    N_COMPONENTS = components
    ALGORITHM_TYPE = algorithm

    print('Filter embedding space to {} dataset...'.format(DATASET))
    n_words = create_embedding_models(dataset=DATASET,
                                      embedding_file_path=EMBEDDINGS_FILE_PATH,
                                      embedding_type=EMBEDDINGS_BIN_TYPE,
                                      datasets_path=DATASETS_PATH,
                                      path_to_save_model=PATH_TO_SAVE_MODEL)


    print('Build topics...')
    results = generate_topics(dataset=DATASET,
                              word_count=n_words,
                              path_to_save_model=PATH_TO_SAVE_MODEL,
                              datasets_path=DATASETS_PATH,
                              path_to_save_results=PATH_TO_SAVE_RESULTS,
                              n_threads=N_THREADS,
                              algorithm_type=ALGORITHM_TYPE,
                              # k=n_words,
                              k=500,
                              threshold=0.4,
                              cossine_filter=0.9,
                              class_path=CLASS_PATH,
                              has_class=HAS_CLASS,
                              n_components=N_COMPONENTS)
    return PATH_TO_SAVE_RESULTS
