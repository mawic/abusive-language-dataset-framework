from script_functions import create_embedding_models, generate_topics
import os
import time

selection = [0.3,0.35,0.4]

for j in selection:
    RUN = str(j)

    ### Paths and files paths
    ## Main path
    MAIN_PATH='C:/Users/airym/Uni/WiSe 1920/Master/Experiments/processing/methods/topic-modeling-cluwords/cluwords'

    ## Embeddings
    # EMBEDDINGS_FILE_PATH = '{}/GoogleNews-vectors-negative300.bin'.format(MAIN_PATH)
    # EMBEDDINGS_BIN_TYPE = True
    EMBEDDINGS_FILE_PATH = "C:/Users/airym/Uni/WiSe 1920/Master/Experiments/processing/resources/crawl-300d-2M.vec"
    EMBEDDINGS_BIN_TYPE = False

    ## Dataset
    DATASETS_PATH = 'C:/Users/airym/Uni/WiSe 1920/Master/Experiments/processing/methods/topic-modeling-cluwords/cluwords/datasets'
    # DATASETS_PATH = '/mnt/d/Work/textual_datasets'
    DATASET = 'waseem'

    ## Dataset-specific w2v
    PATH_TO_SAVE_MODEL = '{}/cluwords{}/dataset/gn_w2v_models'.format(MAIN_PATH, RUN)

    ## Class path
    CLASS_PATH = 'C:/Users/airym/Uni/WiSe 1920/Master/Experiments/processing/methods/topic-modeling-cluwords/cluwords/acm_so_score_Pre'

    ## Algorithm type
    # ALGORITHM_TYPE = 'knn_mahalanobis'
    ALGORITHM_TYPE = 'knn_cosine'

    # Create inital folders
    try:
        os.mkdir('{}/cluwords{}'.format(MAIN_PATH, RUN))
        os.mkdir('{}/cluwords{}/dataset'.format(MAIN_PATH, RUN))
        os.mkdir('{}/cluwords{}/dataset/gn_w2v_models'.format(MAIN_PATH, RUN))
    except FileExistsError:
        pass

    # Create the word2vec models for each dataset
    print('Filter embedding space to {} dataset...'.format(DATASET))
    n_words = create_embedding_models(dataset=DATASET,
                                    embedding_file_path=EMBEDDINGS_FILE_PATH,
                                    embedding_type=EMBEDDINGS_BIN_TYPE,
                                    datasets_path=DATASETS_PATH,
                                    path_to_save_model=PATH_TO_SAVE_MODEL)

    for i in range(5,7):
        print("Round",i, " ",time.time())

        # EMBEDDING_RESULTS = 'fasttext_wiki_mahalanobis'
        #EMBEDDING_RESULTS = 'fasttext_wiki'
        EMBEDDING_RESULTS = 'fasttext_crawl_'+ str(i)
        PATH_TO_SAVE_RESULTS = '{}/cluwords{}/{}/results'.format(MAIN_PATH, RUN, EMBEDDING_RESULTS)
        HAS_CLASS = False
        N_THREADS = 6
        N_COMPONENTS = i

        # Creates directories if they don't exist
        try:
            os.mkdir('{}/cluwords{}/{}'.format(MAIN_PATH, RUN, EMBEDDING_RESULTS))
            os.mkdir('{}/cluwords{}/{}/results'.format(MAIN_PATH, RUN, EMBEDDING_RESULTS))
        except FileExistsError:
            pass

        # Create the word2vec models for each dataset
        '''print('Filter embedding space to {} dataset...'.format(DATASET))
        n_words = create_embedding_models(dataset=DATASET,
                                        embedding_file_path=EMBEDDINGS_FILE_PATH,
                                        embedding_type=EMBEDDINGS_BIN_TYPE,
                                        datasets_path=DATASETS_PATH,
                                        path_to_save_model=PATH_TO_SAVE_MODEL)'''

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
                                threshold=j,
                                #threshold=0.4,
                                cossine_filter=0.9,
                                class_path=CLASS_PATH,
                                has_class=HAS_CLASS,
                                n_components=N_COMPONENTS)
        print(results)
        print("Round",i, " done ", time.time())
        print("---------------------------------------")
