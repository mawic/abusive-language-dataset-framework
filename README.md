# Bias and Comparison Framework for Abusive Language Datasets

## Abstract
Increasing research activities in the field of automatic abusive language or hate speech detection have produced numerous datasets in recent years. A problem with this diversity is that they often differ, among other things, in context, platform, sampling process, collection strategy, and labeling schema. There have been surveys on these datasets, but they compare the datasets only superficially. Therefore, we develop a bias and comparison framework for abusive language datasets to conduct an in-depth analysis of them and provide a comparison of five partially famous datasets based on the framework. We make this framework available to researchers and data scientists who work with such datasets to be aware of the datasets' properties and consider them in their work.


## Data
We do not publish the datasets with the framework. All datasets are publicly available. If you require the crawled/hydrated tweets, feel free to contact us.

### Data pipeline
If you want to include your own data sets for analysis, you need to implement equivalent methods to these to load your own data in the right format. If you need guidance you can check on existing pipelines in the 'pipelines' subfolder.
Every data pipeline consist of three methods:
* get_data(): It returns a list of dicitionaries. The dictionaries have at least the fields 'label' and 'text'. The list contains the full dataset.
* get_data_binary():  It returns a list of dicitionaries. The dictionaries have at least the fields 'label' and 'text'. The list contains the full dataset. The labels are only 'abusive' and 'neutral'.
* get_user_data():  It returns a list of dicitionaries. The dictionaries have at least the fields 'label', 'text', and 'user''id' (nested dictionaries).

### Links to datasets
 * Wasseem: https://github.com/ZeerakW/hatespeech
 * Davidson: https://github.com/t-davidson/hate-speech-and-offensive-language
 * Founta: https://zenodo.org/record/2657374#.X7uOg81Khm8
 * Zampieri: https://competitions.codalab.org/competitions/20011#learn_the_details
 * Vidgen: https://zenodo.org/record/3816667#.X7uOps1Khm8
 * Albadi: https://github.com/nuhaalbadi/Arabic_hatespeech
 * Alsafari: Please contact the authors (see paper for reference)
 * Alshalan: https://github.com/raghadsh/Arabic-Hate-speech
 * Chowdhury: https://github.com/shammur/Arabic-Offensive-Multi-Platform-SocialMedia-Comment-Dataset
 * Mubarak: https://competitions.codalab.org/competitions/22825 / https://competitions.codalab.org/competitions/22826
 * Mulki: https://github.com/Hala-Mulki/L-HSAB-First-Arabic-Levantine-HateSpeech-Dataset

## Setup
 * Get data sets and store them in ./data.
 * Update the data pipelines in ./pipelines. These files load the files and make sure that the output is in the correct format.
 * Update the config.yaml with the datasets you want to analyze. You need to provide the name of the dataset and which labels are the non-hate labels.
 * Download word FASTTEXT embeddings crawl-300d-2M.vec.zip (https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip) AND cc.de.300.bin (https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz)
 * Update the paths to the word embeddings in 02_semantic_perspective_a_similarities.ipynb and 02_semantic_perspective_b_topic_model.ipynb
 * Run 00_init_framework.ipynb to create folders and install requirements.txt


## Computing infrastucture
We used an AWS EC2 G4 (g4dn.2xlarge) instance with Deep Learning AMI (Ubuntu 18.04) Version 36.0 (ami-063585f0e06d22308). As Python kernel we used conda_pytorch_latest_p37 (Python version 3.7.6). The additional packages that we installed can be found in requirement.txt. Requirements_full.txt contains a full list of all packages.
