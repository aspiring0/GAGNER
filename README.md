# <Global-Attn-GateNER: Unified Entity Recognition Based on Global Attention and Dynamic Gateway>

## Description
Unified modeling poses a significant challenge for Named Entity Recognition
(NER)for which efficiently capturing semantic and feature fusion remains critical. The W2NER framework, based on word-word relationship classification,offers a unified approach to NER through two-dimensional word grid modeling.However, this framework suffers from insufficient context awareness and rigid feature interactions. To overcome these limitations, this study proposes a phased enhancement strategy consisting of several steps:(1) a global selfattention mechanism strengthens boundary semantic representations; (2) then,a restructured hierarchical perturbation strategy mitigates over-regularization in the dual-affine module; and (3) a dynamically gated fusion network achieves adaptive aggregation of structured features and contextual representations at the word-pair granularity. Extensive experiments across eight benchmark datasets,which cover flat, overlapping, and discontinuous NER (four Chinese and four English datasets),demonstrate this proposed model’s state-of-the-art performance,thereby propelling unified NER to new frontiers.

## Dataset Information
- **CoNLL-2003 (English)**: Newswire NER dataset with four entity types (PER, ORG, LOC, MISC).  
  Original shared task website: https://www.clips.uantwerpen.be/conll2003/ner/.

- **MSRA (Chinese news)**: Chinese NER dataset released for the SIGHAN 2006 shared task (Levow, 2006).  
  Available from the original organizers and licensed mirrors that distribute the SIGHAN 2006 datasets.

- **Resume (Chinese)**: NER dataset of Chinese resumes (Zhang and Yang, 2018).  
  Public release: https://github.com/jiesutd/ChineseNER.

- **Weibo (Chinese social media)**: NER dataset constructed from Sina Weibo posts (Peng and Dredze, 2015).  
  Public release: https://github.com/hltcoe/golden-horse; mirrored at OpenDataLab: https://opendatalab.com/OpenDataLab/Weibo_NER.

- **OntoNotes 4.0 / 5.0**: Large multi-genre corpora with entity annotations (Weischedel et al., 2011; Pradhan et al., 2013).  
  Distributed by the Linguistic Data Consortium:  
  - OntoNotes 4.0: LDC2011T03 (https://catalog.ldc.upenn.edu/LDC2011T03)  
  - OntoNotes 5.0: LDC2013T19 (https://catalog.ldc.upenn.edu/LDC2013T19, DOI: https://doi.org/10.35111/xmhb-2b84).

- **GENIA (English biomedical)**: Corpus of MEDLINE abstracts annotated with biological entities (Kim et al., 2003).  
  Project website: http://www.geniaproject.org/.

- **CADEC (English medical forum)**: CSIRO Adverse Drug Event Corpus (Karimi et al., 2015).  
  Data portal: https://researchdata.edu.au/cadec/3378330.


Please refer to the original sources and licenses of each dataset before use.

## Code Information
The main components are:
- `dataloader.py`: Data loading, specific dataset preprocessing, and batch processing tools.  
- `model.py`: Implements the model architecture.  
- `main.py`: Model training, evaluation, testing, and main function.  
- `configs/`: Configuration files (e.g., hyperparameters, dataset paths, model settings).  
- `utils.py/`: Evaluation scripts for calculating accuracy, precision, recall, and F1 score.  

## Usage Instructions
1. Clone this repository.
2. Download the third-party datasets from their original sources (see “Dataset Information” section).
3. Place the datasets under the `data/data` directory with the following structure:
   - `data/data/conll2003/`
   - `data/data/msra/`
4. Download pre-trained BERT model

This project requires a local copy of the pre-trained BERT model used as the encoder (e.g., `bert-base-uncased` or `bert-base-chinese`)
If your environment does **not** have internet access (or you want to fix the exact version used in our experiments), please download the model manually and place it under the `data/` directory:

Go to the Hugging Face model page, for example:  
   - `bert-base-uncased`: https://huggingface.co/google-bert/bert-base-uncased  
   - `bert-base-chinese`: https://huggingface.co/google-bert/bert-base-chinese
Download all model files (configuration, vocabulary, and weights) and save them into a folder:
data/<bert-model-name>/
5. Install the required dependencies:
```
-numpy==1.21.4
-torch==1.10.0
-gensim==4.1.2
-transformers==4.13.0
-pandas==1.3.4
-scikit-learn==1.0.1
-prettytable==2.4.0
-torch==1.10.0+cu113 torchvision==0.11.1+cu113
```

6. Train the model
-python main.py --config config/genia.json
