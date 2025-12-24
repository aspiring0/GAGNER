# Global-Attn-GateNER: Unified Entity Recognition Based on Global Attention and Dynamic Gateway

## 1.Introduction
Unified modeling poses a significant challenge for Named Entity Recognition (NER), for which efficiently capturing semantic and feature fusion remains critical.  
The W2NER framework, based on word–word relationship classification, offers a unified approach to NER through two-dimensional word grid modeling but suffers from insufficient context awareness and rigid feature interactions.  
This project proposes:
- A global self-attention mechanism to strengthen boundary semantic representations.
- A hierarchical perturbation strategy to mitigate over-regularization in the dual-affine module.
- A dynamically gated fusion network to adaptively aggregate structured features and contextual representations at the word-pair granularity.

Extensive experiments on eight benchmark datasets (four Chinese and four English) show state-of-the-art performance.

## 2.Dataset Information
Please refer to the original sources and licenses of each dataset before use.

- **CoNLL-2003 (English)**  
  Newswire NER dataset with four entity types (PER, ORG, LOC, MISC).  
  Website: https://www.clips.uantwerpen.be/conll2003/ner/

- **MSRA (Chinese news)**  
  Chinese NER dataset from SIGHAN 2006 shared task (Levow, 2006).  
  Available from the original organizers and licensed mirrors that distribute the SIGHAN 2006 datasets.

- **Resume (Chinese)**  
  Chinese resume NER dataset (Zhang and Yang, 2018).  
  Repo: https://github.com/jiesutd/ChineseNER

- **Weibo (Chinese social media)**  
  NER dataset from Sina Weibo posts (Peng and Dredze, 2015).  
  Repo: https://github.com/hltcoe/golden-horse  
  Mirror: https://opendatalab.com/OpenDataLab/Weibo_NER

- **OntoNotes 4.0 / 5.0**  
  Large multi-genre corpora with entity annotations (Weischedel et al., 2011; Pradhan et al., 2013).  
  Distributed by the Linguistic Data Consortium:  
  - OntoNotes 4.0: LDC2011T03 — https://catalog.ldc.upenn.edu/LDC2011T03  
  - OntoNotes 5.0: LDC2013T19 — https://catalog.ldc.upenn.edu/LDC2013T19 (DOI: https://doi.org/10.35111/xmhb-2b84)

- **GENIA (English biomedical)**  
  MEDLINE abstracts with biological entity annotations (Kim et al., 2003).  
  Website: http://www.geniaproject.org/

- **CADEC (English medical forum)**  
  CSIRO Adverse Drug Event Corpus (Karimi et al., 2015).  
  Data portal: https://researchdata.edu.au/cadec/3378330



## 3. Project Structure

Main components of this repository:

- `dataloader.py`: Data loading, dataset-specific preprocessing, and batch construction utilities.  
- `model.py`: Model architecture implementation.  
- `main.py`: Entry script for training, evaluation, and testing.  
- `config/`: Configuration files (e.g., hyperparameters, dataset paths, model settings).  
- `utils.py`: Utility functions and evaluation metrics (accuracy, precision, recall, F1, etc.).

## 4. Setup

### 4.1 Environment

It is recommended to use Python 3.8+ and a virtual environment:
-`python -m venv venv`

Linux / macOS:
-`source venv/bin/activate`

Windows:
-`venv\Scripts\activate`
Install required dependencies (recommended via `re.txt`):
```
-numpy==1.21.4
-torch==1.10.0
-gensim==4.1.2
-transformers==4.13.0
-pandas==1.3.4
-scikit-learn==1.0.1
-prettytable==2.4.0
-torch==1.10.0+cu113
-torchvision==0.11.1+cu113
```
Then:
-`pip install -r re.txt`

### 4.2 Dataset layout

Download the datasets listed in Section 2 and place them under the `data/data` directory, for example:
```
data/
data/
conll2003/
```
The exact subdirectory names and file formats should match what is expected in the corresponding configuration files under `config/`.

### 4.3 Pretrained BERT

This project uses a pretrained BERT encoder, such as `bert-base-uncased` or `bert-base-chinese`.

If your environment has internet access, Hugging Face `transformers` can download models automatically.  
If your environment does not have internet access, or you want to fix the exact model version, download the model manually and place it under the `data/` directory.

Example model pages:

- `bert-base-uncased`: https://huggingface.co/google-bert/bert-base-uncased  
- `bert-base-chinese`: https://huggingface.co/google-bert/bert-base-chinese  

Download all model files (configuration, vocabulary, weights) and save them into:data/<bert-model-name>/
```
e.g. data/bert-base-uncased/
data/bert-base-chinese/
```
## 5. Training

After preparing datasets and pretrained BERT, train the model with:

`python main.py --config config/genia.json`
You can switch to other datasets by changing the configuration file, for example:
```
python main.py --config config/msra.json
python main.py --config config/conll03.json
python main.py --config config/weibo.json
```

