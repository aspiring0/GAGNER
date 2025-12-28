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
  Repo: https://github.com/aspiring0/GAGNER/tree/main/data/data/resume-zh

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
  Website: https://github.com/aspiring0/GAGNER/tree/main/data/data/genia

- **CADEC (English medical forum)**  
  CSIRO Adverse Drug Event Corpus (Karimi et al., 2015).  
  Data portal: https://researchdata.edu.au/cadec/3378330


## 3. Data Preprocessing
This step does not require executing arbitrary code.

The exact subdirectory names and file formats should match what is expected in the corresponding configuration files under `config/`.
This repository follows the W2NER-style grid formulation. Data processing is implemented in `dataloader.py` and is performed on-the-fly when building PyTorch `Dataset`s. The pipeline assumes each dataset split is stored as JSON:

- `./data/data/<dataset>/train.json`
- `./data/data/<dataset>/dev.json`
- `./data/data/<dataset>/test.json`

Each JSON instance should contain:
- `sentence`: a list of tokens/words (length = N)
- `ner`: a list of entities, where each entity includes:
- `index`: a list of token indices (supports multi-token and discontinuous spans)
- `type`: the entity label string

### 3.1: Build label vocabulary

`fill_vocab(vocab, dataset)` collects all entity types from the dataset and builds a label-to-id mapping. Two special labels are reserved:
- PAD label id = 0
- SUC label id = 1

### 3.2: BERT tokenization and piece-to-word alignment

For each input sentence:
1. Each word is tokenized into WordPiece tokens using a Hugging Face tokenizer (`AutoTokenizer`).
2. All pieces are concatenated and converted into token ids.
3. Special tokens `[CLS]` and `[SEP]` are added.
4. A boolean matrix `pieces2word` of shape `(N, P)` is built to map each original word (N) to its corresponding piece positions (P), offset by 1 because of `[CLS]`.

### 3.3: Construct grid labels and masks

For each sentence of length N:
- `grid_labels`: an integer matrix of shape `(N, N)` initialized to 0.
- `grid_mask2d`: a boolean matrix of shape `(N, N)` initialized to True (all valid positions).

For each entity with token index list `idx = [i1, i2, ..., ik]`:
- For each adjacent pair, assign a link label:
  - `grid_labels[idx[t], idx[t+1]] = 1` (SUC)
- Assign the entity type label to the “tail-to-head” position:
  - `grid_labels[idx[-1], idx[0]] = label_id(entity_type)`

Additionally, `entity_text` stores a set representation of gold entities for evaluation.

### 3.4: Relative distance features

`dist_inputs` is an integer matrix of shape `(N, N)` that encodes relative distances between token positions. Distances are bucketed using the `dis2idx` mapping and shifted to distinguish direction, and zero-distance is mapped to a dedicated value.

### 3.5: Build PyTorch datasets and batching

`load_data_bert(config)` loads JSON splits, constructs the `Vocabulary`, and creates:
- `RelationDataset` for train/dev/test, each item returning:
  - `bert_inputs` (piece ids)
  - `grid_labels`
  - `grid_mask2d`
  - `pieces2word`
  - `dist_inputs`
  - `sent_length`
  - `entity_text`

The `collate_fn` pads variable-length sequences and matrices to batch-level maximum sizes using `pad_sequence` and manual tensor filling.
## 4. Project Structure

Main components of this repository:

- `dataloader.py`: Data loading, dataset-specific preprocessing, and batch construction utilities.  
- `model.py`: Model architecture implementation.  
- `main.py`: Entry script for training, evaluation, and testing.  
- `config/`: Configuration files (e.g., hyperparameters, dataset paths, model settings).  
- `utils.py`: Utility functions and evaluation metrics (accuracy, precision, recall, F1, etc.).

## 5. Setup

### 5.1 Environment

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


### 5.2 Pretrained BERT

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
## 6. Training

After preparing datasets and pretrained BERT, train the model with:

`python main.py --config config/genia.json`
You can switch to other datasets by changing the configuration file, for example:
```
python main.py --config config/msra.json
python main.py --config config/conll03.json
python main.py --config config/weibo.json
```

