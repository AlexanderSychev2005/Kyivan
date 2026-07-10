# Kyivan-Aeneas: Ancient Slavic Text Restoration Model

**Kyivan-Aeneas** is a specialized Transformer-based model designed for the restoration and analysis of ancient Slavic texts (IX–XIX centuries). By leveraging multi-task learning and dynamic physical degradation simulation, the model aims to accurately restore missing characters in historical documents (such as birchbark letters and epigraphica), predict the historical era (dates), and identify the dialectical region of a text.

---

## 📚 1. Data Assets

The model is trained on a highly diverse, deduplicated collection of Old East Slavic and Church Slavonic corpora. All final preprocessed data is stored in the `prepared_datasets/` directory.

### Core Datasets:
- **UD_Old_East_Slavic-RNC**: 322 cleaned documents from the Russian National Corpus. Includes private correspondence (gramotki), administrative acts, chronicles (letopisi), and spells.
- **NKRYA (Russian National Corpus)**: Additional documents from the Old Russian and Middle Russian subcorpus, containing a wealth of historical acts, chronicles, everyday records, and private correspondence (11th–17th centuries). While some of these documents overlap with the RNC dataset, they have been strictly deduplicated.
- **UD_Old_East_Slavic-Ruthenian**: 420 documents, primarily legal and historical records from the Ruthenian (South-Western) dialect area (Polotsk charters, Lithuanian Metrica).
- **Sofia**: 154 documents from the Histdict database. These are primarily Old Bulgarian and Church Slavonic medieval manuscripts (e.g., gospels, chronographs, apocrypha, and hagiographic literature), providing a crucial bridge to South Slavic orthographic traditions.
- **TOROT (Tromsø Old Russian and OCS Treebank)**: 39 major texts, including primary chronicles and historical literature.
- **Pushkin Texts (Pushkin House / IRLI)**: 57 ancient manuscript texts from the archives of the Institute of Russian Literature (Pushkin House). This collection contains medieval Old Russian literature, chronicles, and ecclesiastical manuscripts notable for their authentic archaic orthography.
- **Epigraphica**: 620 historical inscriptions and graffiti from ancient church walls (e.g., St. Sophia in Kyiv and Novgorod). These short, fragmentary texts include precise dating and are mapped directly to cleaned snippets. Crucially, they contain real archaeological lacunae, making them perfect for both training and generating Test B.
- **Birchbark Letters**: Hundreds of everyday medieval letters etched on birch bark from Novgorod, Staraya Russa, Smolensk, and other ancient cities. They provide vital data on the colloquial Old Novgorodian dialect, business correspondence, and everyday spoken language of the era.

### Final Data Splits (`prepared_datasets/hf_dataset`):
The datasets are chunked (1024 chars, stride 512) and strictly separated to prevent data leakage:
- **Train**: 22,752 chunks.
- **Test A**: 2,528 chunks.
- **Test B**: 1,188 textual segments containing **real historical lacunae** (brackets from archaeological transcriptions).

---

## 🧠 2. Model Architecture

The core architecture (`src/model/model.py`) is a customized Transformer Encoder with **Multi-Task Learning** capabilities:
1. **MLM Head (Restoration)**: Predicts the exact missing character (`[-]`) based on context.
2. **Unk Head**: Predicts the true length of unknown continuous lacunae (`[#]`), allowing the model to deduce how many characters were torn off from the edge of a manuscript.
3. **Date Embeddings**: 20 temporal bins embedded into the sequence. These 20 bins correspond exactly to 10 centuries of history (from the 9th century to the 19th century, i.e., 800 AD – 1800 AD, with 50-year intervals).
4. **Region Embeddings**: 4 dialectical macro-regions (`NW` - North-Western/Novgorod, `SW` - South-Western/Ruthenian, `OES` - Old East Slavic, `CS` - Church Slavonic).

---

## 🌪 3. KyivanPhysicalCollator

The secret weapon of the training pipeline is the `KyivanPhysicalCollator` (`src/model/collator.py`). Instead of standard random masking, this collator dynamically simulates physical time degradation on the fly:
- **Character Fading**: Randomly masks individual characters (15% probability).
- **Edge Tears**: Simulates the physical tearing of parchment by randomly replacing the beginning or the end of the text with a special `[#]` token (span lacuna).
- **Continuous Lacunae**: Randomly removes contiguous chunks of text from the middle of the document.

The model is forced to learn robust linguistic patterns rather than memorizing phrases.

---

## 🚀 4. How to Train (Instructions for Colleagues)

### Prerequisites
1. Install Python 3.10+
2. Install dependencies (we use `uv` for lightning-fast environment management):
   ```bash
   uv sync
   ```
3. Activate the virtual environment (if not automatically activated by your IDE).

### Running the Training
Everything is handled by `src/model/train.py`. It uses a custom `KyivanAeneasTrainer` that calculates a weighted multi-task loss and evaluates on historical brackets (`TestBEvalCallback`).

To start training with default parameters:
```bash
python src/model/train.py --dataset_dir prepared_datasets/hf_dataset --char_vocab_path prepared_datasets/tokenizer/char_vocab.json
```

**Training Arguments:**
You can override standard parameters in the script (batch size, learning rate, epochs). Currently defaults to:
- `epochs`: 10
- `learning_rate`: 1e-4
- `batch_size`: 16

### Evaluating Results
During evaluation, the `TestBEvalCallback` bypasses the collator and tests the model strictly on real historical lacunae (Test B). It automatically generates highly readable CSV reports in the project root:
- `pred_report_test_b_step_XXX.csv`
Open this file in Excel or pandas to see:
- `Context`: The surrounding text.
- `True Char`: The actual character that was lost.
- `Top 1-3 Preds & Probs`: What the model guessed and with what confidence.