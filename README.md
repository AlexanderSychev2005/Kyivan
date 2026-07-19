# Kyivan: Ancient Slavic Text Restoration Model

**Kyivan** is a specialized Transformer-based model designed for the restoration and analysis of ancient Slavic texts (IX–XIX centuries). By leveraging multi-task learning and dynamic physical degradation simulation, the model aims to accurately restore missing characters in historical documents (such as birchbark letters and epigraphica), predict the historical era (dates), and identify the dialectical region of a text.

---

## 1. Data Assets

The model is trained on a highly diverse collection of Old East Slavic, Ruthenian, Novgorodian, and Church Slavonic corpora.

### Core Datasets:
- **UD_Old_East_Slavic-RNC**: 322 cleaned documents from the Russian National Corpus. Includes private correspondence (gramotki), administrative acts, chronicles (letopisi), and spells.
- **NKRYA (Russian National Corpus)**: Additional documents from the Old Russian and Middle Russian subcorpus, containing a wealth of historical acts, chronicles, everyday records, and private correspondence (11th–17th centuries). While some of these documents overlap with the RNC dataset, they have been strictly deduplicated.
- **UD_Old_East_Slavic-Ruthenian**: 420 documents, primarily legal and historical records from the Ruthenian (South-Western) dialect area (Polotsk charters, Lithuanian Metrica).
- **Sofia**: 154 documents from the Histdict database. These are primarily Old Bulgarian and Church Slavonic medieval manuscripts (e.g., gospels, chronographs, apocrypha, and hagiographic literature), providing a crucial bridge to South Slavic orthographic traditions.
- **TOROT (Tromsø Old Russian and OCS Treebank)**: 39 major texts, including primary chronicles and historical literature.
- **Pushkin Texts (Pushkin House / IRLI)**: 57 ancient manuscript texts from the archives of the Institute of Russian Literature (Pushkin House). This collection contains medieval Old Russian literature, chronicles, and ecclesiastical manuscripts notable for their authentic archaic orthography.
- **Epigraphica**: 986 historical inscriptions and graffiti from ancient church walls (e.g., St. Sophia in Kyiv and Novgorod). These short, fragmentary texts include precise dating and are mapped directly to cleaned snippets. Crucially, they contain real archaeological lacunae, making them perfect for both training and generating Test B.
- **Birchbark Letters**: 1,241 everyday medieval letters etched on birch bark from Novgorod, Staraya Russa, Smolensk, and other ancient cities. They provide vital data on the colloquial Old Novgorodian dialect, business correspondence, and everyday spoken language of the era.
- **Ostrog Bible (1581)**: The first complete printed Bible in Church Slavonic, all 76 books, sourced from the historic Ostroh press. Unlike the fragmentary, lacuna-ridden sources above, this is complete, clean, and precisely dated text — each book kept whole as a single document — giving the model a large, reliably-dated anchor for the Church Slavonic dialect class.

---

## 2. Model Architecture

The core architecture (`src/model/model.py`) is a customized Transformer Encoder with **Multi-Task Learning** capabilities:
1. **MLM Head (Restoration)**: Predicts the exact missing character (`[-]`) based on context.
2. **Unk Head**: Predicts the true length of unknown continuous lacunae (`[#]`), allowing the model to deduce how many characters were torn off from the edge of a manuscript.
3. **Date Embeddings**: 20 temporal bins embedded into the sequence. These 20 bins correspond exactly to 10 centuries of history (from the 9th century to the 19th century, i.e., 800 AD – 1800 AD, with 50-year intervals).
4. **Region Embeddings**: 4 dialectical macro-regions (`NW` - North-Western/Novgorod, `SW` - South-Western/Ruthenian, `OES` - Old East Slavic, `CS` - Church Slavonic).
