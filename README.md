# Decoder model for Slavic text restoration

## Data

### Pushkin texts
57 texts from http://lib.pushkinskijdom.ru/.

### UD Old East Slavic-Ruthenian
Parsed from Universal Dependencies `.conllu` files and converted to JSON, containing 420 cleaned documents. Sub-sources include:
- **Polotsk charters (`polotsk`)**: 331 documents
- **Ratushna kniga (`RatushnaKniga_1986`)**: 76 documents
- **Lithuanian Metrica (`lit`)**: 3 documents
- **Other legal and historical documents (`uk`, `otpys`, `starbel.narod.ru`, `litopys.org.ua`, `A43`, `falsifikat`)**: 10 documents

### UD Old East Slavic-RNC
Parsed from Universal Dependencies `.conllu` files (Russian National Corpus dataset) and converted to JSON, containing 322 cleaned documents. It comprises a highly diverse set of sub-sources from the Middle Russian and Old East Slavic periods, including:
- **Private correspondence and administrative acts**: Bezobrazov papers (`bezobrazov`), Peter the Great's papers (`petr`), letters (`gramotki`), Morozov household acts (`morozov`), Kungur acts (`kungur`), Acts of the Moscow State (`amg`), and Feudal Landownership Acts (`afz1`, `afz2`, `afz3`).
- **Charters and treaties**: Novgorod and Pskov Charters (`gvnp`, `pskov`), spiritual and contractual charters (`duhdog`).
- **Literary, historical, and miscellaneous texts**: Library of Literature of Ancient Rus (`bldr`), chronicles (`letopisi`, `psrl`), spells and incantations (`zagovor`), early Russian dramaturgy (`drama`), and various other records (`varia`, `rib`).

### SCAT
26 raw texts containing Old Russian hagiographies (Жития), lives of saints, and other spiritual literature.

### Sofia
154 documents from https://histdict.uni-sofia.bg (JSON format).

### Torot
39 texts from the Tromsø Old Russian and OCS Treebank (TOROT), including prominent historical works like Afanasy Nikitin's "Journey Beyond Three Seas".

### UAC (Ukrainian Archaeological Collection)
6 texts containing Old Ruthenian legal documents, court records, and royal decrees (e.g., decrees of Sigismund Augustus).

### Epigraphica
Contains 620 historical inscriptions, graffiti, and records from ancient church walls (e.g. St. Sophia in Kyiv and Novgorod). These short, fragmentary texts include precise dating (years) extracted from a metadata CSV, mapped directly to cleaned text snippets with preserved lacunae formatting ([UNK]).

### Birchbark Letters
Stored in `birchbark_classes.jsonl`, containing textual data and classifications for ancient birchbark letters.