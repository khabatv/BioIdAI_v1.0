# ECO-ID: Error-Correcting Omics IDentifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.19205996-blue)](https://doi.org/10.5281/zenodo.19205996)

**ECO-ID** (Error-Correcting Omics IDentifier) is a versatile, interactive browser-based application designed for high-accuracy multi-omics ID conversion and annotation. By leveraging Large Language Models (LLMs), it specifically addresses the "human-error" barrier in bioinformatics—resolving misspellings, synonyms, and inconsistent naming conventions that traditional machine-blind algorithms fail to process.

##  Key Features

* **AI-Driven Error Correction:** Resolves human-written lists with up to 96% accuracy by interpreting ambiguous entries across chemical and biological data.
* **Multi-Omics Support:** Seamlessly maps and cross-validates IDs across chemicals, metabolites, genes, proteins, and coding sequences (CDS).
* **Massive Database Integration:** Connects to major molecular databases (NCBI, UniProt, Ensembl, KEGG) and chemical resources (PubChem, ChEMBL, ChemSpider, HMDB, METLIN).
* **Green Bioinformatics:** Mitigates the carbon footprint of omics research by reducing failed queries and redundant computations, saving significant energy and computational resources.
* **Privacy-First & Client-Side:** Developed as a React-based single-page application (SPA) where data management and visualizations occur entirely within the browser.

##  Performance

In benchmarks using error-prone lists, ECO-ID significantly outperforms traditional dedicated platforms and conversion tools:

| Category | ECO-ID Success Rate | Traditional Tools (DAVID, g:Profiler, etc.) |
| :--- | :--- | :--- |
| **Chemical Entities** | **96%** | **≤ 7%** |
| **Genes & Proteins** | **92%** | **≤ 3%** |

## 📂 Supplementary Materials & Reproducibility

To ensure the transparency and reproducibility of our benchmarks, all raw data and evaluation scripts are available:

* **Official Archive (Zenodo):** [https://doi.org/10.5281/zenodo.19205996](https://doi.org/10.5281/zenodo.19205996)
* **Supplementary Repository:** [https://github.com/khabatv/ECO-ID-Supplementary](https://github.com/khabatv/ECO-ID-Supplementary)

**What is included:**
* **Test Lists:** 100 chemical and 100 biological entities containing intentional human-generated errors used for validation.
* **Benchmarking Scripts:** Python logic used for batch retrieval from databases like PubChem, ChEMBL, NCBI, and UniProt.
* **Raw Results:** Detailed CSV tables showing the performance comparison of ECO-ID versus traditional tools.

## 🚀 Quick Start

### Prerequisites
* A modern web browser (Chrome, Firefox, Safari, or Edge).
* An LLM API Key (Google Gemini, OpenAI, or Anthropic).

##  Citation
If you use ECO-ID in your research, please cite our manuscript:

Vahabi, K., Barman, M., van Dam, N. M., Witzel, K., Alves, M. F., & Bueno, P. C. P. (2026). ECO-ID: A Versatile Tool for Multi-Omics ID Conversion and Annotation with AI-Driven Error Correction. DOI: 10.5281/zenodo.19205996. Also Submitted to Frontiers in Bioinformatics.


## License
This project is licensed under the MIT License - see the LICENSE file for details

### Installation
```
git clone [https://github.com/khabatv/ECO-ID.git](https://github.com/khabatv/ECO-ID.git)
cd ECO-ID
npm install
```

### Running Locally
Create a .env file in the root directory and add the API key for your preferred provider:
```
# Provide at least one of the following keys:
GEMINI_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GROQ_API_KEY=your_groq_key_here
COHERE_API_KEY=your_cohere_key_here
MISTRAL_API_KEY=your_mistral_key_here
PERPLEXITY_API_KEY=your_perplexity_key_here
TOGETHER_API_KEY=your_together_key_here
```

### Launch the application:
```
npm run dev
```
