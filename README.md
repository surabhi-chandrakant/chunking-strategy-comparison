# 📊 Chunking Strategy Comparison Tool

A production-ready interactive tool for comparing chunking strategies and evaluating retrieval quality. Built with Streamlit and optional Groq API integration.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge\&logo=Streamlit\&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-000000?style=for-the-badge\&logo=groq\&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 What is This Tool?

This tool helps you understand and compare different **text chunking strategies** — a critical component in building Retrieval-Augmented Generation (RAG) systems, document search, and AI applications.

### Supported Chunking Strategies

* **Fixed Size Chunking** – Simple word-count based chunking with configurable overlap
* **Sentence-Based Chunking** – Preserves sentence boundaries for better context retention
* **Semantic Chunking** – Groups semantically related sentences using TF-IDF similarity

### Retrieval Quality Evaluation

Evaluate chunking strategies using:

* Precision
* Recall
* Relevance Score
* Diversity Score

### Interactive Features

* Add custom documents
* Load sample datasets
* Compare chunking strategies
* Analyze retrieval performance
* Visual analytics dashboard

---

## 🚀 Why Use This Tool?

### For RAG Developers

* Optimize chunking strategies before deployment
* Compare retrieval performance
* Improve context quality for LLMs

### For NLP Practitioners

* Experiment with chunking parameters
* Understand semantic chunking behavior
* Visualize document segmentation

### For AI Engineers

* Production-ready architecture
* Groq integration support
* Easy extensibility for new chunking algorithms

---

## ✨ Key Features

### 1️⃣ Multiple Chunking Strategies

| Strategy       | Description                             |
| -------------- | --------------------------------------- |
| Fixed Size     | Chunks based on word count with overlap |
| Sentence Based | Maintains sentence boundaries           |
| Semantic       | Groups similar sentences together       |

### 2️⃣ Retrieval Evaluation Metrics

* Precision
* Recall
* Relevance
* Diversity

### 3️⃣ Visual Analytics

* Strategy comparison charts
* Box plots
* Historical query analysis
* Interactive Plotly visualizations

### 4️⃣ Groq API Integration (Optional)

* Enhanced embeddings
* Better semantic retrieval
* Free-tier compatibility
* Automatic TF-IDF fallback

### 5️⃣ Document Management

* Add custom documents
* Load sample documents
* Clear stored documents
* Multi-document support

---

# 📦 Installation

## Clone Repository

```bash
git clone https://github.com/surabhi-chandrakant/chunking-strategy-comparison.git

cd chunking-strategy-comparison
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

# 📋 Requirements

```txt
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
plotly>=5.17.0
groq>=0.4.0
```

---

# 🎮 How to Use

## Step 1: Add Documents

* Open the Documents tab
* Paste your content
* Click Add Document

Or:

* Load Sample Documents from sidebar

---

## Step 2: Configure Chunking

Choose:

* Chunk Size (50–500 words)
* Overlap (0–50 words)
* Strategy:

  * Fixed Size
  * Sentence Based
  * Semantic

For Semantic Chunking:

* Adjust similarity threshold

---

## Step 3: Process Documents

Navigate to **Chunking Tab**

Click:

```text
Process Documents
```

View:

* Generated chunks
* Chunk statistics
* Distribution analysis

---

## Step 4: Evaluate Retrieval

Navigate to **Evaluation Tab**

Enter a query such as:

```text
What is machine learning and how does it work?
```

Click:

```text
Evaluate
```

View:

* Precision
* Recall
* Relevance
* Diversity
* Top retrieved chunks

---

## Step 5: Analyze Results

Navigate to **Analytics Tab**

Compare:

* Chunking strategies
* Retrieval metrics
* Performance trends

---

# 🔥 Optional: Enable Groq Embeddings

1. Create a free account at:

   https://console.groq.com

2. Generate an API Key

3. Paste API Key in sidebar

4. Enable:

```text
Use Groq Embeddings
```

5. Process documents again

---

# 🏗️ Project Architecture

```text
app.py
│
├── Config
│
├── GroqService
│
├── DocumentProcessor
│
├── ChunkingManager
│
├── Chunking Strategies
│   ├── FixedSizeChunking
│   ├── SentenceBasedChunking
│   └── SemanticChunking
│
├── RetrievalEvaluator
│
└── Streamlit UI
    ├── Sidebar
    ├── Documents Tab
    ├── Chunking Tab
    ├── Evaluation Tab
    └── Analytics Tab
```

---

# 📊 Example Queries

Try:

### Machine Learning

```text
What is machine learning and how does it work?
```

### AI vs Deep Learning

```text
Explain the relationship between AI and deep learning.
```

### Computer Vision

```text
How does computer vision process images?
```

### NLP

```text
What are the applications of NLP?
```

### Comparison

```text
Compare different types of machine learning.
```

---

# 🎨 UI Features

✅ Modern Dark Theme

✅ Responsive Layout

✅ Interactive Plotly Charts

✅ Real-Time Analytics

✅ Professional Design

---

# 🔧 Customization

## Add New Chunking Strategy

```python
class MyCustomChunking:

    strategy_name = "My Strategy"

    def __init__(
        self,
        chunk_size=200,
        overlap=20,
        **kwargs
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text):
        # Custom chunking logic
        return chunks
```

---

## Modify Configuration

```python
@dataclass
class Config:

    CHUNK_SIZES = [50, 100, 200, 300, 500]

    MAX_DOCUMENTS = 20
```

---

# 📈 Use Cases

## RAG Development

Optimize chunking before production deployment.

## NLP Research

Compare retrieval quality across chunking methods.

## Enterprise Search

Improve document search relevance.

## Chatbots

Enhance context retrieval for conversational AI.

## Education

Learn chunking concepts through experimentation.

---

# 🤝 Contributing

Contributions are welcome!

### 1. Fork Repository

```bash
git fork
```

### 2. Create Feature Branch

```bash
git checkout -b feature/amazing-feature
```

### 3. Commit Changes

```bash
git commit -m "Add amazing feature"
```

### 4. Push

```bash
git push origin feature/amazing-feature
```

### 5. Create Pull Request

---

# 📝 License

MIT License

Feel free to use, modify, and distribute.

---

# 🔗 Repository

### GitHub Repository

https://github.com/surabhi-chandrakant/chunking-strategy-comparison

### Clone

```bash
git clone https://github.com/surabhi-chandrakant/chunking-strategy-comparison.git
```

---

# 🏷️ GitHub Repository Settings

## Repository Name

```text
chunking-strategy-comparison
```

## GitHub Description

```text
📊 Compare text chunking strategies (Fixed Size, Sentence Based, Semantic) with retrieval quality metrics. Interactive Streamlit app with optional Groq API integration for enhanced embeddings. Perfect for RAG optimization and NLP experimentation.
```

## Topics

```text
chunking
rag
retrieval-augmented-generation
nlp
text-processing
streamlit
groq
ai
machine-learning
document-search
semantic-chunking
retrieval
embeddings
llm
vector-search
```

---

# ⚡ Quick Start

```bash
# Clone Repository
git clone https://github.com/surabhi-chandrakant/chunking-strategy-comparison.git

# Enter Project Directory
cd chunking-strategy-comparison

# Install Dependencies
pip install -r requirements.txt

# Launch App
streamlit run app.py
```

Visit:

```text
http://localhost:8501
```

---

# ⭐ Support

If you find this project useful:

⭐ Star the repository

🐛 Open an issue for bugs

💡 Suggest improvements

🤝 Contribute new features

---

## Made with ❤️ for the AI, RAG & NLP Community
