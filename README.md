# BERT Strategic Item Classifier

A machine learning solution developed during the **BEST Hackathon 2025**. This project showcases the integration of NLP (Natural Language Processing) into a strategic game logic, using a fine-tuned BERT model to map semantic concepts to optimized game IDs.

---

## 🚀 Features
- **Semantic Understanding:** Uses `AutoModelForSequenceClassification` (BERT) to understand the context of 77+ different items/concepts.
- **Strategic Decision Logic:** Instead of just taking the top AI prediction, the engine analyzes the **Top 3 results** and picks the one with the highest strategic efficiency (lowest cost score).
- **CPU Optimized:** Configured to run efficiently on standard hardware without requiring a dedicated GPU.
- **Diverse Vocabulary:** Handles everything from natural elements (Lava, Tsunami) to complex tech (Anti-virus Nanocloud, Singularity Stabilizer).

---

## 🛠️ Technical Stack
- **Language:** Python
- **AI Frameworks:** PyTorch, HuggingFace Transformers
- **Model Architecture:** BERT (Bidirectional Encoder Representations from Transformers)
- **Concepts covered:** 77 unique classes with custom score mapping.

---

## ⚙️ How it Works
The core of the project is the `predict` function, which follows these steps:
1. **Tokenization:** Converts raw text input into a format BERT can understand.
2. **Probability Mapping:** Calculates the likelihood for each of the 77 categories.
3. **Top-K Filtering:** Extracts the three most likely matches.
4. **Score Optimization:** 
   - Among the top 3 AI predictions, we choose the one that provides the best value based on the internal game dictionary.
