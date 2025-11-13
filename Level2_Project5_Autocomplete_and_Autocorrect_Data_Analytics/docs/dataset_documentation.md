# Dataset Documentation

## 1. Credit Card Fraud Detection Dataset

### Source
- **Provider**: Worldline and Machine Learning Group of ULB
- **Collection Period**: September 2013
- **Geography**: European cardholders

### Dataset Characteristics
- **Total Transactions**: 284,807
- **Fraudulent Transactions**: 492 (0.172%)
- **Number of Features**: 30
- **Time Period**: 2 days

### Feature Description

#### Original Features (Non-PCA)
- **Time**: Seconds elapsed between transaction and first transaction
- **Amount**: Transaction amount
- **Class**: Target variable (1: fraud, 0: genuine)

#### PCA Transformed Features
- **V1-V28**: Principal components obtained from PCA transformation
- **Note**: Original features not provided due to confidentiality

### Data Quality
- No missing values
- Features already scaled and normalized
- High class imbalance

### Recommended Usage
- Use Area Under Precision-Recall Curve (AUPRC) for evaluation
- Confusion matrix accuracy not meaningful due to imbalance
- Consider cost-sensitive learning approaches

## 2. NLP Text Datasets

### Required Data Characteristics
- **Diversity**: Multiple domains and writing styles
- **Size**: Large corpus for robust language modeling
- **Quality**: Clean, well-formatted text

### Potential Data Sources

#### Public Corpora
- **Wikipedia**: General knowledge, diverse topics
- **News Articles**: Current events, formal writing
- **Book Corpora**: Literary text, varied styles
- **Academic Papers**: Technical language

#### Domain-Specific Sources
- **Social Media**: Informal language, abbreviations
- **Technical Documentation**: Specialized terminology
- **Customer Reviews**: Opinionated text, common phrases

### Data Collection Strategy
1. **Initial Collection**: Gather diverse text sources
2. **Preprocessing**: Clean and normalize text
3. **Quality Assessment**: Remove low-quality content
4. **Augmentation**: Add spelling variations for autocorrect training

### Sample Data Structure
```python
# Expected text format
texts = [
    "This is a sample sentence for training.",
    "Another example with different vocabulary.",
    "Texts should cover various domains and styles."
]
