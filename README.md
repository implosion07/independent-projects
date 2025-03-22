# Independent Machine Learning Projects 

This repository contains a collection of three independent machine learning projects developed by me. Each project demonstrates different ML techniques and applications.

## Projects Overview

### 1. Emoji Localization and Identification 
A Convolutional Neural Network (CNN) approach to both classify and locate emojis within images.

### 2. Twitter Sentiment Analysis 
A Natural Language Processing (NLP) project using Naive Bayes classification to determine sentiment in tweets.

### 3. Book Recommender System 
A recommendation system using collaborative filtering techniques to suggest books based on user preferences.

## Detailed Project Descriptions

## Emoji Localization and Identification

### Overview
This project combines image classification with object localization to identify emoji types and their positions within images. The system works under the assumption that each image contains exactly one emoji.

### Technical Approach
- **Model Architecture**: Custom CNN with both classification and regression outputs
- **Classification Task**: Identifying which emoji appears in the image
- **Localization Task**: Predicting bounding box coordinates (x, y, width, height)
- **Implementation**: Implemented in Python using TensorFlow/Keras

### Key Features
- Single-stage detection and classification pipeline
- Simplified approach to object detection without requiring complex frameworks like YOLO or R-CNN
- Demonstrated balance between computational efficiency and accuracy

### Usage
Refer to the `emoji_localization_and_classification.ipynb` notebook for implementation details, model training, and evaluation.

---

## Twitter Sentiment Analysis

### Overview
A text classification project that automatically categorizes tweets based on their sentiment (positive, negative, or neutral). This solution helps businesses understand customer opinions expressed on social media without manual review.

### Technical Approach
- **Algorithm**: Naive Bayes classifier
- **Text Processing**: Tokenization, stop word removal, stemming/lemmatization
- **Feature Extraction**: Bag-of-words and TF-IDF representations
- **Implementation**: Python with scikit-learn and NLTK

### Business Applications
- Brand monitoring and reputation management
- Customer feedback analysis
- Market research and competitive analysis
- Real-time sentiment tracking during events or product launches

### Usage
See the `twitter_sentiment_analysis.ipynb` notebook for the complete implementation, including data preprocessing, model training, and evaluation metrics.

---


## Book Recommender System

### Overview
A recommendation engine that suggests books to users based on collaborative filtering techniques. The system analyzes patterns in user preferences and book ratings to make personalized recommendations.

### Technical Approach
- **Algorithm**: Collaborative filtering
- **Similarity Metrics**: Cosine similarity between user vectors
- **Implementation**: Python with pandas, numpy, and scikit-learn

### Key Features
- User-based collaborative filtering to find similar users
- Item-based collaborative filtering to find similar books
- Handling of the cold-start problem for new users
- Evaluation using standard recommendation system metrics

### Business Value
This system provides significant utility to e-book platforms and online libraries by:
- Increasing user engagement through personalized recommendations
- Improving discovery of less popular but relevant titles
- Enhancing user experience and satisfaction

### Usage
Explore the `book_recommender_system.ipynb` notebook for implementation details, including data preparation, model development, and recommendation generation examples.

---

## Repository Structure

```
machine-learning-projects/
├── emoji_localization_and_classification.ipynb
├── twitter_sentiment_analysis.ipynb
├── book_recommender_system.ipynb
└── README.md
```

## Requirements

- Python 
- TensorFlow 
- scikit-learn
- NLTK
- pandas
- numpy
- matplotlib
- Jupyter Notebook / Google Colab

Detailed requirements for each project are specified in their respective notebooks.

## Future Work

Potential enhancements for each project:
- **Emoji Project**: Extend to multi-emoji detection, add support for custom emojis
- **Sentiment Analysis**: Implement more advanced models (BERT, Transformers), add multi-language support
- **Book Recommender**: Incorporate content-based features, develop hybrid recommendation approach

