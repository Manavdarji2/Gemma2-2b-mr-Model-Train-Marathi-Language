# Gemma2-2b-mr-Model-Train-Marathi-Language
This is the model train in the Marathi Language. Model i have use gemma2-2b-en model to train in Marathi Language 



# Gemma Language Tuning Project

### Team Information
<table border="1" font-size=8px>
  <tr>
    <th>Name</th>
    <th>Kaggle Username</th>
    <th>GitHub ID</th>
  </tr>
  <tr>
    <td>Manav Darji</td>
    <td>manavdarji18</td>
    <td>Manavdarji2</td>
  </tr>
  <tr>
    <td>Vidhaan Das</td>
    <td>vidhaandas</td>
    <td>None</td>
  </tr>
  <tr>
    <td>Renuka Mane</td>
    <td>renukamane141</td>
    <td>renuka141</td>
  </tr>
  <tr>
    <td>Aditi Sanap</td>
    <td>agape20</td>
    <td>aiddie1729</td>
  </tr>
</table>

### Work Distribution
```markdown
- **Programming**: Manav, Vidhaan
- **Data Mining and Processing**: Renuka, Aditi
- **Training and Testing**: All Team Members
```

### Project Timeline
```markdown
1. **Month 1**: Work with small datasets while initiating data mining and processing.
2. **Month 2**: Apply on real dataset, aiming for ~90% accuracy.
3. **Month 3**: Optimize model accuracy and enhance training/testing with additional data.
```
### Project Steps

1. Collect data.
2. Preprocess the data.
3. Split data into train and test sets.
4. Train the model.
5. Test the model.
6. Evaluate the model.
7. Repeat until desired accuracy is achieved.

### Project Overview

- **Goal**: Fine-tune the Gemma model for Marathi language, optimizing performance for NLP tasks like translation, dialogue generation, and text analysis.

### Required Libraries

- **Transformers by Hugging Face**: Leverage pre-trained models and tokenizers.
- **Keras/Keras NLP**: For fine-tuning and deep learning model development.
- **spaCy**: Efficient tokenization and part-of-speech tagging.
- **NLTK**: Foundational NLP tasks, e.g., tokenization and stemming.
- **Pandas**: Essential for data manipulation and analysis.

#### Installation Commands

```bash
!pip install transformers keras keras-nlp spacy nltk pandas tensorflow

# NLTK Downloads
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# spaCy for Marathi
python -m spacy download marathi
```

### Key Aspects for Gemma Model Tuning

1. **Dataset Creation/Curation**: Outline data sources, preprocessing steps, and address cultural sensitivity.
2. **Fine-tuning Gemma**: Detail hyperparameter settings, training procedures, and performance-enhancing techniques.
3. **Inference and Evaluation**: Demonstrate model inference and evaluate performance.

### Important Features for Gemma Model

- **Language Fluency**: Fine-tune for natural text generation in Marathi.
- **Literary Traditions**: Adapt for generating or analyzing traditional literary forms.
- **Historical Texts**: Enable processing of historical documents.

### Project Steps in Detail

1. **Data Collection**
   - Sources: APIs, web scraping (e.g., BeautifulSoup), Kaggle datasets, huggingface.

2. **Data Cleaning**
   - Tokenization (spaCy, NLTK), stop word removal, punctuation stripping, normalization (lowercasing, stemming, lemmatization).
   - Handle missing values using Pandas.

3. **Training and Testing Model**
   - Split data (using `train_test_split` from sklearn).
   - Fine-tune using the Gemma-2 model on Keras.

4. **Model Evaluation**
   - Evaluate with accuracy, F1 score, precision, recall.
   - Use cross-validation to ensure robustness.

5. **Hyperparameter Tuning (Extra)**
   - Use `GridSearchCV` or `RandomizedSearchCV` for optimal parameter setting.

### Reference Resources

- **NLP Processing**: [`YouTube Video`](https://www.youtube.com/watch?v=fNxaJsNG3-s&list=PLQY2H8rRoyvzDbLUZkbudP-MFQZwNmU4S)
- **Gemma Resources**: [`Gemma Docs`](https://ai.google.dev/gemma/docs/codegemma/keras_quickstart)
- **Train/Test Visualization**: [`Train/Test Guide`](https://mlu-explain.github.io/train-test-validation/)
- **Keras Guide**: [`Keras Reference`](https://www.javatpoint.com/keras)
- **Hyperparameter Tuning**: [`Hyperparameter Guide`](https://www.javatpoint.com/hyperparameters-in-machine-learning)

### Submission Instructions

- Publish a public Kaggle Notebook showing fine-tuning of Gemma for Marathi.
- Share the Gemma model variant on Kaggle models with inference steps.
- All team members should be listed as collaborators on the Notebook.
- Submit the Notebook through the [`Google Form`](https://www.kaggle.com/gemma-language-tuning-submissions).

**Deadline**: January 14, 2025, 11:59 PM IST

### Useful Links

- **Google Form for Submission**: [`Submission Form`](https://www.kaggle.com/gemma-language-tuning-submissions)
- **Gemma Model Documentation**: [`Gemma Docs`](https://ai.google.dev/gemma/docs/base)
- **Rules**: [`Competition Rules`](https://www.kaggle.com/competitions/gemma-language-tuning/rules)
- **Dataset Source**: [`Hugging Face Dataset`](https://huggingface.co/datasets/ylacombe/google-marathi/viewer)

**Note**: Using Kaggle's T4-GPU for Gemma fine-tuning is recommended for optimal performance.



**Here is the link of the code of kaggle [click here](https://www.kaggle.com/code/manavdarji18/gemma-model-train-in-marathi-language/notebook)**

**Here is the link of the model i use [click here](https://www.kaggle.com/models/keras/gemma2/Keras/gemma2_instruct_2b_en/1)**

**Here is the link of the model i have train it [click here](https://www.kaggle.com/models/manavdarji18/gemma2_2b_mr)**
