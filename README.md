# Movie Genre Classification

This project is developed as part of the CodSoft Data Science Internship.

## Project Description

The Movie Genre Classification project uses Machine Learning and Natural Language Processing (NLP) to predict the genre of a movie based on its description.

The movie description is converted into numerical features using TF-IDF Vectorization, and a machine learning model is trained to classify the movie into different genres.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Joblib
- Matplotlib

## Project Structure

```text
CodSoft_Task1/
│
├── Dataset/
│   ├── train_data.txt
│   └── test_data.txt
│
├── model/
│   ├── movie_genre_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebook/
│
├── src/
│   ├── movie_genre_classification.py
│   └── predict_genre.py
│
├── .gitignore
├── requirements.txt
└── README.md