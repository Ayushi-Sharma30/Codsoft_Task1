import joblib

# Load saved model and vectorizer
model = joblib.load("model/movie_genre_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

print("Movie Genre Predictor")
print("---------------------")

description = input("Enter movie description: ")

description_tfidf = vectorizer.transform([description])
prediction = model.predict(description_tfidf)

print("\nPredicted Genre:", prediction[0])