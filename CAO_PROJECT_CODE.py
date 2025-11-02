import kagglehub
import os 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import whisper
# Download latest version
path = kagglehub.dataset_download("mealss/call-transcripts-scam-determinations")

print("Path to dataset files:", path)


file_path=os.path.join(path, "BETTER30.csv")

df=pd.read_csv(file_path)
df['TEXT'] = df['TEXT'].str.replace('[^a-zA-Z\s]', '', regex=True)  # Remove non-alphabet characters
df['TEXT'] = df['TEXT'].str.lower()  # Convert to lowercase

# Fill missing values in CONTEXT and FEATURES with 'unknown' or other strategy
df['CONTEXT'] = df['CONTEXT'].fillna('unknown')
df['FEATURES'] = df['FEATURES'].fillna('unknown')


df.drop(columns=['ANNOTATIONS'], inplace=True)

# Check for missing values and drop rows with missing labels (if any)
df.dropna(subset=['TEXT', 'LABEL'], inplace=True)

# Clean the 'LABEL' column: remove leading/trailing spaces and convert to lowercase
df['LABEL'] = df['LABEL'].str.strip().str.lower()

scam_labels = [
    'scam', 'suspicious', 'highly_suspicious', 'slightly_suspicious', 'potential_scam',
    'scam_response', 'citing urgency', 'suggesting a dangerous situation', 'dismissive official protocols',
    'scam_response'
]
non_scam_labels = [
    'neutral', 'legitimate', 'standard_opening, identification_request', 'polite_ending',
    'adhering to protocols', 'emphasizing security and compliance', 'ready for further engagement'
]

def classify_label(label):
    if label in scam_labels:
        return 'scam'
    elif label in non_scam_labels:
        return 'non-scam'
    else:
        return 'unknown'  # For any labels not categorized

df['LABEL'] = df['LABEL'].apply(classify_label)

X = df['TEXT']
y = df['LABEL']

# Encode labels (scam -> 1, non-scam -> 0)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

classifier = LogisticRegression()
classifier.fit(X_train_tfidf, y_train)


y_pred = classifier.predict(X_test_tfidf)


print("Accuracy on Test Data:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# To predict for new messages
def predict_label(message):
    # Transform the input message using the same vectorizer
    message_tfidf = tfidf.transform([message])
    # Predict the label
    prediction = classifier.predict(message_tfidf)
    return label_encoder.inverse_transform(prediction)[0]


model=whisper.load_model("base")
result = model.transcribe("/Users/amangolani/Downloads/CAO/CreditCardScamAudio.mp3")
print(result["text"])


call_transcription=result["text"]

predicted_label=predict_label(call_transcription)
print("\n\nThe predicted label is\n",predicted_label)




df["combined_text"] = df["TEXT"].fillna('') + ' ' + df["CONTEXT"].fillna('') + ' ' + df["FEATURES"].fillna('')

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df["combined_text"], show_progress_bar=True)


kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(embeddings)

df["cluster"] = labels
print("Silhouette Score:", silhouette_score(embeddings, labels))


pca = PCA(n_components=2)
X_2d = pca.fit_transform(embeddings)

plt.scatter(X_2d[:,0], X_2d[:,1], c=labels, cmap='viridis')
plt.xlabel('PCA 1'); plt.ylabel('PCA 2')
plt.show()