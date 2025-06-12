import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load data
data = pd.read_csv('sentiment_140.csv', encoding='latin-1', header=None, 
                   names=['target', 'ids', 'date', 'flag', 'user', 'text'])

# Sample data
data = data.sample(n=100000, random_state=42)

# Map targets: 0->negative, 4->positive
data['sentiment'] = data['target'].map({0: 'negative', 4: 'positive'})
data = data.dropna()

# Clean text (remvoes URLs, mentions, hashtags, and converts to lowercase)
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    return text.lower().strip()

data['clean_text'] = data['text'].apply(clean_text)

# Prepare data
X = data['clean_text']
y = data['sentiment']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Tokenize
tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=100, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=100, padding='post')

# Build model
model = Sequential([
    Embedding(20000, 100, input_length=100),
    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
print("Training...")
model.fit(X_train_pad, y_train, epochs=40, batch_size=512, validation_split=0.1, verbose=1)

# Test
accuracy = model.evaluate(X_test_pad, y_test, verbose=0)[1]
print(f"Accuracy: {accuracy:.4f}")

# Save everything
model.save('model.h5')
joblib.dump(tokenizer, 'tokenizer.joblib')
joblib.dump(label_encoder, 'encoder.joblib')

print("Done! Files saved: model.h5, tokenizer.joblib, encoder.joblib")