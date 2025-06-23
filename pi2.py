# Стандартные библиотеки для работы с массивами и табличными данными
import learn
import numpy as np
import pandas as pd
# Для работы с регулярными выражениями
import re
import sklearn
# Для визуализации графиков
import matplotlib.pyplot as plt
# Библиотека для обработки естественного языка, включая работу со стоп-словами и лемматизацию
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Библиотека для построения и тренировки модели с использованием глубокого обучения, включая слои нейронной сети
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.regularizers import l2
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Библиотека для разделения данных на обучающую и тестовую выборки, балансировки классов и вычисления метрик
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix

# FastAPI и Pydantic
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Cкачивание необходимых данных для лемматизации и работы со стоп-словами
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# Функция для чтения CSV-файлов по ссылке. В URL заменяется часть для получения прямого доступа
def readCsvByLink(url):
    url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]  # Извлекаем id файла
    return pd.read_csv(url)


# Загрузка обучающего и тестового наборов данных из Google Drive
train_df = readCsvByLink("https://drive.google.com/file/d/10_zwLLKTklGvnZpTXJYWouPmGmVt5Je_/view?usp=drive_link")
test_df = readCsvByLink("https://drive.google.com/file/d/17epsBjuyGCkBbNMtyyOo6DJTfeTW-am_/view?usp=drive_link")

f"Постов для обучения: {train_df.shape[0]}, для тестирования: {str(test_df.shape[0])}"

train_df[["text", "target"]]

test_df["text"]


# Функция приводит текст к нижнему регистру, лемматизирует каждое слово, удаляя стоп-слова и слова длиной менее 3 символов
def clean_text(texts):
    cleaned_texts = []
    for text in texts:
        if not isinstance(text, str):
            text = ""
        text = re.sub(r'[^a-zA-Z]', ' ', text)  # Удаление символов
        text = text.lower()
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
        cleaned_texts.append(' '.join(words))
    return cleaned_texts


train_texts = clean_text(train_df['text'])
test_texts = clean_text(test_df['text'])

num_words = 10000  # Максимальное количество слов
max_post_len = 200  # Максимальная длина текстов

tokenizer = Tokenizer(
    num_words=num_words)  # Создается токенизатор, который преобразует текст в последовательности индексов слов
tokenizer.fit_on_texts(train_texts)  # Обучение токенизатора на текстах обучающего набора данных
# Сохраняем токенизатор в файл
import joblib
joblib.dump(tokenizer, 'tokenizer.pkl')  # Сохраняем токенизатор в файл

list(tokenizer.word_index.items())[:20]  # 20 наиболее часто встречающихся слов и соответствующие им номера

x_train_seq = tokenizer.texts_to_sequences(train_texts)
x_test_seq = tokenizer.texts_to_sequences(test_texts)

print(x_train_seq[0], "\n", train_texts[0])

x_train_padded = pad_sequences(x_train_seq, maxlen=max_post_len)
x_test_padded = pad_sequences(x_test_seq, maxlen=max_post_len)

y_train = train_df['target'].values

# Процесс повторного добавления примеров с целевым значением 1 в обучающий набор, чтобы сбалансировать количество примеров с метками 0 и 1
train_df_balanced = pd.concat([train_df, train_df[train_df['target'] == 1]])
# Это делается с помощью функции resample(), что позволяет избежать проблемы несбалансированных классов при обучении
train_df_balanced = resample(train_df_balanced, replace=True, n_samples=len(train_df_balanced), random_state=42)
x_train_balanced = clean_text(train_df_balanced['text'])
x_train_seq_balanced = tokenizer.texts_to_sequences(x_train_balanced)
x_train_padded_balanced = pad_sequences(x_train_seq_balanced, maxlen=max_post_len)
y_train_balanced = train_df_balanced['target'].values

# Используется, чтобы разделить данные на обучающую и валидационную выборки (80% для обучения, 20% для валидации).
x_train, x_val, y_train, y_val = train_test_split(
    x_train_padded_balanced, y_train_balanced, test_size=0.2
)

# Создание последовательной модели
model = Sequential([
    Embedding(input_dim=num_words, output_dim=100),
    Conv1D(filters=64, kernel_size=5, activation='relu'),  # 1D свертка
    MaxPooling1D(pool_size=2),  # Субдискретизация
    GlobalMaxPooling1D(),  # Глобальное максимальное объединение
    Dense(1, activation='sigmoid')  # Выходной слой
])

# Компиляция модели
model.compile(
    optimizer='adam',  # Настройка оптимизатора
    loss='binary_crossentropy',  # Функции потерь
    metrics=['accuracy']  # Метрики для оценки модели
)

# Использование обратных вызовов для сохранения наилучших весов модели и остановки обучения, если валидационная ошибка не улучшается.
model_save_path = 'best_1conv_model_balanced.keras'
checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Обучение модели
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[checkpoint, early_stopping],
    verbose=1
)

# Загрузка лучших весов
model.load_weights(model_save_path)

# Предсказания на тестовой выборке
predictions = model.predict(x_test_padded)
binary_predictions = (predictions > 0.5).astype(int).flatten()

# Сохранение предсказаний
test_df['target'] = binary_predictions
test_df[['id', 'target']].to_csv('submission.csv', index=False)

# Проверка метрик на валидационной выборке
y_pred_val = (model.predict(x_val) > 0.5).astype(int).flatten()
print("Показатели проверки:")
print(classification_report(y_val, y_pred_val))

for i in range(50):
    print(f"Твит: {test_df['text'].iloc[i]}")
    print(f"Предсказание: {'Настоящая Катастрофа' if binary_predictions[i] else 'Фейк'}\n")

train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

plt.plot(train_accuracy, label='Точность обучения')
plt.plot(val_accuracy, label='Точность валидации')
plt.title('Точность в зависимости от эпохи')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.show()

common_texts = set(train_df['text']).intersection(set(test_df['text']))
print(f"Общее количество пересечений текстов: {len(common_texts)}")

train_vocab = set(' '.join(train_df['text']).split())
test_vocab = set(' '.join(test_df['text']).split())
overlap = len(train_vocab.intersection(test_vocab)) / len(test_vocab) * 100
print(f"Совпадение токенов между обучением и тестом: {overlap:.2f}%")

model.summary()

# FastAPI интеграция

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Инициализация FastAPI
app = FastAPI()

# Загрузка модели и токенизатора
model = load_model('best_1conv_model_balanced.keras')
tokenizer = joblib.load('tokenizer.pkl')  # Загрузка токенизатора


# Модель для запроса
class TextRequest(BaseModel):
    text: str


# Функция для чистки текста
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(texts):
    cleaned_texts = []
    for text in texts:
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        text = text.lower()
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
        cleaned_texts.append(' '.join(words))
    return cleaned_texts


# API для предсказания
@app.post("/predict/")
def predict(request: TextRequest):
    # Чистим текст
    cleaned_text = clean_text([request.text])

    # Преобразуем текст в последовательность
    seq = tokenizer.texts_to_sequences(cleaned_text)
    padded_seq = pad_sequences(seq, maxlen=200)

    # Получаем предсказание
    prediction = model.predict(padded_seq)
    result = "Настоящая Катастрофа" if prediction > 0.5 else "Фейк"

    return {"prediction": result}


# Тестовая страница
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}
