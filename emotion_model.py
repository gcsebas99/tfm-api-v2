import sys, os
print("Current sys.path:", sys.path)
print("Current working directory:", os.getcwd())
#import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# ---------------------------
# Configuración de TensorFlow y parámetros globales
# ---------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Minimiza los mensajes de log de TensorFlow
tf.get_logger().setLevel('ERROR')

# Dimensiones de las imágenes y tamaño de batch
IMG_HEIGHT, IMG_WIDTH = 48, 48
BATCH_SIZE = 16

# Ruta base del proyecto y archivo donde se guardará el modelo entrenado y el dataset
DATA_DIR = r'./'
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "emotion_recognition_model.keras"
MODEL_PATH = os.path.join(DATA_DIR, MODEL_FILENAME)

# Directorio donde se encuentra el dataset (ajusta esta ruta según tu entorno)
#DATA_DIR = r'C:\Dataset_Affecnet'

# ---------------------------
# Mapeo de emociones y traducción
# ---------------------------
# Diccionario que asigna un índice a cada emoción (tal como se organizan las carpetas en DATA_DIR)
emotion_mapping = {
    "Alegria": 0,
    "Desprecio": 1,
    "Disgusto": 2,
    "Ira": 3,
    "Miedo": 4,
    "Neutral": 5,
    "Sorpresa": 6,
    "Tristeza": 7
}

# Diccionario para traducir los nombres de las emociones a inglés (o a la nomenclatura que prefieras)
translation = {
    "Alegria": "happy",
    "Desprecio": "contempt",
    "Disgusto": "disgust",
    "Ira": "angry",
    "Miedo": "fear",
    "Neutral": "neutral",
    "Sorpresa": "surprise",
    "Tristeza": "sad"
}

# ---------------------------
# Función para preprocesar la imagen
# ---------------------------
def preprocess_image(image_path):
    """
    Carga y preprocesa una imagen:
      - Redimensiona la imagen a (IMG_HEIGHT, IMG_WIDTH)
      - La convierte a escala de grises
      - Normaliza los valores de pixel a [0,1]

    Parámetro:
      image_path: Ruta de la imagen a procesar

    Retorna:
      img_array: Array preprocesado listo para la predicción.
    """
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale')
    img_array = img_to_array(img) / 255.0
    return img_array

# ---------------------------
# Función para construir y compilar el modelo
# ---------------------------
def build_model():
    """
    Define y compila la arquitectura del modelo de reconocimiento de emociones.

    Retorna:
      model: Modelo compilado listo para entrenar o predecir.
    """
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(len(emotion_mapping), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------------------------
# Función para entrenar el modelo
# ---------------------------
def train_model():
    """
    Entrena el modelo de reconocimiento de emociones usando las imágenes del dataset.
    Se asume que en DATA_DIR existen carpetas nombradas según las claves de emotion_mapping.

    Procedimiento:
      - Se generan listas de rutas de imágenes y sus etiquetas
      - Se dividen en entrenamiento y validación
      - Se crean generadores de datos para alimentar el entrenamiento
      - Se entrena el modelo por un número definido de épocas y se guarda en MODEL_PATH

    Retorna:
      model: El modelo entrenado.
    """
    # Generar listas de imágenes y etiquetas según las carpetas
    image_paths = []
    image_labels = []
    for folder, emotion_id in emotion_mapping.items():
        folder_path = os.path.join(DATA_DIR, folder)
        if os.path.isdir(folder_path):
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(folder_path, img_file))
                    image_labels.append(emotion_id)
    print(f"Total imágenes detectadas: {len(image_paths)}")

    # Dividir en conjuntos de entrenamiento y validación
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, image_labels, test_size=0.2, random_state=42
    )
    print(f"Imágenes en entrenamiento: {len(train_paths)}")
    print(f"Imágenes en validación: {len(val_paths)}")

    # Definir el generador de datos
    def data_generator(paths, labels):
        while True:
            for i in range(0, len(paths), BATCH_SIZE):
                batch_paths = paths[i:i+BATCH_SIZE]
                batch_labels = labels[i:i+BATCH_SIZE]
                images = np.array([preprocess_image(path) for path in batch_paths])
                one_hot_labels = tf.keras.utils.to_categorical(batch_labels, num_classes=len(emotion_mapping))
                yield images, one_hot_labels

    train_gen = data_generator(train_paths, train_labels)
    val_gen = data_generator(val_paths, val_labels)

    train_steps = (len(train_paths) + BATCH_SIZE - 1) // BATCH_SIZE
    val_steps = (len(val_paths) + BATCH_SIZE - 1) // BATCH_SIZE

    # Construir y compilar el modelo
    model = build_model()

    # Número de épocas para entrenar (ajusta según sea necesario)
    EPOCHS = 20
    print("Iniciando entrenamiento...")
    model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=EPOCHS
    )

    # Guardar el modelo entrenado
    model.save(MODEL_PATH)
    print("Modelo entrenado y guardado en:", MODEL_PATH)
    return model

# ---------------------------
# Función para obtener el modelo: cargarlo si existe, o entrenarlo en caso contrario
# ---------------------------
def get_model():
    """
    Intenta cargar un modelo ya entrenado desde MODEL_PATH.
    Si no existe, entrena el modelo utilizando la función train_model().

    Retorna:
      model: El modelo cargado o entrenado.
    """
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("Modelo cargado exitosamente desde:", MODEL_PATH)
    else:
        print("No se encontró un modelo guardado. Se procederá a entrenarlo...")
        model = train_model()
    return model

# Cargar (o entrenar) el modelo y asignarlo a una variable global
model = get_model()

# ---------------------------
# Función principal: process_emotions
# ---------------------------
#def process_emotions(image_path):
def process_emotions(face_image):
    """
    Recibe la ruta de una imagen, la procesa y retorna un diccionario
    con los porcentajes de cada emoción en formato fracción (0 a 1).

    Ejemplo de retorno:
      {"neutral": 0.20, "happy": 0.50, "angry": 0.05, "fear": 0.02,
       "disgust": 0.01, "surprise": 0.06, "contempt": 0.01, "sad": 0.10}

    Parámetro:
      image_path: Ruta a la imagen que se desea evaluar.

    Retorna:
      results: Diccionario con el nombre de cada emoción (en inglés) y su probabilidad.
    """
    # Preprocesar la imagen y agregar la dimensión de batch
    #img = preprocess_image(image_path)
    face_crop_resized = cv2.resize(face_image, (IMG_HEIGHT, IMG_WIDTH))
    face_crop_gray = cv2.cvtColor(face_crop_resized, cv2.COLOR_BGR2GRAY)
    face_array = img_to_array(face_crop_gray)  # shape (48, 48, 1)
    img = face_array / 255.0
    img = np.expand_dims(img, axis=0)

    # Realizar la predicción
    prediction = model.predict(img)
    probabilities = prediction[0] * 100  # Convertir a porcentaje (opcional)

    # Construir el diccionario de resultados
    results = {}
    for idx, prob in enumerate(probabilities):
        # print(f"% at index {idx} is {prob} and rounded to {round(prob, 1)}")
        # Obtener la emoción correspondiente al índice
        emotion_spanish = list(emotion_mapping.keys())[list(emotion_mapping.values()).index(idx)]
        emotion_english = translation[emotion_spanish]
        results[emotion_english] = float(f"{prob:.2f}") # Convertir a fracción y redondear
    return results

# ---------------------------
# Bloque de prueba
# ---------------------------
if __name__ == '__main__':
    import sys
    # Si se pasa el argumento "train" se fuerza el reentrenamiento
    if len(sys.argv) > 1 and sys.argv[1].lower() == "train":
        print("Reentrenando el modelo...")
        model = train_model()
    else:
        print("Usando el modelo cargado (o entrenado previamente).")

    # Ejemplo de uso de process_emotions con una imagen de prueba
    test_image_path = os.path.join(BASE_DIR, "test_image.jpg")
    if os.path.exists(test_image_path):
        resultados = process_emotions(test_image_path)
        print("Resultados de la predicción:", resultados)
    else:
        print("No se encontró la imagen de prueba:", test_image_path)
