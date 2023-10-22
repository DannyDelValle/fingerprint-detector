# Importación de bibliotecas necesarias
import cv2  # OpenCV para el procesamiento de imágenes
import os  # Para interactuar con el sistema de archivos
import timeit  # Para medir el tiempo de ejecución
from concurrent.futures import ThreadPoolExecutor  # Para paralelizar la tarea

# Función para preprocesar la imagen mediante el desenfoque gaussiano
def preprocess_image(image):
    return cv2.GaussianBlur(image, (3, 3), 0)

# Función para cargar imágenes de la base de datos y procesarlas
def load_database_images(folder_path):
    db_images = {}  # Diccionario para almacenar imágenes y sus descriptores
    for filename in os.listdir(folder_path):  # Recorrer todos los archivos en la carpeta
        image_path = os.path.join(folder_path, filename)  # Construir la ruta completa del archivo
        image = cv2.imread(image_path)  # Leer la imagen
        processed_image = preprocess_image(image)  # Preprocesar la imagen
        sift = cv2.SIFT_create()  # Crear el objeto SIFT para la extracción de características
        keypoints, descriptors = sift.detectAndCompute(processed_image, None)  # Extraer características
        db_images[filename] = (keypoints, descriptors)  # Almacenar en el diccionario
    return db_images

# Función para comparar dos imágenes en función de sus descriptores
def match_images(desc1, desc2):
    flann_params = dict(algorithm=1, trees=10)  # Parámetros para el algoritmo FLANN
    matcher = cv2.FlannBasedMatcher(flann_params, {})  # Crear objeto para emparejar
    matches = matcher.knnMatch(desc1, desc2, k=2)  # Encontrar los k=2 emparejamientos más cercanos
    match_points = [p for p, q in matches if p.distance < 0.7 * q.distance]  # Filtrar emparejamientos
    return match_points

# Función para procesar cada archivo de prueba
def process_file(filename, folder, db_images):
    score = 0  # Puntuación para aciertos
    bad_score = 0  # Puntuación para fallos

    image_path = os.path.join("./test_fingerprint", folder, filename)  # Ruta del archivo de prueba
    test_image = cv2.imread(image_path)  # Leer imagen de prueba
    processed_test_image = preprocess_image(test_image)  # Preprocesar imagen de prueba

    sift = cv2.SIFT_create()  # Crear objeto SIFT
    test_keypoints, test_descriptors = sift.detectAndCompute(processed_test_image, None)  # Extraer características

    max_accuracy = 0  # Para almacenar la precisión máxima encontrada
    for db_filename, (db_keypoints, db_descriptors) in db_images.items():  # Iterar sobre cada imagen en la base de datos
        match_points = match_images(test_descriptors, db_descriptors)  # Comparar características
        keypoints = min(len(test_keypoints), len(db_keypoints))  
        accuracy = 0 if keypoints == 0 else len(match_points) / keypoints  # Calcular precisión
        if max_accuracy < accuracy:  # Actualizar precisión máxima si es necesario
            max_accuracy = accuracy

        if max_accuracy > 0.5:  # Si la precisión es mayor a 0.5
            db_filename = db_filename.split("_finger")[0]
            filename = filename.split("_finger")[0]
            if db_filename == filename:
                with open("resultados3.txt", "a") as file:
                    file.write(f"Imagen: {filename}\tCoincidencia: {len(match_points) / keypoints * 100}\n")
                score += 1
                return score, bad_score
            else:
                with open("resultados3.txt", "a") as file:
                    file.write(f"ERROR\tImagen: {filename}\tCoincidencia: {len(match_points) / keypoints * 100}\n")
                bad_score += 1
                return score, bad_score
        if db_filename == list(db_images.keys())[-1] and max_accuracy < 0.5:
            with open("resultados3.txt", "a") as file:
                file.write(f"ERROR\tImagen: {filename}\tMAXIMA Coincidencia: {max_accuracy}\n")
            bad_score += 1
            return score, bad_score 

# Punto de entrada del programa
if __name__ == "__main__":

    db_images = load_database_images("./base_fingerprint")  # Cargar base de datos de imágenes
    print("Base de datos cargada: ", len(db_images))
    
    start = timeit.default_timer()  # Iniciar el temporizador
    total_score = 0  # Puntuación total para aciertos
    total_bad_score = 0  # Puntuación total para fallos

    # Paralelizar el procesamiento de las imágenes de prueba
    with ThreadPoolExecutor() as executor:
        for folder in ["Altered-Hard"]:  # Para cada nivel de dificultad
            futures = {executor.submit(process_file, filename, folder, db_images): filename for filename in os.listdir(f"./test_fingerprint/{folder}")}
            
            # Recopilar los resultados
            for future in futures:
                score, bad_score = future.result()
                total_score += score
                total_bad_score += bad_score
                print(f"Score actual: {total_score}, Bad Score actual: {total_bad_score}")

    stop = timeit.default_timer()  # Detener el temporizador
    print(f"Score final: {total_score}")
    print(f"Bad Score final: {total_bad_score}")
    print(f"Tiempo de ejecución: {stop - start}")  # Imprimir tiempo total de ejecución
