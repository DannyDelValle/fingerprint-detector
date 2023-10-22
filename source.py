import cv2
import numpy as np
import os
import timeit

score = 0
bad_score = 0

def preprocess_image(image):
    """
    Mejora de bordes
    """
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image

def get_test_image(carpet, num, gen, pos, type_fin, etc):
    """
    Obtener imagen de prueba
    """
    # print(f"Imagen de prueba: {num}__{gen}_{pos}_{type_fin}_finger_{etc}")
    image_test = cv2.imread(f"./test_fingerprint/{carpet}/{num}__{gen}_{pos}_{type_fin}_finger_{etc}")
    image_test = preprocess_image(image_test)
    original = f"{num}__{gen}_{pos}_{type_fin}_finger.BMP"
    return image_test, original

def test_image(score, bad_score, image_test, original):
    max_acurracy = 0
    for file in [file for file in os.listdir("base_fingerprint")]:
        fingerprint_database_image = cv2.imread("./base_fingerprint/"+file)
        fingerprint_database_image = preprocess_image(fingerprint_database_image)
        
        sift = cv2.SIFT_create()
        keypoints_1, descriptors_1 = sift.detectAndCompute(image_test, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_database_image, None)
        
        flann_params = dict(algorithm=1, trees=10)
        matcher = cv2.FlannBasedMatcher(flann_params, {})
        matches = matcher.knnMatch(descriptors_1, descriptors_2, k=2)
        
        match_points = []
        for p, q in matches:
            if p.distance < 0.7 * q.distance:
                match_points.append(p)

        keypoints = min(len(keypoints_1), len(keypoints_2))
        acurracy = len(match_points) / keypoints

        if max_acurracy < acurracy:
            max_acurracy = acurracy
            max_acurracy = int(max_acurracy * 10000) / 10000

        # guardar en un archivo txt el nombre de la imagen y el porcentaje de coincidencia
        if keypoints and (len(match_points) / keypoints) > 0.5: 
            # print(f"% Coincidencia: {len(match_points) / keypoints * 100}")
            # print(f"ID de la huella desconocida: {file}")
            if file == original:
                # mostrar imagen fingerprint_database_image
                # cv2.imshow("Imagen de la base de datos", fingerprint_database_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                with open("resultados2.txt", "a") as file:
                    file.write(f"Imagen: {original}\tCoincidencia: {len(match_points) / keypoints * 100}\n")
                # print("La huella desconocida pertenece a la persona de la base de datos. Acurracy: ", max_acurracy, file, original)
                # mostrar imagen que coincide
                score = score + 1
                return score, bad_score
            else:
                with open("resultados2.txt", "a") as file:
                    file.write(f"ERROR\tImagen: {original}\tCoincidencia: {len(match_points) / keypoints * 100}\n")
                # print("La huella desconocida no pertenece a la persona de la base de datos")
                bad_score = bad_score + 1
        elif file == os.listdir("base_fingerprint")[-1]:
            with open("resultados2.txt", "a") as file:
                file.write(f"ERROR\tImagen: {original}\n")
            print("La huella desconocida no pertenece a la persona de la base de datos")
            bad_score = bad_score + 1
            return score, bad_score
        print(f"Score: {score}\tBad Score: {bad_score}\t{max_acurracy}\tProcesando imagen: {original} con {file}", end="\r")

        
def get_score(score, bad_score):
    for file in [file for file in os.listdir("test_fingerprint/Altered-Easy")]:
        parametres = file.split("_")
        num = parametres[0]
        gen = parametres[2]
        pos = parametres[3]
        type_fin = parametres[4]
        etc = parametres[6]
        image_test, original = get_test_image('Altered-Easy', num, gen, pos, type_fin, etc)
        score, bad_score = test_image(score, bad_score, image_test, original)

    for file in [file for file in os.listdir("test_fingerprint/Altered-Medium")]:
        parametres = file.split("_")
        num = parametres[0]
        gen = parametres[2]
        pos = parametres[3]
        type_fin = parametres[4]
        etc = parametres[6]
        image_test, original = get_test_image('Altered-Medium', num, gen, pos, type_fin, etc)
        score, bad_score = test_image(score, bad_score, image_test, original)

    for file in [file for file in os.listdir("test_fingerprint/Altered-Hard")]:
        parametres = file.split("_")
        num = parametres[0]
        gen = parametres[2]
        pos = parametres[3]
        type_fin = parametres[4]
        etc = parametres[6]
        image_test, original = get_test_image('Altered-Hard', num, gen, pos, type_fin, etc)
        score, bad_score = test_image(score, bad_score, image_test, original)

    return score, bad_score

# declarar main
if __name__ == "__main__":
    get_score(score, bad_score)
    print(f"Score: {score}")
    print(f"Bad Score: {bad_score}")
    start = timeit.default_timer()    
    # image_test, label_original = get_test_image('Altered-Hard', '101', 'M', 'Right', 'little', 'Zcut.BMP')
    # cv2.imshow("Imagen de prueba", image_test)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # score, bad_score = test_image(score, bad_score, image_test, label_original)
    stop = timeit.default_timer()
    # print(f"Score: {score}")
    # print(f"Bad Score: {bad_score}")
    print(f"Tiempo de ejecución: {stop - start}")
