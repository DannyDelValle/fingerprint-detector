from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import os
import timeit

score = 0
bad_score = 0

def preprocess_image(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image

def get_test_image(carpet, num, gen, pos, type_fin, etc):
    image_test = cv2.imread(f"./test_fingerprint/{carpet}/{num}__{gen}_{pos}_{type_fin}_finger_{etc}")
    image_test = preprocess_image(image_test)
    original = f"{num}__{gen}_{pos}_{type_fin}_finger.BMP"
    return image_test, original

def test_single_image(file, image_test, original):
    global score
    global bad_score

    max_acurracy = 0

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

    if keypoints and (len(match_points) / keypoints) > 0.5: 
        if file == original:
            # with open("resultados2.txt", "a") as file:
            #     file.write(f"Imagen: {original}\tCoincidencia: {len(match_points) / keypoints * 100}\n")
            score += 1
        else:
            # with open("resultados2.txt", "a") as file:
            #     file.write(f"ERROR\tImagen: {original}\tCoincidencia: {len(match_points) / keypoints * 100}\n")
            bad_score += 1

def test_image(image_test, original):
    with ThreadPoolExecutor() as executor:
        executor.map(test_single_image, [file for file in os.listdir("base_fingerprint")], [image_test]*len(os.listdir("base_fingerprint")), [original]*len(os.listdir("base_fingerprint")))

def get_score():
    global score
    global bad_score

    for folder in ["Altered-Easy", "Altered-Medium", "Altered-Hard"]:
        for file in os.listdir(f"test_fingerprint/{folder}"):
            parameters = file.split("_")
            num = parameters[0]
            gen = parameters[2]
            pos = parameters[3]
            type_fin = parameters[4]
            etc = parameters[6]
            image_test, original = get_test_image(folder, num, gen, pos, type_fin, etc)
            test_image(image_test, original)
            print(f"Score: {score}, Bad Score: {bad_score}", end="\r")

if __name__ == "__main__":
    start = timeit.default_timer()
    get_score()
    stop = timeit.default_timer()

    print(f"Score: {score}")
    print(f"Bad Score: {bad_score}")
    print(f"Tiempo de ejecución: {stop - start}")
