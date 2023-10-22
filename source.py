import cv2
import os
import timeit
from concurrent.futures import ThreadPoolExecutor

def preprocess_image(image):
    return cv2.GaussianBlur(image, (3, 3), 0)

def load_database_images(folder_path):
    db_images = {}
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        processed_image = preprocess_image(image)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(processed_image, None)
        db_images[filename] = (keypoints, descriptors)
    return db_images

def match_images(desc1, desc2):
    flann_params = dict(algorithm=1, trees=10)
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    matches = matcher.knnMatch(desc1, desc2, k=2)
    match_points = [p for p, q in matches if p.distance < 0.7 * q.distance]
    return match_points

def process_file(filename, folder, db_images):
    score = 0
    bad_score = 0

    image_path = os.path.join("./test_fingerprint", folder, filename)
    test_image = cv2.imread(image_path)
    processed_test_image = preprocess_image(test_image)

    sift = cv2.SIFT_create()
    test_keypoints, test_descriptors = sift.detectAndCompute(processed_test_image, None)

    max_accuracy = 0
    for db_filename, (db_keypoints, db_descriptors) in db_images.items():
        match_points = match_images(test_descriptors, db_descriptors)
        keypoints = min(len(test_keypoints), len(db_keypoints))
        accuracy = 0 if keypoints == 0 else len(match_points) / keypoints
        if max_accuracy < accuracy:
            max_accuracy = accuracy

    if max_accuracy > 0.5:
        score += 1
    else:
        bad_score += 1

    return score, bad_score

if __name__ == "__main__":
    start = timeit.default_timer()

    db_images = load_database_images("./base_fingerprint")
    
    total_score = 0
    total_bad_score = 0

    with ThreadPoolExecutor() as executor:
        for folder in ["Altered-Easy", "Altered-Medium", "Altered-Hard"]:
            futures = {executor.submit(process_file, filename, folder, db_images): filename for filename in os.listdir(f"./test_fingerprint/{folder}")}
            
            for future in futures:
                score, bad_score = future.result()
                total_score += score
                total_bad_score += bad_score
                print(f"Score actual: {total_score}, Bad Score actual: {total_bad_score}")

    stop = timeit.default_timer()
    print(f"Score final: {total_score}")
    print(f"Bad Score final: {total_bad_score}")
    print(f"Tiempo de ejecución: {stop - start}")
