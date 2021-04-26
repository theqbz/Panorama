import cv2
import numpy as np
import random


# szukseg esetere egy atmeretezes

def resized(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)


# egyezo(-nek gondolt) pontok osszekotese

def draw_matctes(img1, img2, keyP1, keyP2, commonPts):
    r, c = img1.shape[:2]
    r1, c1 = img2.shape[:2]

    imgToReturn = np.zeros((max([r, r1]), c + c1, 3), dtype='uint8')
    imgToReturn[:r, :c, :] = np.dstack([img1, img1, img1])
    imgToReturn[:r1, c:c + c1, :] = np.dstack([img2, img2, img2])

    for match in commonPts:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = keyP1[img1_idx].pt
        (x2, y2) = keyP2[img2_idx].pt

        cv2.circle(imgToReturn, (int(x1), int(y1)), 4, (0, 255, 255), 1)
        cv2.circle(imgToReturn, (int(x2) + c, int(y2)), 4, (0, 255, 255), 1)

        cv2.line(imgToReturn, (int(x1), int(y1)), (int(x2) + c, int(y2)), (0, 255, 255), 1)

    return imgToReturn


# transzformacio

def warp(img1, img2, H):
    # szelessegi es magassagi adatok
    r1, c1 = img1.shape[:2]
    r2, c2 = img2.shape[:2]

    points1 = np.float32([[0, 0], [0, r1], [c1, r1], [c1, 0]]).reshape(-1, 1, 2)
    temp = np.float32([[0, 0], [0, r2], [c2, r2], [c2, 0]]).reshape(-1, 1, 2)
    points2 = cv2.perspectiveTransform(temp, H)

    points = np.concatenate((points1, points2), axis=0)

    [x_min, y_min] = np.int32(points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]

    # 3*3-as transzformacios matrix
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    # A masodik kep torzitasa majd az eelso beillesztese
    imgToReturn = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    imgToReturn[translation_dist[1]:r1 + translation_dist[1], translation_dist[0]:c1 + translation_dist[0]] = img1

    return imgToReturn


# random zaj hozzáadása
# forrás: https://gist.github.com/Prasad9/28f6a2df8e8d463c6ddd040f4f6a028a#gistcomment-2857098
def add_g_noise(img):
    mean = 1
    var = 100
    sigma = var ** 0.9
    gaussian0 = np.random.normal(mean, sigma, img.shape[:2])
    gaussian1 = np.random.normal(mean, sigma, img.shape[:2])
    gaussian2 = np.random.normal(mean, sigma, img.shape[:2])
    noisy_image = np.zeros(img.shape, np.float32)
    if len(img.shape) == 2:
        noisy_image = img + gaussian0
    else:
        noisy_image[:, :, 0] = img[:, :, 0] + gaussian0
        noisy_image[:, :, 2] = img[:, :, 2] + gaussian1
        noisy_image[:, :, 1] = img[:, :, 1] + gaussian2
    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)
#    cv2.imshow("gaussian0", gaussian0)
#    cv2.imshow("gaussian1", gaussian1)
#    cv2.imshow("gaussian2", gaussian2)
    cv2.imshow("noisy", noisy_image)
#    cv2.waitKey(0)

    return noisy_image


# még egy random zaj
# forrás: https://www.geeksforgeeks.org/add-a-salt-and-pepper-noise-to-an-image-with-python/
def add_noise(bwimg):
    # Getting the dimensions of the image
    row, col = bwimg.shape

    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to white
        bwimg[y_coord][x_coord] = 255

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        bwimg[y_coord][x_coord] = 0

    return bwimg


# Kepek betoltese

img1 = cv2.imread('kepek/park1s.jpg')
img2 = cv2.imread('kepek/park2s.jpg')

# zaj hozzáadása az elso képhez

img1_gnoise = add_g_noise(img1)

# Szurkearnyalat

img1_gray = cv2.cvtColor(img1_gnoise, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# masfele zaj a masodik kepre

img1_gray = add_noise(img1_gray)
img2_gray = add_noise(img2_gray)

# Szurke kepek megmutatasa

cv2.imshow("1 szurke", resized(img1_gray, 1))
cv2.imshow("2 szurke", resized(img2_gray, 1))

# Kulcspontok keresese es leiro keszitese

orb = cv2.ORB_create(nfeatures=2000)
keyPts1, descriptors1 = orb.detectAndCompute(img1_gray, None)
keyPts2, descriptors2 = orb.detectAndCompute(img2_gray, None)

# Kulcspontok kepre rajzolasa

img1_keypts = cv2.drawKeypoints(img1_gray, keyPts1, None, (255, 0, 255))
img2_keypts = cv2.drawKeypoints(img2_gray, keyPts2, None, (255, 0, 255))

# Kulcspontos kepek megmutatasa

# cv2.imshow("1 kulcspontok", resized(img1_keypts, 1))
# cv2.imshow("2 kulcspontok", resized(img2_keypts, 1))

# Kulcspontok osszehasonlitasa és párosítása

bfm = cv2.BFMatcher_create(cv2.NORM_HAMMING)
commonPts = bfm.knnMatch(descriptors1, descriptors2, k=2)

# egyezonek felismert pontok osszekotese

all_matches = []
for m, n in commonPts:
    all_matches.append(m)
img3 = draw_matctes(img1_gray, img2_gray, keyPts1, keyPts2, all_matches[:30])

# Osszekottetesek megmutatasa

# cv2.imshow("osszekottetesek", resized(img3, 0.5))

# Legjobb egyezesek keresese

bestPts = []
for m, n in commonPts:
    if m.distance < 0.6 * n.distance:
        bestPts.append(m)

# Legjobb egyezesek kepre rajzolasa

img1_bestCommPts = cv2.drawKeypoints(img1_gray, [keyPts1[m.queryIdx] for m in bestPts], None, (255, 0, 255))
img2_bestCommPts = cv2.drawKeypoints(img2_gray, [keyPts2[m.queryIdx] for m in bestPts], None, (255, 0, 255))

# Legjobb egyezesek megmutatasa (kulon kepen)

# cv2.imshow("Legjobb kulcspontok1", img1_bestCommPts)
# cv2.imshow("Legjobb kulcspontok2", img2_bestCommPts)

# Legjobb egyezesek osszekotese es megmutatasa

img4 = draw_matctes(img1_gray, img2_gray, keyPts1, keyPts2, bestPts)
cv2.imshow("osszekottetesek2", resized(img4, 0.6))

MinMatchCount = 10

# Ha legalább 10 egyező pontot talált akkor torzítás majd összefűzés

if len(bestPts) > MinMatchCount:
    scr_pts = np.float32([keyPts1[m.queryIdx].pt for m in bestPts]).reshape(-1, 1, 2)
    dst_pts = np.float32([keyPts2[m.trainIdx].pt for m in bestPts]).reshape(-1, 1, 2)

    H, m = cv2.findHomography(scr_pts, dst_pts, cv2.RANSAC, 5.0)
    result = warp(img2_gray, img1_gray, H)

    cv2.imshow("eredmeny", resized(result, 0.5))

else:
    print("Nem talalhato elegendo egyezo keppont a ket kepen.")

cv2.waitKey(0)
