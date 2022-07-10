import cv2
import skimage
import numpy as np
from skimage import img_as_ubyte
import matplotlib.pyplot as plt


def openCVtoSkimage(img):
    if img.ndim <= 2:
        return img
    return img[:, :, ::-1]


def SkimagetoOpenCV(img):
    return img_as_ubyte(img)


def removeShadowsFromPaper(img):
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(
            diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    # if len(result_planes) >= 1:
    #     result = cv2.merge(result_planes)
    # else:
    #     result = img
    if len(result_norm_planes) >= 1:
        result_norm = cv2.merge(result_norm_planes)
    else:
        result_norm = img

    # skimage.io.imshow(result)
    return result_norm


def getGaussianThreshold(img):
    img = img.astype("uint8")
    hist, bin_edges = skimage.exposure.histogram(img, nbins=265)
    Tnew = round((sum(hist*bin_edges))/(img.shape[0]*img.shape[1]))
    Told = 0
    while Tnew != Told:
        Told = Tnew
        hist_low = hist[hist < Told]
        hist_high = hist[hist >= Told]
        Tlow = Told
        Thigh = Told
        if sum(hist_low) != 0:
            Tlow = round(
                (sum(hist_low*bin_edges[0:len(hist_low)]))/sum(hist_low))
        if sum(hist_high) != 0:
            Thigh = round(
                (sum(hist_high*bin_edges[len(hist_low):len(hist)]))/sum(hist_high))
        Tnew = (Tlow + Thigh)/2
    return Tnew


def splitAndSegmentN(img, N):
    parts = np.hsplit(img, N)

    for i in range(len(parts)):
        T = getGaussianThreshold(parts[i])
        parts[i] = np.where(parts[i] > T, 1, 0)

    return np.hstack(parts)


def imageToBinary(img):
    gray = skimage.color.rgb2gray(img)
    binary = np.where(gray < 0.8, 0, 1)
    return binary


def plotBinaryHistogram(img):
    ax = plt.hist(img.ravel(), bins=256)
    plt.show()


def normalize(img):
    if np.max(img) <= 1:
        return img*255
    else:
        return img


def findContours(img):
    img = normalize(img)
    img = skimage.img_as_ubyte(img)
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours_inv, hierarchy_inv = cv2.findContours(
        cv2.bitwise_not(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_inv) > len(contours):
        return contours_inv, hierarchy_inv
    else:
        return contours, hierarchy


def removeBlank2(ioImage):
    img = SkimagetoOpenCV(ioImage)
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        gray = img
    gray = 255*(gray < 128).astype(np.uint8)
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    rect = img[y:y+h, x:x+w]
    return rect


def removeBlank(ioImage):
    binary = ioImage.copy()
    contours, _ = findContours(binary)
    rightest_x = 0
    leftest_x = binary.shape[0]
    lowest_y = binary.shape[1]
    highest_Y = 0
    for contour in contours[1:]:
        x, y, w, h = cv2.boundingRect(contour)
        if w*h > 100:
            if x+w > rightest_x:
                rightest_x = x+w
            if x < leftest_x:
                leftest_x = x
            if y+h > highest_Y:
                highest_Y = y+h
            if y < lowest_y:
                lowest_y = y
    margin = 100
    while lowest_y-margin < 0:
        margin -= 10
    lowest_y = lowest_y-margin
    margin = 100
    while highest_Y+margin > binary.shape[1]:
        margin -= 10
    highest_Y = highest_Y+margin
    margin = 100
    while leftest_x-margin < 0:
        margin -= 10
    leftest_x = leftest_x-margin
    margin = 100
    while rightest_x+margin > binary.shape[0]:
        margin -= 10
    rightest_x = rightest_x+margin
    return ioImage[lowest_y:highest_Y, leftest_x:rightest_x]


def drawContours(img, contours):
    img = SkimagetoOpenCV(img)
    copy = img.copy()
    cv2.drawContours(copy, contours, -1, (0, 255, 0), 3)
    return copy


def getTextLinesFromBinary(gray):
    kernel = np.ones((1, 20), np.uint8)
    binary = cv2.erode(SkimagetoOpenCV(gray), kernel, iterations=5)
    if np.mean(binary) > 0.1:
        binary = 1 - binary
    contours, _ = findContours(binary)
    boxes = []
    margin = 30
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if w > 200 and h > 50:
            ystart = y
            yend = y + h
            if y-margin >= 0:
                ystart = y-margin
            if y+h+margin <= gray.shape[1]:
                yend = y+h+margin
            box = gray[ystart:yend, x:x+w]
            boxes.append(box)
    return boxes


def extractCharactersFromBinaryImage(gray):
    binary = gray.copy()
    if np.mean(binary) > 0.1:
        binary = 1 - binary
    contours, _ = findContours(binary)
    boxes = []
    margin = 5
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if w > 10 and h > 40 and w*h > 100:
            ystart = y
            yend = y + h
            xstart = x
            xend = x + w
            if y-margin >= 0:
                ystart = y-margin
            if x-margin >= 0:
                xstart = x-margin
            if y+h+margin <= gray.shape[1]:
                yend = y+h+margin
            if x+h+margin <= gray.shape[0]:
                xend = x+h+margin
            box = gray[ystart:yend, xstart:xend]
            boxes.append(box)
    return boxes


def show_images(images, titles=None):
    if isinstance(images, list) == False:
        images = [images]
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def findAndDrawLines(img):
    copy = img.copy()
    cannied = cv2.Canny(img, 50, 200, 3)
    lines = cv2.HoughLinesP(cannied, rho=1, theta=np.pi /
                            180, threshold=80, minLineLength=30, maxLineGap=10)
    for line in lines:
        leftx, boty, rightx, topy = line[0]
        cv2.line(copy, (leftx, boty), (rightx, topy), (255, 255, 0), 2)
    return copy


def histogramOfLBP(image, eps=1e-7, numberOfPoints=16, radius=8):
    lbp = skimage.feature.local_binary_pattern(image, numberOfPoints,
                                               radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, numberOfPoints + 3),
                             range=(0, numberOfPoints + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist
