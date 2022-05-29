import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np
import math
import pandas as pd
from collections import Counter
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
model = tf.keras.models.load_model('model4.h5')
GESTURES = ["Flat Palm", "One", "Two", "closed fist", "three"]
x_1 = 0
x_2 = 640
y_1 = 0
y_2 = 480
camwidth = (x_2 - x_1)
camheight = (y_2 - y_1)
cap = cv2.VideoCapture(0)
notthere = 0
pastlist_x = []
pastlist_y = []
pastlistg = []

# ---------------------------------------


data = pd.DataFrame(pd.read_csv("testData.csv"))

catColNames = "fabric_type,brand,main_colour".split(',')
# conColNames = "price".split(',')

# preference = {"fabric_type": {"nylon": 4, "silk": 2, "wool": 1},
#               "brand": {"nighkey": 2, "soupreme": 1},
#               "main_colour": {"red": 2, "blue": 1},
#               "price": 20}

linWeight = 5

weights = {"fabric_type": 1,
           "brand": 1,
           "main_colour": 1,
           }  # "price": 1

nTestRounds = 10

nRows = len(data[data.columns[0]])
rollingAvgSize = 10

choiceHistory = {"fabric_type": [],
                 "brand": [],
                 "main_colour": [],
                 }  # "price": []

def thing(idx):
    vals = list(data.iloc[idx])
    cols = list(data.columns)
    stuff = list(zip(cols, vals))
    ops = ""
    for a,b in stuff:
        if a == "image_name":
            ops += f"Name: {b.split('.')[0]}, "
        else:
            ops += f"{a}: {b}, "
    return ops
def get_preference(prollingAvgSize, idx, chosen):
    global choiceHistory
    entry = data.iloc[idx].to_dict()
    if chosen:
        for col in catColNames:
            choiceHistory[col].append(entry[col])
            choiceHistory[col] = choiceHistory[col][::-1][:prollingAvgSize]
    else:
        for col in catColNames:
            if entry[col] in choiceHistory[col]:
                del choiceHistory[col][choiceHistory[col].index(entry[col])]


    preference = {}
    for col in catColNames:
        preference[col] = Counter(choiceHistory[col])
    # for col in conColNames:
    #    preference[col] = sum(choiceHistory[col][len(choiceHistory[col]) - rollingAvgSize:]) / rollingAvgSize

    return preference


def get_score(preference, target):
    score = 0
    for col in catColNames:
        if target[col] in preference[col].keys():
            score += (rollingAvgSize - preference[col][target[col]]) * weights[col]

    # sum = 0

    # for col in conColNames:
    #    sum += ((target[col] - preference[col]) ** 2) * weights[col]

    # score += linWeight / math.sqrt(sum)

    return score
def get_most_relevant_prod(currPreference, indices):
    scores = []

    for x in indices:
        b = data.iloc[x].to_dict()
        scores.append(get_score(currPreference, b))

    scores, indices = zip(*sorted(zip(scores, indices)))

    return indices[::-1][0]


# cart = []
# prod_idx = 0
# inds = list(range(nTestRounds + 1, nRows))
# current_preference = {}

class DataObject:
    def __init__(self):
        self.cart = []
        self.prod_idx = 0
        self.inds = list(range(nTestRounds + 1, nRows))
        self.current_preference = {}
currData = DataObject()
def swiped(direction):
    if currData.prod_idx <= 10:

        if direction == "UP" or direction == "RIGHT":
            t = get_preference(rollingAvgSize, currData.prod_idx, chosen=True)
            if direction == "UP":
                f = open('cart.txt', "a")
                f.write(thing(currData.prod_idx) + '\n')
                f.close()
                currData.cart.append(currData.prod_idx)
        else:
            t = get_preference(rollingAvgSize, currData.prod_idx, chosen=False)

        currData.prod_idx += 1
 
        if currData.prod_idx == 10:
            currData.current_preference = t

    if currData.prod_idx >= 10:
        if len(currData.inds) > 0:

            if direction == "UP" or direction == "RIGHT":
                currData.current_preference = get_preference(rollingAvgSize, currData.prod_idx, chosen=True)
                if direction == "UP":
                    f = open('cart.txt', "a")
                    f.write(thing(currData.prod_idx) + '\n')
                    f.close()
                    currData.cart.append(currData.prod_idx)
            else:
                currData.current_preference = get_preference(rollingAvgSize, currData.prod_idx, chosen=False)

            currData.prod_id = get_most_relevant_prod(currData.current_preference, currData.inds)
            del currData.inds[currData.inds.index(currData.prod_id)]

        else:
            print('cards ahve been exhausted')


    # TODO render card for new prod_idx


# ---------------------------------------


def swipe(difference):
    if difference > 0:
        print('left swipe')
        swiped("LEFT")

    else:
        print("right swipe")
        swiped("RIGHT")
def vswipe(difference):
    if difference < 0:
        print('up swipe')
        swiped("UP")


    else:
        print("down swipe")
        # TODO end stuff


images = {}
for filename in os.listdir('swags'):
    img = cv2.imread(os.path.join('swags', filename))
    if img is not None:
        print(filename)
        images[filename] = img

import copy


def getPerpCoord(a, b, length):
    [aX, aY] = a
    [bX, bY] = b
    vX = bX - aX
    vY = bY - aY
    # print(str(vX)+" "+str(vY))
    if (vX == 0 or vY == 0):




        return 0, 0, 0, 0
    mag = math.sqrt(vX * vX + vY * vY)
    vX = vX / mag
    vY = vY / mag
    temp = vX
    vX = 0 - vY
    vY = temp
    cX = bX + vX * length
    cY = bY + vY * length
    dX = bX - vX * length
    dY = bY - vY * length
    return [int(cX), int(cY), int(dX), int(dY)]


def triangles(points):
    points = np.where(points, points, 1)
    subdiv = cv2.Subdiv2D((*points.min(0), *points.max(0)))
    for p in list(points):
        pt = tuple([int(round(p[0])), int(round(p[1]))])
        subdiv.insert(tuple(pt))
    for pts in subdiv.getTriangleList().reshape(-1, 3, 2):
        yield [np.where(np.all(points == pt, 1))[0][0] for pt in pts]


def crop(img, pts):
    x, y, w, h = cv2.boundingRect(pts)
    img_cropped = img[y: y + h, x: x + w]
    pts[:, 0] -= x
    pts[:, 1] -= y
    return img_cropped, pts



def warp(img1, img2, pts1, pts2):
    for indices in triangles(pts1):
        img1_cropped, triangle1 = crop(img1, pts1[indices])
        img2_cropped, triangle2 = crop(img2, pts2[indices])
        transform = cv2.getAffineTransform(np.float32(triangle1), np.float32(triangle2))
        img2_warped = cv2.warpAffine(img1_cropped, transform, img2_cropped.shape[:2][::-1], None, cv2.INTER_LINEAR,
                                     cv2.BORDER_REFLECT_101)
        mask = np.zeros_like(img2_cropped)
        cv2.fillConvexPoly(mask, np.int32(triangle2), (1, 1, 1), 16, 0)
        img2_cropped *= 1 - mask
        img2_cropped += img2_warped * mask







cap = cv2.VideoCapture(0)






with mp_pose.Pose(
        min_tracking_confidence=0.5) as pose, mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image111 = copy.deepcopy(image)

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img = image
        xpos = []
        ypos = []
        p_landmarks = results.pose_landmarks
        if p_landmarks:
            vis = []
            for i in str(p_landmarks).split('landmark')[1:26]:
                i_1 = i.split()
                xpos.append(int(640 * float(i_1[2])))
                ypos.append(int(480 * float(i_1[4])))
                vis.append(float(i_1[8]))
            minx = min(xpos)
            maxx = max(xpos)
            miny = min(ypos)
            maxy = max(ypos)
            image = img
        array = np.zeros([480, 640, 3],
                         dtype=np.uint8)
        array[:, :] = [255, 255, 255]
        if len(xpos) > 24:
            r_elbowcoords = [xpos[14], ypos[14]]
            r_shouldercoords = [xpos[12], ypos[12]]
            l_elbowcoords = [xpos[13], ypos[13]]
            l_shouldercoords = [xpos[11], ypos[11]]
            l1 = getPerpCoord(r_shouldercoords, r_elbowcoords, 5)
            l2 = getPerpCoord(l_shouldercoords, l_elbowcoords, 5)
            ipos1 = [l1[0], l1[1]]

            ipos3 = [xpos[12], ypos[12] - 20]
            ipos4 = [(xpos[11] + xpos[12]) / 2, ((ypos[11] + ypos[12]) / 2) - 40]
            ipos5 = [xpos[11], ypos[11] - 20]

            ipos7 = [l2[2], l2[3]]
            ipos8 = [xpos[24] - 40, ypos[24]]
            ipos9 = [xpos[23] + 40, ypos[23]]
            pts1 = np.array([[70, 294], [254, 64], [421, 22], [587, 64], [762, 293], [256, 548], [567, 548]])
            pts2 = np.array([[int(i[0]), int(i[1])] for i in [ipos1, ipos3, ipos4, ipos5, ipos7, ipos8, ipos9]])
            try:
                warp(images[data.iloc[currData.prod_idx]["image_name"]], array, pts1, pts2)
            except:
                a = 'deez'






        h, w, c = array.shape
        image_bgra = np.concatenate([array, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
        white = np.all(array == [255, 255, 255], axis=-1)
        image_bgra[white, -1] = 0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        alpha_background = image[:, :, 3] / 255.0
        alpha_foreground = image_bgra[:, :, 3] / 255.0

        # set adjusted colors

        for color in range(0, 3):
            image[:, :, color] = alpha_foreground * array[:, :, color] + \
                                 alpha_background * image[:, :, color] * (1 - alpha_foreground)

        # set adjusted alpha and denormalize back to 0-255
        image[:, :, 3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255
        ret, buffer = cv2.imencode('.jpg', image)

        frame = buffer.tobytes()
        image = image[y_1:y_2, x_1:x_2]
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            numrighthands = 0
            for handno, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if results.multi_handedness[handno].classification[0].label == "Right":
                    numrighthands += 1
                    xcoords = [handmark.x for handmark in hand_landmarks.landmark]
                    ycoords = [handmark.y for handmark in hand_landmarks.landmark]
                    minx = min(xcoords)
                    maxx = max(xcoords)
                    miny = min(ycoords)
                    maxy = max(ycoords)
                    image = cv2.rectangle(image, (int(minx * camwidth), int(miny * camheight)),
                                          (int(maxx * camwidth), int(maxy * camheight)), (0, 0, 255), 2)
                    xcoords1 = [(i - minx) / (maxx - minx) for i in xcoords]
                    ycoords1 = [(i - miny) / (maxy - miny) for i in ycoords]
                    temp = []
                    for i in range(21):
                        temp.append(xcoords1[i])
                        temp.append(ycoords1[i])
                        temp.append(hand_landmarks.landmark[i].z)
                    ans = list(model.predict([temp])[0])
                    if max(ans) > 0.85:
                        image = cv2.putText(image, GESTURES[ans.index(max(ans))], (50, 50 * (handno + 4)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                        if ans.index(max(ans)) == 2:
                            notthere = 0
                            ee = hand_landmarks.landmark[8]
                            pastlist_x.append(ee.x)
                            pastlist_y.append(ee.y)
                            pastlistg.append(1)

                        else:
                            notthere += 1
            if numrighthands == 0:
                notthere += 1
        else:
            notthere += 1
        if notthere >= 4:
            if len(pastlist_x) > 0:

                if abs(pastlist_x[0] - pastlist_x[-1]) >= 0.1 and abs(pastlist_y[0] - pastlist_y[-1]) < 0.2:
                    swipe(pastlist_x[-1] - pastlist_x[0])
                elif abs(pastlist_x[0] - pastlist_x[-1]) < 0.2 and abs(pastlist_y[0] - pastlist_y[-1]) >= 0.2:
                    vswipe(pastlist_y[-1] - pastlist_y[0])
                elif abs(pastlist_x[0] - pastlist_x[-1]) >= 0.2 and abs(pastlist_y[0] - pastlist_y[-1]) >= 0.2:
                    if abs(pastlist_x[0] - pastlist_x[-1]) > abs(pastlist_y[0] - pastlist_y[-1]):
                        swipe(pastlist_x[-1] - pastlist_x[0])
                    elif abs(pastlist_x[0] - pastlist_x[-1]) < abs(pastlist_y[0] - pastlist_y[-1]):
                        vswipe(pastlist_y[-1] - pastlist_y[0])
            pastlist_x = []
            pastlist_y = []
            pastlistg = []
            notthere = 0
        image = cv2.putText(image, str(data.iloc[currData.prod_idx]["fabric_type"]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        image = cv2.putText(image, str(data.iloc[currData.prod_idx]["brand"]), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        image = cv2.putText(image, str(data.iloc[currData.prod_idx]["price"]), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)            
        cv2.imshow("image", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
