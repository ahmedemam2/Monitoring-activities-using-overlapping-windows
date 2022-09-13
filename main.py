import mediapipe as mp
import cv2
from dollarpy import Recognizer, Template, Point
from deepface import DeepFace
from sklearn.neighbors import KNeighborsClassifier
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()



X_train = []
landmarks = []
labellist = []
cap = cv2.VideoCapture('standtrain.mp4')
landlist = []
while cap.isOpened():
    success, img = cap.read()
    try:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:

        break
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            #             if id == 16 or id == 12 or id == 11:
            h, w, c = img.shape
            landlist.append(Point(lm.x, lm.y))
            landmarks.append((lm.x,lm.y))
            labellist.append('stand')

    cv2.imshow("Image", img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
X_train.append(landmarks)
tmpl_0 = Template('stand', landlist)

cap = cv2.VideoCapture('raisehandtrain.mp4')
landlist = []
while True:
    success, img = cap.read()
    try:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:

        break
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            #             if id == 16 or id == 12 or id == 11:
            h, w, c = img.shape
            landlist.append(Point(lm.x, lm.y))
            landmarks.append((lm.x,lm.y))
            labellist.append('raisehand')
    cv2.imshow("Image", img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
X_train.append(landmarks)
tmpl_1 = Template('raisehand', landlist)

cap = cv2.VideoCapture('lowerhandtrain.mp4')
landlist = []
while True:
    success, img = cap.read()
    try:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:

        break
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            #             if id == 16 or id == 12 or id == 11:
            h, w, c = img.shape
            landlist.append(Point(lm.x, lm.y))
            landmarks.append((lm.x,lm.y))
            labellist.append('lowerhand')

    cv2.imshow("Image", img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
X_train.append(landmarks)
tmpl_2 = Template('lowerhand', landlist)

cap = cv2.VideoCapture('sittrain.mp4')
landlist = []
while True:
    success, img = cap.read()
    try:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:

        break
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            #             if id == 16 or id == 12 or id == 11:
            h, w, c = img.shape
            landlist.append(Point(lm.x, lm.y))
            landmarks.append((lm.x,lm.y))
            labellist.append('sit')
    cv2.imshow("Image", img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
X_train.append(landmarks)
tmpl_3 = Template('sit', landlist)

recognizer = Recognizer([tmpl_0,tmpl_1,tmpl_2,tmpl_3
                        ])# recognizer = Recognizer([tmpl_1])


def GetFramesOfVideo(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    #     print ("FRAMES COUNT OF TEST :" ,frames)
    return frames


def GetLandMarksFromVideo(videoPath, target_lms):
    mp_drawing = mp.solutions.drawing_utils

    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(videoPath)

    landmarks = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Recolor image to RGZB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)
            if results.pose_landmarks == None:
                continue

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            lm = results.pose_landmarks.landmark

            index = 0

            for target in target_lms:
                landmarks.append(Point(lm[target].x, lm[target].y))

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    return landmarks


#
# print (GetLandMarksFromVideo("VIDEOS/juggle/Juggle_1.mp4"))


cap = cv2.VideoCapture('testvideoadhd.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps
seconds = int(duration % 60)
video = frame_count / seconds
framesOfVideo = GetFramesOfVideo("testvideoadhd.mp4")
landmarkstest = []
end = int(fps * 2)
# print(end)
step = int(end / 10)
# print(step)
raisehand = 0
stand = 0
sit = 0
lowerhand = 0
testlist = []
count = 0
y_train = labellist
templist = []
templist2 = []
flagfirst = False
cap = cv2.VideoCapture('testvideoadhd.mp4')
while True:
    success, img = cap.read()
    try:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        break
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            #             if id == 16 or id == 12 or id == 11:
            h, w, c = img.shape
            testlist.append(Point(lm.x, lm.y))
            if flagfirst == True:
                if count >= end - step and count <= end:
                    templist.append((Point(lm.x, lm.y)))
    if count == end:
        count = 0
        #         count=0
        flagfirst = True
        testlist = templist2 + testlist
        results, score = recognizer.recognize(testlist)
        text = results
        print(results)
        print(score)
        if results == 'stand':
            stand += 1
        if results == 'sit':
            sit += 1
        if results == 'raisehand':
            raisehand += 1
        if results == 'lowerhand':
            lowerhand += 1
    count += 1
    img = cv2.putText(img, "stand" + ":" + str(stand), (00, 185), cv2.FONT_HERSHEY_SIMPLEX, 1,
                      (0, 0, 255), 2, cv2.LINE_AA, False)
    img = cv2.putText(img, "sit" + ":" + str(sit), (00, 220), cv2.FONT_HERSHEY_SIMPLEX, 1,
                      (0, 0, 255), 2, cv2.LINE_AA, False)
    img = cv2.putText(img, "raisehand" + ":" + str(raisehand), (00, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                      (0, 0, 255), 2, cv2.LINE_AA, False)
    img = cv2.putText(img, "lowerhand" + ":" + str(lowerhand), (00, 280), cv2.FONT_HERSHEY_SIMPLEX, 1,
                      (0, 0, 255), 2, cv2.LINE_AA, False)
    testlist = []
    templist2 = templist
    templist = []
    cv2.imshow("feed", img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
print(stand, sit, raisehand, lowerhand)
