import cv2
import mediapipe as mp
import time


# 打开2号摄像头
cap = cv2.VideoCapture(0)

mpPose = mp.solutions.pose
pose = mpPose.Pose(min_tracking_confidence = 0.9,min_detection_confidence=0.9)

# 画出身体的坐标
mpDraw = mp.solutions.drawing_utils

# 坐标点样式
poseLmsStyle = mpDraw.DrawingSpec(color=(0,0,255),thickness=5)
# 线条样式
poseConStyle = mpDraw.DrawingSpec(color=(0,255,0),thickness=10)

# 计算FPS的条件初始化
pTime = 0
cTime = 0

handTag = 0
shoulderTag = 0

List = [0]*33

while True:
    ret,img = cap.read()
    if ret:

        # 身体侦测需要RGB图片，OpenCV读取的都是BGR图片，进行转换
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(imgRGB)
        img = cv2.resize(img,(1920,1080))

        # 设置窗口高度和宽度
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]

        #如果侦测到身体，则进行以下事件：画出身体的坐标点
        if result.pose_landmarks:
            # img：画在img上
            # mpPose.POSE_CONNECTIONS：将各个点连接起来形成线段
            # poseLmsStyle:绘制时用设置好的坐标点样式
            # poseConStyle:绘制时用设置好的线条样式
            mpDraw.draw_landmarks(img,result.pose_landmarks, mpPose.POSE_CONNECTIONS,poseLmsStyle,poseConStyle)

            # 打印坐标点的位置,并且知道是第几个点的坐标
            for i,lm in enumerate(result.pose_landmarks.landmark):
                xPos = int(lm.x * imgWidth)
                List[i] = yPos = int(lm.y * imgHeight)
                # 用Hershey_simplex字体样式在img上的选的位置标记每个坐标点的编号，字体大小为0.4，红色，粗度为2
                cv2.putText(img,str(i),(xPos-25,yPos+5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),2)

                # 给某个坐标点单独设置样式，以4号为例
                # if i == 4:
                    # cv2.circle(img,(xPos,yPos),20,(164,56,23),cv2.FILLED)

                # print(i,List[i])


                if i==12:
                    shoulderTag = List[i]
                if i==16:
                    handTag = List[i]
                if handTag < shoulderTag:
                    print("挥手")
                    print("----------")

                # print(i,xPos,yPos)



        # 算出FPS，一秒更新几次画面
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS:{int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)


        cv2.imshow('img',img)

    if cv2.waitKey(1) == ord(' '):
        break
