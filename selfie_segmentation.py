import cv2
import mediapipe as mp
import numpy as np

# 初始化 MediaPipe Selfie Segmentation 模块
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# 初始化 MediaPipe 绘图工具（可选，用于显示分割边界）
mp_drawing = mp.solutions.drawing_utils

# 打开摄像头
cap = cv2.VideoCapture(0)

# 加载背景图片（确保与摄像头分辨率匹配或进行调整）
background = cv2.imread('whiteboard.png')  # 替换为你自己的背景图片路径

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("无法读取摄像头帧。")
        break

    # 将 BGR 图像转换为 RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 进行自我分割
    results = selfie_segmentation.process(image)

    # 将图像转换回 BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 创建遮罩
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1

    # 如果背景图片大小与摄像头捕捉的帧不同，调整背景图片大小
    if background.shape[:2] != image.shape[:2]:
        background_resized = cv2.resize(background, (image.shape[1], image.shape[0]))
    else:
        background_resized = background

    # 合成图像：前景 + 背景
    output_image = np.where(condition, image, background_resized)

    # 可选：绘制分割边界
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_selfie_segmentation.POSE_CONNECTIONS)

    # 显示结果
    cv2.imshow('Selfie Segmentation - 背景去除', output_image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
