import cv2
import mediapipe as mp

# 初始化 MediaPipe Face Mesh 模块
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 设置 Face Mesh 的参数
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,        # 设置检测的最大人脸数量
    refine_landmarks=True,  # 是否进行瞳孔等更细节的检测
    min_detection_confidence=0.5,  # 检测置信度阈值
    min_tracking_confidence=0.5    # 跟踪置信度阈值
)

# 打开摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("无法读取摄像头帧。")
        break

    # 将 BGR 图像转换为 RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 禁用图像写入（提高性能）
    image.flags.writeable = False

    # 处理图像，得到人脸网格结果
    results = face_mesh.process(image)

    # 恢复图像写入
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 如果检测到人脸
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 绘制人脸网格
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            
            # 绘制面部轮廓、眼睛、嘴巴等细节
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            
            # 绘制瞳孔
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

    # 显示结果
    cv2.imshow('MediaPipe Face Mesh', image)

    if cv2.waitKey(15) & 0xFF == 27:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
