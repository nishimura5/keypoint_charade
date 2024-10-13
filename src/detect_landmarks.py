# detect bodies landmarks by mediapipe

import cv2
import mediapipe as mp


class BodyLandmarks:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False,
            model_complexity=1,
        )

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        if results.pose_landmarks:
            return results.pose_landmarks
        return None

    def draw(self, frame, landmarks, view_switches):
        # face
        if view_switches[0]:
            for i in range(0, 11):
                self.draw_circle(frame, landmarks.landmark[i])
        # righthand
        right_hand_kp_idx_list = [16, 18, 20, 22]
        if (
            view_switches[16]
            or view_switches[18]
            or view_switches[20]
            or view_switches[22]
        ):
            for idx in right_hand_kp_idx_list:
                self.draw_circle(frame, landmarks.landmark[idx])
        # lefthand
        left_hand_kp_idx_list = [15, 17, 19, 21]
        if (
            view_switches[15]
            or view_switches[17]
            or view_switches[19]
            or view_switches[21]
        ):
            for idx in left_hand_kp_idx_list:
                self.draw_circle(frame, landmarks.landmark[idx])
        # right foot
        right_foot_kp_idx_list = [28, 30, 32]
        if view_switches[28] or view_switches[30] or view_switches[32]:
            for idx in right_foot_kp_idx_list:
                self.draw_circle(frame, landmarks.landmark[idx])

        # left foot
        left_foot_kp_idx_list = [27, 29, 31]
        if view_switches[27] or view_switches[29] or view_switches[31]:
            for idx in left_foot_kp_idx_list:
                self.draw_circle(frame, landmarks.landmark[idx])

        for i, landmark in enumerate(landmarks.landmark):
            if view_switches[i]:
                self.draw_circle(frame, landmark)

    def draw_circle(self, frame, landmark):
        h, w, c = frame.shape
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

    def draw_bone(self, frame, landmark):
        self.mp_drawing.draw_landmarks(
            frame,
            landmark,
            self.mp_pose.POSE_CONNECTIONS,
        )

    def close(self):
        self.pose.close()
