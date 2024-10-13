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
        self.landmarks = results.pose_landmarks

    def draw(self, src_img, view_switches):
        if self.landmarks is None:
            return
        # face
        self.draw_circles(src_img, view_switches, range(0, 11))
        # righthand
        self.draw_circles(src_img, view_switches, [16, 18, 20, 22])
        # lefthand
        self.draw_circles(src_img, view_switches, [15, 17, 19, 21])
        # right foot
        self.draw_circles(src_img, view_switches, [28, 30, 32])
        # left foot
        self.draw_circles(src_img, view_switches, [27, 29, 31])
        # other single keypoints
        for i, landmark in enumerate(self.landmarks.landmark):
            if view_switches[i]:
                self.draw_circle(src_img, landmark)

    def draw_circles(self, src_img, view_switches, tar_keypoints):
        if any(view_switches[i] for i in tar_keypoints):
            for idx in tar_keypoints:
                self.draw_circle(src_img, self.landmarks.landmark[idx])

    def draw_circle(self, frame, landmark):
        h, w, c = frame.shape
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

    def draw_bone(self, frame):
        self.mp_drawing.draw_landmarks(
            frame,
            self.landmarks,
            self.mp_pose.POSE_CONNECTIONS,
        )

    def close(self):
        self.pose.close()
