import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
import PIL.Image
import PIL.ImageTk

import detect_landmarks


class App(ttk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master.title("Keypoint Charades")
        self.pack()
        self.body = detect_landmarks.BodyLandmarks()
        self.num_of_keypoints = 33
        self.keypoint_labels = [
            "顔",
            "左目(内側)",
            "L目",
            "左目(外側)",
            "右目(内側)",
            "R目",
            "右目(外側)",
            "左耳",
            "右耳",
            "L口",
            "R口",
            "左肩",
            "右肩",
            "左肘",
            "右肘",
            "左手",
            "右手",
            "左小指",
            "右小指",
            "左人差し指",
            "右人差し指",
            "左親指",
            "右親指",
            "左腰",
            "右腰",
            "左膝",
            "右膝",
            "左足首",
            "右足首",
            "左かかと",
            "右かかと",
            "左つま先",
            "右つま先",
        ]
        self.view_switches = [False] * self.num_of_keypoints
        self.show_frame_flg = True
        self.show_bones_flg = True
        self.create_widgets()

    def create_widgets(self):
        self.cap = cv2.VideoCapture(0)
        _, frame = self.cap.read()
        frame_ratio = frame.shape[1] / frame.shape[0]
        self.frame_width = 800
        self.frame_height = int(self.frame_width / frame_ratio)
        self.canvas = tk.Canvas(self, width=self.frame_width, height=self.frame_height)
        self.canvas.pack(side=tk.LEFT)

        contorl_frame = ttk.Frame(self)
        contorl_frame.pack(padx=10, pady=10, side=tk.RIGHT)

        checkbtn_frame = ttk.Frame(contorl_frame)
        checkbtn_frame.pack()
        left_frame = ttk.Frame(checkbtn_frame)
        left_frame.pack(padx=5, side=tk.LEFT)
        right_frame = ttk.Frame(checkbtn_frame)
        right_frame.pack(padx=5, side=tk.RIGHT)

        button_frame = ttk.Frame(contorl_frame)
        button_frame.pack(pady=10)

        self.checkbtns = []
        for i in range(self.num_of_keypoints):
            if self.keypoint_labels[i] in [
                "鼻",
                "左耳",
                "右耳",
                "L目",
                "R目",
                "左目(内側)",
                "右目(内側)",
                "左目(外側)",
                "右目(外側)",
                "L口",
                "R口",
            ]:
                self.checkbtns.append(None)
                continue
            if "指" in self.keypoint_labels[i]:
                self.checkbtns.append(None)
                continue
            if "左" in self.keypoint_labels[i] or "顔" in self.keypoint_labels[i]:
                checkbtn = ttk.Checkbutton(
                    left_frame,
                    text=self.keypoint_labels[i],
                    command=lambda: self.checkbtn_callback(),
                    onvalue=True,
                    offvalue=False,
                )
            elif "右" in self.keypoint_labels[i]:
                checkbtn = ttk.Checkbutton(
                    right_frame,
                    text=self.keypoint_labels[i],
                    command=lambda: self.checkbtn_callback(),
                    onvalue=True,
                    offvalue=False,
                )
            else:
                raise ValueError(
                    f"Invalid keypoint label{i}: {self.keypoint_labels[i]}"
                )
            checkbtn.state(["!alternate"])

            checkbtn.pack(anchor=tk.W)
            self.checkbtns.append(checkbtn)

        self.show_frame_btn = ttk.Button(button_frame, text="Hide Frame")
        self.show_frame_btn.pack()
        self.show_frame_btn["command"] = self.show_frame_btn_callback

        self.show_bones_btn = ttk.Button(button_frame, text="Hide Bones")
        self.show_bones_btn.pack()
        self.show_bones_btn["command"] = self.show_bones_btn_callback

        self.update()

    def update(self):
        ret, frame = self.cap.read()
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        detected_data = self.body.detect(frame)

        if self.show_frame_flg:
            annotated_frame = frame.copy()
        else:
            annotated_frame = np.zeros_like(frame)

        if detected_data:
            if self.show_bones_flg:
                self.body.mp_drawing.draw_landmarks(
                    annotated_frame,
                    detected_data,
                    self.body.mp_pose.POSE_CONNECTIONS,
                )
            else:
                self.body.draw(annotated_frame, detected_data, self.view_switches)
        else:
            annotated_frame = frame

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(
                image=PIL.Image.fromarray(
                    cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                )
            )
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.after(10, self.update)

    def checkbtn_callback(self):
        for i, checkbtn in enumerate(self.checkbtns):
            if checkbtn is not None:
                self.view_switches[i] = checkbtn.instate(["selected"])
            else:
                self.view_switches[i] = False

    def show_frame_btn_callback(self):
        self.show_frame_flg = not self.show_frame_flg
        if self.show_frame_flg:
            self.show_frame_btn["text"] = "Hide Frame"
        else:
            self.show_frame_btn["text"] = "Show Frame"

    def show_bones_btn_callback(self):
        self.show_bones_flg = not self.show_bones_flg
        if self.show_bones_flg:
            self.show_bones_btn["text"] = "Hide Bones"
        else:
            self.show_bones_btn["text"] = "Show Bones"


def main():
    root = tk.Tk()
    app = App(master=root)
    app.mainloop()


if __name__ == "__main__":
    main()
