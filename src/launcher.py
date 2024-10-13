import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
import PIL.Image
import PIL.ImageTk

import detect_landmarks
import keypoint_names as kn


class App(ttk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master.title("Keypoint Charades")
        self.pack()
        self.body = detect_landmarks.BodyLandmarks()
        self.view_switches = [False] * kn.num_of_keypoints
        self.show_frame_flg = True
        self.show_bones_flg = True
        self.create_widgets()

    def create_widgets(self):
        self.cap = cv2.VideoCapture(0)
        _, frame = self.cap.read()
        frame_ratio = frame.shape[1] / frame.shape[0]
        self.frame_width = 1200
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
        for i in range(kn.num_of_keypoints):
            if kn.keypoint_labels[i] in kn.no_checkbox_labels:
                self.checkbtns.append(None)
                continue
            if kn.LEFT in kn.keypoint_labels[i] or kn.FACE in kn.keypoint_labels[i]:
                checkbtn = ttk.Checkbutton(
                    left_frame,
                    text=kn.keypoint_labels[i],
                    command=lambda: self.checkbtn_callback(),
                    onvalue=True,
                    offvalue=False,
                )
            elif kn.RIGHT in kn.keypoint_labels[i]:
                checkbtn = ttk.Checkbutton(
                    right_frame,
                    text=kn.keypoint_labels[i],
                    command=lambda: self.checkbtn_callback(),
                    onvalue=True,
                    offvalue=False,
                )
            else:
                raise ValueError(f"Invalid keypoint label{i}: {kn.keypoint_labels[i]}")
            checkbtn.state(["!alternate"])

            checkbtn.pack(anchor=tk.W)
            self.checkbtns.append(checkbtn)

        self.show_frame_btn = ttk.Button(button_frame, text=kn.HIDE_FRAME, width=14)
        self.show_frame_btn.pack()
        self.show_frame_btn["command"] = self.show_frame_btn_callback

        self.show_bones_btn = ttk.Button(button_frame, text=kn.HIDE_BONE, width=14)
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
                self.body.draw_bone(annotated_frame, detected_data)
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
            self.show_frame_btn["text"] = kn.HIDE_FRAME
        else:
            self.show_frame_btn["text"] = kn.SHOW_FRAME

    def show_bones_btn_callback(self):
        self.show_bones_flg = not self.show_bones_flg
        if self.show_bones_flg:
            self.show_bones_btn["text"] = kn.HIDE_BONE
        else:
            self.show_bones_btn["text"] = kn.SHOW_BONE


def main():
    root = tk.Tk()
    app = App(master=root)
    app.mainloop()


if __name__ == "__main__":
    main()
