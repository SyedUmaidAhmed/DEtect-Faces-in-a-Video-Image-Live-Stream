import numpy as np
import tensorflow as tf
import sys
if "Tkinter" not in sys.modules:
    from tkinter import *
from tkinter import *
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText
from tkinter.filedialog import askopenfilename
import cv2
import os
import tkinter as tk
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import pickle
import time
import face_recognition
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
args = vars(ap.parse_args())



data = pickle.loads(open('C:\\Python_CV\\Python37\\FACE_RECOGNITION_PROJECTS\\In a Live VideoPicture\\face-recognition-opencv\\encodings.pickle', "rb").read())




class Test():

    def __init__(self):
        
        self.root = Tk()
        self.root.title('Security and Surveillance Database')
        self.root.geometry('1000x650+0+0')
        #self.root.attributes("-fullscreen", True)



        def Camera():
            vid = askopenfilename(initialdir="C:/",filetypes =(("Video File", "*.mp4"),("All Files","*.*")),title = "Choose a file.")
            stream = cv2.VideoCapture(vid)
            writer = None
            while True:
                (grabbed, frame) = stream.read()
                if not grabbed:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = imutils.resize(frame, width=750)
                r = frame.shape[1] / float(rgb.shape[1])

                boxes = face_recognition.face_locations(rgb,model='hog')
                encodings = face_recognition.face_encodings(rgb, boxes)
                names = []

                for encoding in encodings:
                    matches = face_recognition.compare_faces(data["encodings"],encoding)
                    name = "Unknown"
                    if True in matches:
                        matchedIndxs = [i for (i, b) in enumerate(matches) if b]
                        counts = {}
                        for i in matchedIndxs:
                            name = data["names"][i]
                            counts[name] = counts.get(name, 0) + 1

                        name = max(counts, key=counts.get)
                        
                    names.append(name)


                for ((top, right, bottom, left), name) in zip(boxes, names):
                    top = int(top * r)
                    right = int(right * r)
                    bottom = int(bottom * r)
                    left = int(left * r)

                    cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
                    y = top - 15 if top - 15 > 15 else top + 15
                    cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)

                if writer is None and args["output"] is not None:
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    writer = cv2.VideoWriter(args["output"], fourcc, 24,(frame.shape[1], frame.shape[0]), True)

                
                if writer is not None:
                    writer.write(frame)

                if args["display"] > 0:
                    cv2.imshow("Press Q to Quit Window", frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        cv2.destroyAllWindows()
                        break
                    

            stream.release()
            
            if writer is not None:
                writer.release()   

            
        def select_image():

            name = askopenfilename(initialdir="C:/",filetypes =(("Text File", "*.png"),("All Files","*.*")),title = "Choose a file.")



            image = cv2.imread(name)

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb,model='hog')

            encodings = face_recognition.face_encodings(rgb, boxes)
            names=[]
            for encoding in encodings:
                matches = face_recognition.compare_faces(data["encodings"],encoding)
                name = "Unknown"

                if True in matches:
                    matchedIdxs = [i for (i,b) in enumerate(matches) if b]
                    counts = {}

                    for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name,0) + 1

                    name =max(counts, key =counts.get)
                names.append(name)
            for ((top, right, bottom, left), name) in zip(boxes, names):
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
            
            cv2.imshow("Image",image)
            cv2.waitKey(0)               
                    

        def about():
            top = tk.Toplevel()
            top.title("Guidance For Execution")
            top.geometry("400x200+180+200")
            t_lbl = tk.Label(top, text="\n\n1. Select the Image and wait for process...")
            t_lbl.pack()


            t_lbl2 = tk.Label(top, text="\n\n2. Select the Video and wait for process...")
            t_lbl2.pack()


            t_lbl3 = tk.Label(top, text="\n\n3. For Real TIme Video Connect Webcam ...")
            t_lbl3.pack()


        def live_cam():
            vs = VideoStream(src=0).start()
            writer = None
            time.sleep(2.0)
            while True:
                frame = vs.read()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = imutils.resize(frame, width=750)
                r = frame.shape[1] / float(rgb.shape[1])
                boxes = face_recognition.face_locations(rgb,model='hog')
                encodings = face_recognition.face_encodings(rgb, boxes)
                names = []
                for encoding in encodings:
                    matches = face_recognition.compare_faces(data["encodings"],encoding)
                    name = "Unknown"
                    if True in matches:
                        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                        counts = {}
                        for i in matchedIdxs:
                            name = data["names"][i]
                            counts[name] = counts.get(name, 0) + 1
                        name = max(counts, key=counts.get)
                    names.append(name)
                for ((top, right, bottom, left), name) in zip(boxes, names):
                    top = int(top * r)
                    right = int(right * r)
                    bottom = int(bottom * r)
                    left = int(left * r)
                    cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
                    y = top - 15 if top - 15 > 15 else top + 15
                    cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
                if writer is None and args["output"] is not None:
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    writer = cv2.VideoWriter(args["output"], fourcc, 20,(frame.shape[1], frame.shape[0]), True)


                if writer is not None:
                    writer.write(frame)

                if args["display"] > 0:
                    cv2.imshow("Press Q to Quit Window", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        cv2.destroyAllWindows()
                        break
            vs.stop()
            cv2.destroyAllWindows()
            
            if writer is not None:
                writer.release()   
               


#        background_image = PhotoImage(file="bg_image.png")
#        background_label = Label(self.root, image=background_image)
#        background_label.place(x=0, y=0, relwidth=1, relheight=1)
#        Label(self.root, bg="blue4").pack()
       


        Button(self.root, text="Surveillance Database", fg="Blue", font=('Times New Roman', '40', 'bold')).pack(padx=5, pady=5)
        Button(self.root, text="Security Mechanism Systems", font=('Times New Roman', '30', 'bold')).pack(padx=5, pady=10)

        self.about = Button(self.root, text="Saved Video Execution", width="30",font=('Helvetica', '12', 'italic'), command=Camera)
        self.about.pack(padx=5, pady=15)       

        self.about = Button(self.root, text="Image Analysis", width="30", font=('Helvetica', '12', 'italic'), command=select_image)
        self.about.pack(padx=5, pady=20)

        self.about = Button(self.root, text="Live Webcam Video", width="30",font=('Helvetica', '12', 'italic'), command=live_cam)
        self.about.pack(padx=5, pady=25)

        self.about = Button(self.root, text="About Application", width="30", font=('Helvetica', '12', 'italic'),command=about)
        self.about.pack(padx=5, pady=30)

        good = Button(self.root, text="Closing the Window", width="30",font=('Helvetica', '12', 'italic'), command=self.quit)
        good.pack(padx=5, pady=35)

        self.root.mainloop()

    def quit(self):
        self.root.destroy()


app = Test()



