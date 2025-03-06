from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt
import os.path

import NoseDetection
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
import tkinter as tk # For GUI creation
from PIL import Image, ImageTk




pathToModel = "C:\\Users\\ilmar\\OneDrive\\Python\\Nose detector\\runs\\detect\\train15\\weights\\best.torchscript"
pathToTrain = "C:\\Users\\ilmar\\OneDrive\\Python\\Nose detector\\train\\"
model = YOLO(pathToModel)

class Modal(tk.Toplevel):
    def __init__(self,master,imgCopy, bb, modal_callback):
        tk.Toplevel.__init__(self,master)

        left = bb[0].int()
        top =  bb[1].int()
        right = bb[2].int()
        bottom = bb[3].int()

        self.bb = bb
        self.imgCopy = imgCopy

        cropImage = imgCopy[top:bottom,left:right]

        self.modal_callback = modal_callback

        
        self.title("Modal")
    
        self.panel = tk.Label(self, text = "Accept image for training?")
        self.panel.pack(padx=30, pady=20)
        # cv2.imshow(cropImage)



        self.current_image = Image.fromarray(cv2.cvtColor
                    (cv2.resize(cropImage, [320, 200]), cv2.COLOR_BGR2RGB))
        self.imgtk = ImageTk.PhotoImage(image=self.current_image)

        self.image_label = tk.Label(self, image=self.imgtk)
        self.image_label.pack()

        button1 = tk.Button(self, text="Process", command=self.YOLOsave)
        button1.pack(side="left", expand = True)

        button2 = tk.Button(self, text="Discard", command=self.discard)
        button2.pack(side="right", expand = True)

        self.close_button.pack()
        self.grab_set()
        self.overrideredirect(True)
    def YOLOsave(self):
        
        i = 1
        while True:

            fnameRaw = pathToTrain + "images\\nose_image" + str(i) 
            fname = fnameRaw + ".jpg"
            if os.path.isfile(fname):
                i += 1
            else:
                break
        cv2.imwrite(fname, self.imgCopy)
        
        fnameTXT = fnameRaw.replace("images", "labels") + ".txt"
        

        self.bb # (left, top, right, bottom)
        width = float((self.bb[2]-self.bb[0])/self.imgCopy.shape[1])
        heigth = float((self.bb[3]-self.bb[1])/self.imgCopy.shape[0])
        x_center = float((self.bb[2]+self.bb[0])/2/self.imgCopy.shape[1])
        y_center = float((self.bb[3]+self.bb[1])/2/self.imgCopy.shape[0])

        with open(fnameTXT, 'w') as file:
            # Append a new line of text
            string = "1 " + str(x_center) + " " + str(y_center) +" "+  str(width) +" "+ str(heigth)
            file.write(string)
           
        


        self.modal_callback("process")

        self.destroy()
    def discard(self):
        self.modal_callback("discard")
        self.destroy()   
class NoseDetector:
    def __init__(self):
        self.cameraInstance = None
    def capture(self):
        self.cameraInstance = cv2.VideoCapture(0)
        return self.cameraInstance
ND = NoseDetector()
cam = ND.capture()

class Application:
    def __init__(self, output_path = "./"):
        """
        Initialize application which uses  OpenCV +Tkinter. It displays
        a video stream on Tkinter windows and has interaction with the user
        """
        self.vs = ND.capture()
        self.output_path = output_path
        self.current_image = None
        self.paused = 1
        self.root = tk.Tk() # initialize root window
        self.root.title("Nose capturer")

        # self.destructor function gets fired when window is closed
        # self.root.protocol("WM_DELETE_WINDOW", self.destructor)
        # self.root.attributes("-fullscreen", True)

        # Getting size to resize to 30, button space
        self.size = (self.root.winfo_screenmmwidth, self.root.winfo_screenheight()-30)

        self.panel = tk.Label(self.root)
        self.panel.pack(fill= "both", expand = True)

        # create a button for taking current frame (nose)
        self.btn = tk.Button(self.root, text = "Nose Capture", command = self.take_snapshot)
        self.btn.pack(fill="x", expand = True)

        # Start a seld.videoloop that pools the webcam
        self.video_loop()
        self.root.mainloop()

    def video_loop(self):
        """Get frame from webcam and display in Tkinter"""
    
    
        ret, frame = self.vs.read()
        self.imgCopy = frame.copy()
        if ret:# Frame capture without issues
            results = model.predict(frame, conf= 0.35)


            for r in results:
                print(type(frame))
                self.annotator = Annotator(frame)

                boxes = r.boxes
                for box in boxes:
                    
                    c = box.cls
                    if model.names[int(c)] == "Nose":
                        self.bb = box.xyxy[0] # get box coordinates in (left, top, right, bottom) format
                        self.annotator.box_label(self.bb, model.names[int(c)])
            
            self.img = self.annotator.result()     
            self.current_image = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            self.panel.imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.config(image=self.panel.imgtk)
        
        if self.paused:
            self.root.after(10, self.video_loop)  # Schedule the next frame update
    def take_snapshot(self):
        
        self.paused = 0

        
        createModal(self.root, self.imgCopy, self.bb , self.modal_callback)

    def modal_callback(self, action):
        if action == "process":
            # Handle the "Process" action
            self.paused = 1
            print("Image processed")
        elif action == "discard":
            # Handle the "Discard" action
            print("Image discarded")
            self.paused = 1    
        self.video_loop()
        



def createModal(root,imgCopy,bb,modal_callback):
    Modal(root,imgCopy, bb,modal_callback)

asd = Application()


"""
cam.release()
cv2.destroyAllWindows()

while True:
    ret, frame = cam.read()

    # Object detection
    results = model.predict(frame, conf= 0.55)


    for r in results:
        print(type(frame))
        annotator = Annotator(frame)

        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]
            c = box.cls
            if model.names[int(c)] == "Nose":
                annotator.box_label(b, model.names[int(c)])
    
    img = annotator.result()

    img_pil = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    
    tk.Label(root, image = img_tk).pack()
    root.mainloop()
    
    # Nose detection

    #

    if cv2.waitKey(1) == ord("q"):
        break

# Craete a main window

"""
