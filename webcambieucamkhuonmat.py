import cv2
import numpy as np
import time
from tkinter import *
from keras.preprocessing import image
from keras.models import load_model
from PIL import Image,ImageTk
from datetime import datetime

#Load file trong so mode.h5 train tu Google Colab
camxuc_model = load_model('modelNhandangBieucam.h5')
camxuc_labels={0:'buon ',1:'ghetom',2:'ngacnhien',3:'sohai',4:'tucgian',5:'vuive'}
camxuc=""

#Khoi tao giao dien gui
tk=Tk()
tk.title("Nhận dạng cảm xúc")
tk.geometry("800x500+0+0")
tk.resizable(0,0)
tk.configure(background="white")

#Hien thi ten tieu de camxuc
lb1=Label(tk,fg="green",bg="white",font="Times 18",text="camxuc: ")
lb1.pack()
lb1.place(x=80,y=450)

#khoi tao camera bang webcam laptop
capture = cv2.VideoCapture(0)
def close_window():
    tk.destroy()
def ConvertImage(convert_img):
    image = convert_img[:,80:(80+480)]
    image = cv2.resize(image, dsize =(150,150))
    image = np.expand_dims(image, axis=0)
    return image
def Regconition(reg_img):
    #camxuc
    camxuc_predict = camxuc_model.predict(reg_img)
    camxuc_label= camxuc_labels[np.argmax(camxuc_predict)]
    return camxuc_label
while capture.isOpened():
    ret, image_ori = capture.read()
    cv2.imwrite('image_ori.jpg',image_ori)
    imagelg=Image.open('image_ori.jpg')
    imagelg=imagelg.resize((400,300),Image.ANTIALIAS)
    imagelg=ImageTk.PhotoImage(imagelg)
    lb2=Label(image=imagelg)
    lb2.image=imagelg
    lb2.pack()
    lb2.place(x=200,y=110)
    tk.update()

    image = ConvertImage(image_ori)
    camxuc = Regconition(image)
    lb3 = Label(tk, fg="blue", bg="white", font="Times 18", text=camxuc)
    lb3.pack()
    lb3.place(x=160, y=450)
    camxuc=""
    if cv2.waitKey(1) == ord('q'):
        close_window()
cv2.destroyAllWindows()
tk.mainloop()
