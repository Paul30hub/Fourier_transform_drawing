from PIL import Image, ImageTk
import numpy as np
import tkinter

var = np.linspace(0,100)
top = tkinter.Tk()
top.geometry('400x400')
#scale = tkinter.Scale(top, variable = var, activebackground = 'blue')
#label = tkinter.Label(top, text = 'Hello world', )
sizeH = 400
sizeW = 400
C = tkinter.Canvas(top, bg = "white", height = sizeH, width = sizeW)
x = sizeH/2
y = sizeW/2
r = 50
x0 = x - r
y0 = y - r
x1 = x + r
y1 = y + r
abscisse = C.create_line(0,y,sizeH,y)
ordonnee = C.create_line(x,0,x,sizeW)
circle1 = C.create_oval(x0, y0, x1, y1)
C.pack(pady = 20)
#label.pack()
#scale.pack()
top.mainloop()
