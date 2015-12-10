__author__ = 'GatorSquad'

import Tkinter

diskUI= Tkinter.Tk()
diskUI.mainloop()

def runAnalysis():
    print("Hello")

startButton= Tkinter.Button(diskUI,text="Start",command=runAnalysis)