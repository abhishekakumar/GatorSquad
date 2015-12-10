__author__ = 'GatorSquad'
from Tkinter import *
import ttk,tkMessageBox,ttkcalendar
import Tkinter as tk
import xmlrpclib,pickle
import datetime
from xmlrpclib import Binary
import traceback,socket
from PIL import ImageTk,Image
import base64


diskUIRoot= tk.Tk()
diskUIRoot.title("Disk Failure Prediction")
bkImage =Image.open('/home/user/Desktop/Cloud/Image/Many-Computer-Servers-Wallpaper.gif')
bkImage1 =Image.open('/home/user/Desktop/Cloud/Image/University_of_Florida_logo.jpg')
bkImage2 =Image.open('/home/user/Desktop/Cloud/Image/images.png')
bkgdImage = ImageTk.PhotoImage(bkImage)
imageWidth = bkgdImage.width()
imageHeight = bkgdImage.height()
diskUIRoot.geometry('%dx%d+0+0' % (1000,500))

imageLabel = Label(diskUIRoot,image=bkgdImage)
imageLabel.place(x=0, y=0, relwidth=1, relheight=1)
imageLabel.pack(fill=BOTH, expand=YES,side=TOP)
#diskServer=xmlrpclib.ServerProxy("http://8.35.197.79:12000",allow_none=True)
diskServer=xmlrpclib.ServerProxy("http://localhost:12000",allow_none=True)

modelOptionFrame = Frame(diskUIRoot)
modelOptionLabel = Label(modelOptionFrame,height=2)
modelOptionLabel["text"] = "Model Name"
modelOptionLabel.pack(side=LEFT)

modelName=""

#method to extract ModelName from configuration file
def getModelNames():
    filePointer = open("/home/user/Desktop/Cloud/ModelConfigFile.txt", "r+")
    modelList=[]
    for line in filePointer:
        modelName = line.replace(" ", "_")
        modelName = modelName.replace("\n","")
        modelList.append(modelName)
    return tuple(modelList)

def resultWindow(result,modelName):
    print "Inside the sub window"
    subWindowFrame = tk.Toplevel()
    subWindowFrame.wm_title("Result")
    if result==0:
        subWindow = tk.Label(subWindowFrame, text="The Disk "+modelName+" did not Fail on this Date")
    else:
	subWindow = tk.Label(subWindowFrame, text="The Disk "+modelName+" Failed on this Date!!")
    subWindow.pack(side=LEFT, fill="both", expand=True, padx=50, pady=25)

def errorWindow(test="No error"):
    print "Inside the error window"
    errorWindowFrame = tk.Toplevel()
    errorWindowFrame.wm_title("Result")
    errorWindow = tk.Label(errorWindowFrame, text=test)
    errorWindow.pack(side=LEFT, fill="both", expand=True, padx=50, pady=25)

def runAnalysis():
    try:
        result = {}
    	modelName = optionMenuWidget.cget("text")
    	print "value is",modelName
    	predictDate = cal.selection.date()
    	print ("Date selected:"+str(predictDate.year))
    	inputValues={'model':modelName,'year':predictDate.year,'month':predictDate.month,'day':predictDate.day,}
    	prediction = diskServer.getPrediction(Binary(pickle.dumps(inputValues)))
    	print "prediction:::",pickle.loads(prediction.data)
    	resultWindow(pickle.loads(prediction.data)['predicted'],modelName)
    	#diskUIRoot.quit()
    except xmlrpclib.Fault, errcode:
         errorMessage = str(errcode)
         if errorMessage.find("IOError"):
	     errorWindow("Invalid Input!! Please enter a valid Disk or Date")
	 else:
             errorWindow("Remote Server Fault.. Try again later")
    except xmlrpclib.ProtocolError,  errcode:
	 errorWindow("Remote Server is Busy or Down.. Try again later")
    except xmlrpclib.ResponseError,  errcode:
	 errorWindow("Invalid Response from the Remote server")
    except socket.error as err:
         errorWindow("Remote Server connection error")
    except AttributeError as err:
         if str(err).find("date"):
            errorWindow("Date Error... Please enter a Valid Date!!")
         else:
	    errorWindow(str(err))
    except Exception as err:
         errorWindow("Unknown Error:"+str(err))


cal = ttkcalendar.Calendar(diskUIRoot)
var = StringVar(modelOptionFrame)
optionTuple=getModelNames()
optionMenuWidget = apply(OptionMenu, (modelOptionFrame, var) + optionTuple)
optionMenuWidget.pack(side=TOP)


cal.pack()
modelOptionFrame.pack()

startButton= tk.Button(diskUIRoot,text="Start",command=runAnalysis)
startButton.pack()
diskUIRoot.mainloop()
