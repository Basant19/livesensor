'''Logger is used  track tge project
'''
import logging
import os
import sys
from datetime import datetime


#here log file name is created on the basis of time 
#datetime.now() is used to store the current time 
#strtime is used to convert the time in string 

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

#a folder is created to store th above file
#getcwd():to get current working together

logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)

#if above folder don't exists then it will create the folder otherwise if folder already exist then it will create the folder  
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH= os.path.join(logs_path,LOG_FILE)


#basicconfig() is used to get logging 
logging.basicConfig(
filename =LOG_FILE_PATH,
format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
level= logging.INFO,#it has 5 levels DEBUG,INFO,WARNING,ERROR,CRITICAL (DEBUG TO CRITICAL in increasing order)
)





