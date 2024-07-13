from sensor.exception import SensorException
import os
import sys
from sensor.logger import logging

def test_exception():
    try:
        logging.info("yaha par error hai bro") #in main.py info should be in logging 
        a=1/0
    except Exception as e: 
        raise  SensorException (e,sys)


if __name__=="__main__":#module exception control to control the flow of execution 
    try:
        test_exception()
    except Exception as e:
        print (e)