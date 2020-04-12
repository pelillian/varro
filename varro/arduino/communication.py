import numpy as np
import serial
from time import sleep
from dowel import logger

from varro.util.variables import ARDUINO_PORT, SLEEP_TIME


def initialize_connection(baud_rate=14400):
    ard = serial.Serial(ARDUINO_PORT,baud_rate,timeout=5)
    sleep(2) # wait for the Arduino to initialize
    ard.flush()

    return ard

arduino = initialize_connection() # This has to be here

def send_datum(datum): 
    send(str(datum))
    sleep(SLEEP_TIME)
    return_value = receive()
    return_value = return_value.decode("utf-8")
    return return_value

def evaluate_int_int(datum): 
    return int(send_datum(datum))

def evaluate_bool_bool(datum):
    return_value = send_datum(datum)
    if return_value[-1] == ';':
        return_value = return_value[:-1]
    return_value = return_value.split(";")[-1]
    return_value = return_value.split(",")
    return_value = list(map(int, return_value))
    pred = np.mean(return_value) / 1024
    return pred

def evaluate_arduino(datum, send_type=int, return_type=int):
    if send_type is int and return_type is int: 
        return_val = evaluate_int_int(datum)
    elif send_type is bool and return_type is bool: 
        return_val = evaluate_bool_bool(datum)
    else:
        raise NotImplementedError
    if not isinstance(return_val, return_type):
        logger.log(return_val)
        import pdb; pdb.set_trace()
    return return_val

def send_char(arduino, val):
    send_byte = int(val).to_bytes(1, byteorder="big") # Makes sure that the value can be represented as one byte
    retval = arduino.write(send_byte)
    arduino.flush()
    return retval

def send(val):
    for c in val:
        send_char(arduino, c)
    arduino.write(str.encode('\n'))
    arduino.flush()

def receive():
    msg = arduino.read(arduino.in_waiting)
    return msg

if __name__=="__main__":
    val = 1

    while True:
        # Serial write section
        arduino.flush()
        send(arduino, 0)
        sleep(0.96)
        msg = receive(arduino)
        # Serial read section
        print (msg.decode("utf-8"))

        val = 0 if val else 1
