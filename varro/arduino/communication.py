import serial
import time
from varro.misc.variables import ARDUINO_PORT


def initialize_connection():
    ard = serial.Serial(ARDUINO_PORT,9600,timeout=5)
    time.sleep(2) # wait for the Arduino to initialize
    ard.flush()

    return ard

def send_char(arduino, val):
    send_byte = int(val).to_bytes(1, byteorder="little") # Makes sure that the value can be represented as one byte

    retval = arduino.write(send_byte)
    arduino.flush()

    return retval

def send(arduino, val):
    for c in val:
        import pdb; pdb.set_trace()
        send_char(arduino, c)

def receive(arduino):
    msg = arduino.read(arduino.inWaiting())
    return msg

if __name__=="__main__":
    arduino = initialize_connection()
    val = 1

    while True:
        # Serial write section
        arduino.flush()
        send(arduino, 0)
        time.sleep(0.96)
        msg = receive(arduino)
        # Serial read section
        print (msg.decode("utf-8"))

        val = 0 if val else 1

