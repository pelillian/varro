import serial
import time

#The following line is for serial over GPIO
SERIAL_PORT = '/dev/ttyACM0'


def initialize_connection():
    ard = serial.Serial(SERIAL_PORT,9600,timeout=5)
    time.sleep(2) # wait for the Arduino to initialize
    ard.flush()

    return ard

def send_char(arduino, val):
    send_byte = int(val).to_bytes(1, byteorder="little") # Makes sure that the value can be represented as one byte

    retval = arduino.write(send_byte)
    arduino.flush()

    return retval

def send(arduino, val):
    send_char(arduino, val)

def receive(arduino):
    msg = arduino.read(arduino,inWaiting())
    return msg

if __name__=="__main__":
    arduino = initialize_connection()
    val = 1

    while True:
        # Serial write section
        arduino.flush()
        msg = send_and_receive(arduino, 0, 0.960)
        # Serial read section
        print (msg.decode("utf-8"))

        val = 0 if val else 1

