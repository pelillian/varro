import serial
import time

#The following line is for serial over GPIO
port = '/dev/ttyACM0'


def initialize_connection():

    ard = serial.Serial(port,9600,timeout=5)
    time.sleep(2) # wait for Arduino to initialize
    ard.flush()

    return ard

def send_char(arduino, val):

    send_byte = int(val).to_bytes(1, byteorder="little") # Makes sure that the value can be represented as one byte
    #send_byte = int(val)

    print('Sending {}'.format(send_byte))

    retval = arduino.write(send_byte)
    arduino.flush()

    return retval

def send_and_recieve(arduino, val, wait_time):

    print("Python message: {}".format(val))

    send_char(arduino, val)

    time.sleep(wait_time)

    msg = arduino.read(arduino.inWaiting()) # read all characters in buffer

    return msg



ard = initialize_connection()

i = 0

zero = True

while (i < 4):
    # Serial write section

    ard.flush()

    if zero:
        val = 0
    else:
        val = 1

    # Serial read section
    msg = send_and_recieve(ard, val, 2)
    print ("Message from arduino: ")
    print (msg.decode("utf-8"))
    
    i = i + 1
    zero = False if zero else True
else:
    print("Exiting")
exit()