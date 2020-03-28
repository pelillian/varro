const int CLOCK_SIGNAL = 13;
#include <math.h>
#include <string.h>
#include <stdlib.h>

int analogPorts[] = {A0, A1, A2, A3, A4, A5};
int digitalPorts[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};

/**
 * Represents int between 0 and 4095 as a binary char array of length 12
 * 
 * @param num the integer that will be converted
 * @param vals the char array that will contain a big-endian conversion of the int
 */
void integerAsCharArray(int num, char* vals) {

    num %= (int)pow(2, 12); 
    
    for (int i = 1; i <= 12; i++) {
        int powerOfTwo = pow(2, 12 - i + 1); 
        if (num > powerOfTwo) {
            num -= powerOfTwo; 
            vals[i - 1] = 1; 
        } else {
            vals[i - 1] = 0; 
        }
    }
}

/**
 * Retrieves int as text over serial and converts it to an int
 * 
 * @return the int that was received over serial
 */
char receivedChars[32]; 
bool newData = false; 

void receiveInt() {
    static byte ndx = 0; 
    char endMarker = '\n'; 
    char rc; 
    int numChars = sizeof(receivedChars);
    
    if (Serial.available() > 0) {
        rc = Serial.read(); 

        if (rc != endMarker) {
            receivedChars[ndx] = rc; 
            ndx++; 
            if (ndx >= numChars) {
                ndx = numChars - 1; 
            }
        } else {
            receivedChars[ndx] = '\0'; // terminate the string
            ndx = 0; 
            newData = true;     
        }
    } 
}

/**
 * Set digital pins based on values in char array
 * 
 * @param arr Reference to the char array to be read
 */
void setDigitalPins(char* vals) {
    for (int i = 0; i < 12; i++) {
        if (vals[i] == 1) 
           digitalWrite(digitalPorts[i], HIGH); 
        else 
           digitalWrite(digitalPorts[i], LOW); 
    }
}

/**
 * Reads analog pins from FPGA into a char array
 * 
 * @param portValues Array used to store the analog values
 */
void readAnalogPins(int* portValues) {
    for (int i = 0; i < 6; i++) {
        portValues[i] = analogRead(analogPorts[i]); 
    }
}

/**
 * Sends an integer over serial as a null-terminated series of bytes
 * 
 * @param num Number to be sent
 */
void sendInt(int num) {
   
    while (num > 0) {
        Serial.print(num % 10); 
        num /= 10;  
    }
    Serial.println(); 
    Serial.flush();   
}

char vals[12]; 
int portValues[6];

void setup()
{

    Serial.begin(9600);  // initialize serial communications at 9600 bps
    pinMode(CLOCK_SIGNAL, OUTPUT);
    // int output = 0; 
    // Config all pins that will have inputs at some point
    for (int port : analogPorts) {
        pinMode(port, INPUT);
    }
    for (int port : digitalPorts) {
        pinMode(port, OUTPUT);
    }
}

int output = 0; 
int portIndex = 0; 

int analogToInt(int* portValues) {
    int result = 0; 
    for (int i = 0; i < sizeof(portValues); i++) {
        result += portValues[i] * pow(2, i); 
    }
    
    return result; 
}

void loop()
{
    // Continue to read from serial
    receiveInt();
    
    // If the val is complete, process it
    if (newData) {
        
        newData = false; 

        int num = atoi(receivedChars); 
        integerAsCharArray(num, vals);        
        
        // Send values to FPGA
        setDigitalPins(vals); 

        // Read analog pins
        readAnalogPins(portValues);  

        // Convert analog values to denormalized int
        num = analogToInt(portValues); 

        // Send resulting int over serial
        sendInt(num); 
    }
//    // Read in data from serial buffer
//    char buf[50]; 
//    // Serial.readBytes(buf, 12);
//    bool high; 
//    int numBytes = Serial.readBytes(buf, 1); 
//    char c = buf[0]; 
//    if (c == 1) {
//        high = true; 
//    } else {
//        high = false; 
//    }
//
//    for (int port : digitalPorts) {
//        if (high) 
//            digitalWrite(port, HIGH); 
//        else 
//            digitalWrite(port, LOW); 
//    }
//
//   // delayMicroseconds(100); 
//
//    int portValues[6]; 
//    for (int i = 0; i < 6; i++) {
//        portValues[i] = analogRead(analogPorts[i]); 
//    }
//
//    char sendBuf[50]; 
//    sprintf(sendBuf, "%d,%d,%d,%d,%d,%d;", portValues[0], portValues[1], portValues[2], portValues[3], portValues[4], portValues[5]);
//    Serial.print(sendBuf);
//    Serial.flush(); 
    
}

