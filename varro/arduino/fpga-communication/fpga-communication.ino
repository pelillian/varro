const int CLOCK_SIGNAL = 13;

int analogPorts[] = {A0, A1, A2, A3, A4, A5};
int digitalPorts[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
void setup()
{

  Serial.begin(115200);  // initialize serial communications at 9600 bps
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

void loop()
{
    // Read in data from serial buffer
    char buf[50]; 
    // Serial.readBytes(buf, 12);
    bool high; 
    int numBytes = Serial.readBytes(buf, 1); 
    char c = buf[0]; 
    if (c == 1) {
        high = true; 
    } else {
        high = false; 
    }

    for (int port : digitalPorts) {
        if (high) 
            digitalWrite(port, HIGH); 
        else 
            digitalWrite(port, LOW); 
    }

   // delayMicroseconds(100); 

    int portValues[6]; 
    for (int i = 0; i < 6; i++) {
        portValues[i] = analogRead(analogPorts[i]); 
    }

    char sendBuf[50]; 
    sprintf(sendBuf, "%d,%d,%d,%d,%d,%d;", portValues[0], portValues[1], portValues[2], portValues[3], portValues[4], portValues[5]);
    Serial.print(sendBuf);
    Serial.flush(); 
    
}

