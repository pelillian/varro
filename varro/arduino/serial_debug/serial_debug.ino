const int CLOCK_SIGNAL = 13;

int analogPorts[] = {A0, A1, A2, A3, A4, A5};
int digitalPorts[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
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

void loop()
{
    // Read in data from serial buffer
    char buf[50]; 
    // Serial.readBytes(buf, 12);

    // Activate digital ports based the content of the buffer
//    for (int i = 0; i < sizeof(buf); i++) {
//        
//        // NOTE: We assume that digital pins will only be assigned to 0 or 1
//        char c = buf[i]; 
//        if (c != ',') {
//            if (c == '1') {
//                digitalWrite(digitalPorts[i], HIGH);
//            } else {
//                digitalWrite(digitalPorts[i], LOW);
//            }
//        }
//    }

    // Wait for 100us for generous propagation delay
    // delayMicroseconds(100); 

    // Read values at analog ports
    // int portValues[sizeof(analogPorts)]; 
    // for (int i = 0; i < sizeof(analogPorts); i++) {
    //    portValues[i] = analogRead(analogPorts[i]); 
    // } 

    // Assemble serial message assuming there are 6 analog pins
    // sprintf(buf, "%d,%d,%d,%d,%d,%d", portValues[0], portValues[1], portValues[2], portValues[3], portValues[4], portValues[5], portValues[6]);     

    sprintf(buf, "1,2,3,4,5");
    // Send message to serial
    Serial.println(buf); 

    // Flush serial
    Serial.flush();
}


