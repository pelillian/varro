const int CLOCK_SIGNAL = 13;

int adjacentPorts[] = {A0, A1, A2, A3, A4, A5};

void setup()
{

  Serial.begin(9600);  // initialize serial communications at 9600 bps
  pinMode(CLOCK_SIGNAL, OUTPUT);
  // int output = 0; 
  // Config all pins that will have inputs at some point
  for (int port : adjacentPorts) {
    pinMode(port, INPUT);
  }
}

int output = 0; 
int portIndex = 0; 

void loop()
{
  // Alterate the clock signal
  char buf[50]; 
  sprintf(buf, "Setting clock signal to %d", output); 
  Serial.println(buf); 
  digitalWrite(CLOCK_SIGNAL, output); 
  output ^= 1; 
  
  // print out the pin readings for each analog pin
  for (int pin : adjacentPorts) {
    int val = analogRead(pin);
    sprintf(buf, "%d: %d", pin, val);
    Serial.println(buf);
  }
  Serial.flush();
  delay(960); 
}
