const int CLOCK_SIGNAL = 13;
const int INPUT_ADC_READ = A0; 

int* adjacentPorts = {A1, A2, A3, A4, A5};

void setup()
{

  Serial.begin(9600);  // initialize serial communications at 9600 bps
  pinMode(CLOCK_SIGNAL, OUTPUT);
  pinMode(INPUT_ADC_READ, INPUT); 
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
  char* buf[50]; 
  sprintf(buf, "Setting clock signal to %d", output); 
  Serial.println(buf); 
  digitalWrite(CLOCK_SIGNAL, output); 
  output ^= 1; 

  // increment count to correspond with pin on FPGA
  sprintf(buf, "Attempting to read pin %d", adjacentPorts[portIndex]);  
      
  // TODO: Sample the analong input pin
  // TODO: Print information over serial
  Serial.flush();
  delay(100); 
}
