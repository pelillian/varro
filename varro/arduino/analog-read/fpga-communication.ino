const int OUTPUT_PIN = 13;
const int INPUT_ADC_ADJACENT = A1; 
const int INPUT_ADC_READ = A0; 

void setup()
{

  Serial.begin(9600);  // initialize serial communications at 9600 bps
  pinMode(OUTPUT_PIN, OUTPUT);
  pinMode(INPUT_ADC_READ, INPUT); 
  // int output = 0; 
  // TODO: Config all pins that will have inputs at some point
  // TODO: Configure clock pin (digital I/O pin)
}

int output = 0; 
void loop()
{
  // TODO: increment count to correspond with pin on FPGA
  // TODO: Sample the analong input pin
  // TODO: Print information over serial
  int val = analogRead(INPUT_ADC_READ); 
  // Serial.println("Analog value read from adjacent pin: ");
  char str [3];   
  for (int i = 0; i < sizeof(str); i++) {
    str[i] = 0; 
  }
  sprintf(str, "%d", val); 
  Serial.println(str); 
  if (output == 0) {
    digitalWrite(OUTPUT_PIN, LOW); 
  } else {
    digitalWrite(OUTPUT_PIN, HIGH);
  }

  output %= 1;  

  Serial.flush();
  delay(100); 
}
