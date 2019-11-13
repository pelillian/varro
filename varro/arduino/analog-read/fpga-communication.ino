const int OUTPUT_PIN = 13;
const int INPUT_ADC_ADJACENT = A1; 
const int INPUT_ADC_READ = A0; 

void setup()
{

  Serial.begin(9600);  // initialize serial communications at 9600 bps
  pinMode(OUTPUT_PIN, OUTPUT);
  pinMode(INPUT_ADC_READ, INPUT); 
  // int output = 0; 

}

int output = 0; 
void loop()
{
  char c = ' ';
  while(!Serial.available()) {}

  if (Serial.available() > 0)
  {
    c = Serial.read();  //gets one byte from serial buffer
  }

  if(c == 0)
  {
    // digitalWrite(OUTPUT_PIN, HIGH);   // turn the LED on (HIGH is the voltage level)
    // Serial.println("LED set to high");

  } else if(c == 1){
    // digitalWrite(OUTPUT_PIN, LOW);   // turn the LED on (HIGH is the voltage level)
    // Serial.println("LED set to low");

  } else {
    // Serial.println("Could not process input: ");
    // Serial.println((int)c);

  }

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
