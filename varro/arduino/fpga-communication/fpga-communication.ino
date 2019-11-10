const int OUTPUT_PIN = 13;
const int INPUT_ADC_ADJACENT = A1; 
const int INPUT_ADC_READ = A0; 

void setup()
{

  Serial.begin(250000);  // initialize serial communications at 9600 bps
  pinMode(OUTPUT_PIN, OUTPUT);
  pinMode(INPUT_ADC_READ, INPUT); 

}

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
    digitalWrite(OUTPUT_PIN, HIGH);   // turn the LED on (HIGH is the voltage level)
    Serial.println("LED set to high");

  } else if(c == 1){
    digitalWrite(OUTPUT_PIN, LOW);   // turn the LED on (HIGH is the voltage level)
    Serial.println("LED set to low");

  } else {
    Serial.print("Could not process input: ");
    Serial.println((int)c);

  }

  int val = analogRead(INPUT_ADC_READ); 
  Serial.print("Analog value read from adjacent pin: ");  
  Serial.println('%d', val); 


  Serial.flush();

}
