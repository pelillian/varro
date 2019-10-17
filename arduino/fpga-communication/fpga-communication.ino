

void setup()
{

  Serial.begin(9600);  // initialize serial communications at 9600 bps
  pinMode(LED_BUILTIN, OUTPUT);


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
    digitalWrite(LED_BUILTIN, HIGH);   // turn the LED on (HIGH is the voltage level)
    Serial.println("LED set to high");

  } else if(c == 1){
    digitalWrite(LED_BUILTIN, LOW);   // turn the LED on (HIGH is the voltage level)
    Serial.println("LED set to low");
    
  } else {
    Serial.print("Could not process input: ");    
    Serial.println((int)c);

  }

  Serial.flush();

  delay(100);

}
