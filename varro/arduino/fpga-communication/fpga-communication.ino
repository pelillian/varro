

const int OUTPUT_PIN = 13;
const int INPUT_PIN = 2;

void setup()
{

  Serial.begin(250000);  // initialize serial communications at 9600 bps
  pinMode(OUTPUT_PIN, OUTPUT);
  pinMode(INPUT_PIN, INPUT);

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

  int inputVal = digitalRead(INPUT_PIN);
  Serial.print("Read value: ");
  Serial.println('1' ? inputVal : '0');


  Serial.flush();

}
