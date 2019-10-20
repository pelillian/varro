

const int OUTPUT_PIN = LED_BUILTIN;
const int INPUT_PIN = 2;

void setup()
{

  SerialUSB.begin(9600);  // initialize serial communications at 9600 bps
  pinMode(OUTPUT_PIN, OUTPUT);
  pinMode(INPUT_PIN, INPUT);

}

void loop()
{
  char c = ' ';
  while(!SerialUSB.available()) {}

  if (SerialUSB.available() > 0)
  {
    c = SerialUSB.read();  //gets one byte from serial buffer
  }

  if(c == 0)
  {
    digitalWrite(OUTPUT_PIN, HIGH);   // turn the LED on (HIGH is the voltage level)
    SerialUSB.println("LED set to high");

  } else if(c == 1){
    digitalWrite(OUTPUT_PIN, LOW);   // turn the LED on (HIGH is the voltage level)
    SerialUSB.println("LED set to low");
    
  } else {
    SerialUSB.print("Could not process input: ");    
    SerialUSB.println((int)c);

  }

  int inputVal = digitalRead(INPUT_PIN);
  SerialUSB.print("Read value: ");
  SerialUSB.println('1' ? inputVal : '0');


  SerialUSB.flush();

  delay(100);

}
