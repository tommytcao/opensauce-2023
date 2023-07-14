#include <Servo.h>

Servo servoX;
Servo servoY;

int x_angle = 90;
int y_angle = 90;
int laserPin = 7; // Set the digital pin for the laser diode
bool laserState = false;

void setup()
{
    Serial.begin(9600);
    servoX.attach(9);          // Attach servoX to pin 9
    servoY.attach(10);         // Attach servoY to pin 10
    pinMode(laserPin, OUTPUT); // Set laserPin as output

    servoX.write(x_angle);
    servoY.write(y_angle);
    digitalWrite(laserPin, LOW);
}

void loop()
{
    if (Serial.available() > 0)
    {
        handleIncomingData();
    }
}

void handleIncomingData()
{
    char cmd = Serial.read();

    if (cmd == 'X')
    {
        x_angle = Serial.parseInt();
        servoX.write(x_angle);
    }
    else if (cmd == 'Y')
    {
        y_angle = Serial.parseInt();
        servoY.write(y_angle);
    }
    else if (cmd == 'L')
    {
        toggleLaser();
    }
}

void toggleLaser()
{
    laserState = !laserState;
    digitalWrite(laserPin, laserState ? HIGH : LOW);
}
