/*
 * This code measures 25Hz data with 8G from the Sparkfun Accelerometer
 * and logs the data to a micro SD in the Sparkfun OpenLog
 * It is based on the Library for the MMA8452Q 
 * and the Library for the SparkFun Qwiic OpenLog
 * Modified by Patrick Mayerhofer July 2020
 */

/*
  Library for the MMA8452Q
  By: Jim Lindblom and Andrea DeVore
  SparkFun Electronics

  Do you like this library? Help support SparkFun. Buy a board!
  https://www.sparkfun.com/products/14587

  This sketch uses the SparkFun_MMA8452Q library to initialize
  the accelerometer and stream raw x, y, z, acceleration
  values from it.

  Hardware hookup:
  Arduino --------------- MMA8452Q Breakout
    3.3V  ---------------     3.3V
    GND   ---------------     GND
  SDA (A4) --\/330 Ohm\/--    SDA
  SCL (A5) --\/330 Ohm\/--    SCL

  The MMA8452Q is a 3.3V max sensor, so you'll need to do some
  level-shifting between the Arduino and the breakout. Series
  resistors on the SDA and SCL lines should do the trick.

  License: This code is public domain, but if you see me
  (or any other SparkFun employee) at the local, and you've
  found our code helpful, please buy us a round (Beerware
  license).

  Distributed as is; no warrenty given.
*/

#define sf 25 //change this for wanted sampling fq                    
#define tc (1000/(sf))     // time constant  
#include <Wire.h>                 // Must include Wire library for I2C
#include "SparkFun_MMA8452Q.h"    // Click here to get the library: http://librarymanager/All#SparkFun_MMA8452Q
#include "SparkFun_Qwiic_OpenLog_Arduino_Library.h"
double x;
double y;
double z;
unsigned long last_time = 0;

OpenLog myLog; //Create instance
const byte OpenLogAddress = 42; //Default Qwiic OpenLog I2C address
char filename[] = "ACCdata.txt";

MMA8452Q accel;                   // create instance of the MMA8452 class


void setup() {
  Wire.begin();
  myLog.begin(); //Open connection to OpenLog (no pun intended)
  delay(500);
  myLog.append(filename);
  
  

  if (accel.begin() == false) {
    myLog.println("Not Connected. Please check connections and read the hookup guide.");
    while (1);
  }

  accel.setDataRate(ODR_800);
  accel.setScale(SCALE_8G);
}

void loop() {
  
  
  if (accel.available()) {      // Wait for new data from accelerometer
   
    // Raw of acceleration of x, y, and z directions
    x = accel.getX();
    y = accel.getY();
    z = accel.getZ();
  }
  if (millis() >= last_time + tc) {
    last_time = millis();
    myLog.append(filename);
    myLog.print(x/256);
    myLog.print("\t");
    myLog.print(y/256);
    myLog.print("\t");
    myLog.print(z/256);
    myLog.print("\t");
    myLog.print(last_time);
    myLog.println();
}
}
