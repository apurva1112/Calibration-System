import serial

def rotate(angle="90"):                                              #function to rotate the camera attached to the servo motor to the specified angle
    arduino=serial.Serial('/dev/ttyACM0',9600)
    arduino.write(angle_input)                                        #sends data to arduino through Serial
