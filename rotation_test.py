import serial

arduino=serial.Serial('/dev/ttyACM0',9600)

angle_input="90"

while True:
    angle_input=input("Write angle: ")
    if angle_input=="exit":
        break
    arduino.write(angle_input)
