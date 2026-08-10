import time
import RPi.GPIO as GPIO

class PumpController:
    def __init__(self, pins=[18, 19, 20, 21]):
        self.pins = pins
        GPIO.setmode(GPIO.BCM)
        for pin in pins:
            GPIO.setup(pin, GPIO.OUT)
    
    def activate_pump(self, pin, duration=5):
        print(f"🔧 Attivazione pompa su pin {pin}...")
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(pin, GPIO.LOW)
        print(f"✅ Pompa su pin {pin} disattivata.")
    
    def test_all_pumps(self):
        for pin in self.pins:
            self.activate_pump(pin, 2)
    
    def cleanup(self):
        GPIO.cleanup()

if __name__ == "__main__":
    controller = PumpController()
    try:
        controller.test_all_pumps()
    finally:
        controller.cleanup()
