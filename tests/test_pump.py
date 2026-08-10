import unittest
from unittest.mock import patch, MagicMock
from src.pump_controller import PeristalticPump, PumpController, StallError

class TestPeristalticPump(unittest.TestCase):

    @patch('src.pump_controller.PWMOutputDevice')
    @patch('src.pump_controller.DigitalInputDevice')
    def test_start_stop(self, mock_digital, mock_pwm):
        pump = PeristalticPump('test', pwm_pin=18, stall_pin=23)
        pump.start(speed=0.8)
        self.assertTrue(pump.is_running)
        mock_pwm.return_value.value = 0.8
        pump.stop()
        self.assertFalse(pump.is_running)
        mock_pwm.return_value.value = 0

    @patch('src.pump_controller.PWMOutputDevice')
    @patch('src.pump_controller.DigitalInputDevice')
    def test_speed_validation(self, mock_digital, mock_pwm):
        pump = PeristalticPump('test', pwm_pin=18)
        with self.assertRaises(ValueError):
            pump.start(speed=1.5)
        with self.assertRaises(ValueError):
            pump.start(speed=-0.1)

    @patch('src.pump_controller.PWMOutputDevice')
    @patch('src.pump_controller.DigitalInputDevice')
    @patch('builtins.input', return_value='50.0')
    def test_calibration(self, mock_input, mock_digital, mock_pwm):
        pump = PeristalticPump('test', pwm_pin=18)
        factor = pump.calibrate(duration_sec=10)
        # 50 ml in 10s -> 300 ml/min at speed 1.0
        self.assertAlmostEqual(factor, 300.0, places=1)

    @patch('src.pump_controller.PWMOutputDevice')
    @patch('src.pump_controller.DigitalInputDevice')
    def test_set_flow_rate_uncalibrated(self, mock_digital, mock_pwm):
        pump = PeristalticPump('test', pwm_pin=18)
        with self.assertRaises(RuntimeError):
            pump.set_flow_rate(100)

    @patch('src.pump_controller.PWMOutputDevice')
    @patch('src.pump_controller.DigitalInputDevice')
    def test_stall_detection(self, mock_digital, mock_pwm):
        mock_digital.return_value.value = 1  # stall signal high
        pump = PeristalticPump('test', pwm_pin=18, stall_pin=23)
        with self.assertRaises(StallError):
            pump.start()
            pump._monitor_stall()  # simulate immediate check

class TestPumpController(unittest.TestCase):
    @patch('src.pump_controller.PeristalticPump')
    def test_add_and_remove_pump(self, mock_pump_class):
        controller = PumpController()
        controller.add_pump('A', 18)
        self.assertIn('A', controller.pumps)
        controller.remove_pump('A')
