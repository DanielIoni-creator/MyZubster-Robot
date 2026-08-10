import time
from gpiozero import PWMOutputDevice, DigitalInputDevice
from threading import Thread, Event
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class PumpError(Exception):
    """Base exception for pump errors."""
    pass

class StallError(PumpError):
    """Raised when the pump motor stalls."""
    def __init__(self, pump_id):
        self.pump_id = pump_id
        super().__init__(f"Pump {pump_id} stall detected")

class PeristalticPump:
    """Driver for a single peristaltic pump."""

    def __init__(self, pump_id: str, pwm_pin: int, stall_pin: Optional[int] = None,
                 frequency: int = 1000, stall_threshold: float = 0.5):
        self.pump_id = pump_id
        self.pwm = PWMOutputDevice(pwm_pin, frequency=frequency)
        self.stall_sensor = DigitalInputDevice(stall_pin) if stall_pin is not None else None
        self.stall_threshold = stall_threshold
        self._running = False
        self._speed = 0.0  # 0.0 to 1.0 duty cycle
        self._flow_rate_ml_per_min = 0.0
        self._calibration_factor = 1.0  # ml/min per unit speed
        self._stall_monitor_thread: Optional[Thread] = None
        self._stop_event = Event()

    def start(self, speed: float = 1.0):
        """Start the pump at the given speed (0.0 to 1.0)."""
        if not 0 <= speed <= 1:
            raise ValueError("Speed must be between 0 and 1")
        self._speed = speed
        self.pwm.value = speed
        self._running = True
        if self.stall_sensor:
            self._stop_event.clear()
            self._stall_monitor_thread = Thread(target=self._monitor_stall, daemon=True)
            self._stall_monitor_thread.start()
        logger.info(f"Pump {self.pump_id} started at speed {speed:.2f}")

    def stop(self):
        """Stop the pump."""
        self._running = False
        self._stop_event.set()
        self.pwm.value = 0
        if self._stall_monitor_thread and self._stall_monitor_thread.is_alive():
            self._stall_monitor_thread.join(timeout=1.0)
        logger.info(f"Pump {self.pump_id} stopped")

    def set_flow_rate(self, ml_per_min: float):
        """Set target flow rate in ml/min using calibration."""
        if self._calibration_factor <= 0:
            raise RuntimeError("Pump not calibrated")
        speed = ml_per_min / self._calibration_factor
        speed = max(0.0, min(1.0, speed))
        self.start(speed)

    def calibrate(self, duration_sec: float = 10.0) -> float:
        """
        Run pump at full speed for given duration and prompt user to measure actual volume.
        Returns calibration factor (ml/min per unit speed).
        """
        self.start(1.0)
        time.sleep(duration_sec)
        self.stop()
        measured_ml = float(input(f"Enter the measured volume in ml dispensed in {duration_sec}s: "))
        if measured_ml <= 0:
            raise ValueError("Measured volume must be positive")
        self._calibration_factor = (measured_ml / duration_sec) * 60.0  # ml/min at speed 1.0
        logger.info(f"Pump {self.pump_id} calibration factor: {self._calibration_factor:.02f} ml/min at full speed")
        return self._calibration_factor

    def _monitor_stall(self):
        """Monitor stall sensor and raise error if stall detected."""
        while self._running and not self._stop_event.is_set():
            if self.stall_sensor.value == 1:  # Assuming active high on stall
                self.stop()
                raise StallError(self.pump_id)
            time.sleep(0.1)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def current_speed(self) -> float:
        return self._speed

    def close(self):
        """Release hardware resources."""
        self.stop()
        self.pwm.close()
        if self.stall_sensor:
            self.stall_sensor.close()

class PumpController:
    """Manages multiple peristaltic pumps."""

    def __init__(self):
        self.pumps: Dict[str, PeristalticPump] = {}

    def add_pump(self, pump_id: str, pwm_pin: int, stall_pin: Optional[int] = None,
                 frequency: int = 1000):
        if pump_id in self.pumps:
            raise ValueError(f"Pump {pump_id} already exists")
        self.pumps[pump_id] = PeristalticPump(pump_id, pwm_pin, stall_pin, frequency)

    def remove_pump(self, pump_id: str):
        if pump_id in self.pumps:
            self.pumps[pump_id].close()
            del self.pumps[pump_id]

    def start_pump(self, pump_id: str, speed: float = 1.0):
        self.pumps[pump_id].start(speed)

    def stop_pump(self, pump_id: str):
        self.pumps[pump_id].stop()

    def stop_all(self):
        for pump in self.pumps.values():
            pump.stop()

    def close(self):
        self.stop_all()
