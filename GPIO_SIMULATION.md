# GPIO gate simulation

Field-test output: Raspberry Pi BCM GPIO17, physical pin 11.

Connect GPIO17 through a 330 ohm resistor to the LED anode (long leg). Connect the LED cathode (short leg) to GND, physical pin 9.

The output is driven HIGH for 1.0 second only after OCR reaches CONFIRMED. A manual test endpoint is also available at `/api/gate/trigger` when the web server exposes the GPIO extension.
