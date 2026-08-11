# Release 0.8.19 – Progressive LED Status

GPIO17 remains the field-test gate simulation output.

During recognition activity the LED progressively accelerates as the system approaches confirmation, using approximately 300 ms down to 60 ms half-periods over the first 1.8 seconds.

On `CONFIRMED`, the LED is driven HIGH for the configured gate pulse (1 second by default). The OCR and Plate Lock pipeline are unchanged.
