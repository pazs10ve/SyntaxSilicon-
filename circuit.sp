* Differential Amplifier

* Supply voltages
VCC 10 0 DC 5
VEE 20 0 DC -5

* Bias current source (simplified with a resistor)
Rbias 21 20 10k

* Input stage transistors
Q1 3 1 21 NPN
Q2 4 2 21 NPN

* Load resistors
RC1 5 3 1k
RC2 6 4 1k

* Output stage (single-ended output taken from collector of Q1)
Vout 5 0

* Input signals
Vin+ 1 0 DC 0 AC 1
Vin- 2 0 DC 0 AC 1 PHASE=180

* Model for NPN transistor (generic)
.MODEL NPN NPN (Bf=100 Vaf=100 Cje=1pF Cjc=0.5pF)

.OP
.AC DEC 10 1k 10Meg
.END