; seed_grazer: absorbs dissolved nutrients and swims with run-and-tumble taxis on LIGHT_DELTA
; (temporal comparison -- no direction sensing): keep going while light improves, else tumble.
LABEL 0
  SET r0 1.0
  TRAIT ABSORB r0
  TRAIT MOTOR r0
  SET r0 0.3
  TRAIT PHOTO r0
LABEL 1
  SENSE r1 LIGHT_DELTA
  SET r2 0.0
  JGT r1 r2 5        ; getting brighter -> keep heading
  RAND r3            ; else tumble
  SET r4 0.5
  SUB r3 r3 r4
  SET r4 4.0
  MUL r3 r3 r4
  TURN r3
LABEL 5
  SET r5 0.4
  THRUST r5
  SENSE r1 ENERGY_FRAC
  SET r2 0.7
  JLT r1 r2 2
  SENSE r1 MASS
  SET r2 6.0
  JLT r1 r2 2
  RAND r3
  SET r4 6.283
  MUL r3 r3 r4
  SET r4 0.5
  SPLIT r3 r4 0
LABEL 2
  YIELD
  JMP 1
