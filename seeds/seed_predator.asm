; seed_predator: drifts randomly with a high PREDATION trait and bites whatever it touches.
; Exists to prove the predation mechanics (EAT, bite, engulf, DAMAGE sensing); not optimal.
LABEL 0
  SET r0 1.0
  TRAIT PREDATION r0
  TRAIT MOTOR r0
  SET r0 0.2
  TRAIT PHOTO r0
LABEL 1
  SET r1 1.0
  EAT r1
  RAND r3            ; random walk
  SET r4 0.5
  SUB r3 r3 r4
  SET r4 3.0
  MUL r3 r3 r4
  TURN r3
  SET r5 0.3
  THRUST r5
  SENSE r1 ENERGY_FRAC
  SET r2 0.8
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
