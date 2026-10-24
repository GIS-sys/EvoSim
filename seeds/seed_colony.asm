; seed_colony: grows a curved chain of 6 bonded cells with constant turning per link (split angle
; 0.5 rad relative to heading). r5 is a link counter that daughters inherit (reg_inherit=copy).
; Cells differentiate through the `state` register: 0 = growing, 1 = body, 2 = tip.
LABEL 0              ; daughters start here
  SETSTATE 0
  SET r0 1.0
  TRAIT PHOTO r0
LABEL 1
  SENSE r1 ENERGY_FRAC
  SET r2 0.7
  JLT r1 r2 2        ; not enough energy
  SENSE r1 MASS
  SET r2 6.0
  JLT r1 r2 2        ; not enough mass
  SET r2 4.0
  JGT r5 r2 4        ; 5 links already -> become the tip
  SET r6 1.0
  ADD r5 r5 r6       ; count this link (daughter inherits it)
  SET r3 0.5         ; split angle relative to heading (the curvature)
  SET r4 0.5
  SPLITB r3 r4 0
  SETSTATE 1         ; parent is now a body cell
  JMP 3
LABEL 2
  YIELD
  JMP 1
LABEL 4              ; tip cell
  SETSTATE 2
  SET r0 1.0
  TRAIT PIGMENT r0
LABEL 3              ; finished: idle, share energy with the neighbours
  SET r7 0.05
  SHARE r7
  YIELD
  JMP 3
