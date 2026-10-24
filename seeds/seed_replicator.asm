; minimal photosynthetic replicator
LABEL 0              ; daughters start here
  SET r0 1.0
  TRAIT PHOTO r0     ; invest in photosynthesis
  SET r0 0.0
  TRAIT MOTOR r0
LABEL 1
  SENSE r1 ENERGY_FRAC
  SET r2 0.8
  JLT r1 r2 2        ; not enough energy -> skip split
  SENSE r1 MASS
  SET r2 6.0         ; min mass to split (matches min_split_mass)
  JLT r1 r2 2
  RAND r3
  SET r4 6.283
  MUL r3 r3 r4       ; random angle
  SET r4 0.5
  SPLIT r3 r4 0      ; free split, daughter starts at LABEL 0
LABEL 2
  YIELD
  JMP 1
