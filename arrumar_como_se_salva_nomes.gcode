;Layer height: 2.0
;DPI: 300.0
;Camadas: 10
;void_max: 1.0
;max_internal_walls: 1.0
;max_external_walls: 1.0
;max_external_walls: 1.0
;path_radius: 16
;n_max: 4.0
G91
M400
M42 P4 S255; turn off welder
M400
G1 F360; speed g1
M400; Inicio; Layer1
G1 F360; speed g1
G1 Z2.0 ; POS INICIAL
G4 P30000
M400; Inicio; Layer2
G1 F360; speed g1
G1 Z2.0 ; POS INICIAL
G4 P30000
M400; Inicio; Layer3
G1 F360; speed g1
G1 Z2.0 ; POS INICIAL
G4 P30000
M400; Inicio; Layer4
G1 F360; speed g1
G1 Z2.0 ; POS INICIAL
G4 P30000
M400; Inicio; Layer5
G1 F360; speed g1
G1 Z2.0 ; POS INICIAL
G4 P30000
M400; Inicio; Layer6
G1 F360; speed g1
G1 Z2.0 ; POS INICIAL
G4 P30000
M400; Inicio; Layer7
G1 F360; speed g1
G1 Z2.0 ; POS INICIAL
G4 P30000
M400; Inicio; Layer8
G1 F360; speed g1
G1 Z2.0 ; POS INICIAL
G4 P30000
M400; Inicio; Layer9
G1 F360; speed g1
G1 Z2.0 ; POS INICIAL
G4 P30000
M400; Inicio; Layer10
G1 F360; speed g1
G1 Z2.0 ; POS INICIAL
G4 P30000
G1 Z4.0 ; POS FINAL
M104 S0; End of Gcode
