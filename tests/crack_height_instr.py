import mcstasscript as ms 

instr = ms.McStas_instr("crack_height_instr")

file = ms.McStas_file("../tests/crack_height.instr")

file.add_to_instr(instr)
