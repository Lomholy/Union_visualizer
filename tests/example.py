import mcstasscript as ms


instr = ms.McStas_instr("test_mcstas_to_cad", author="Daniel Lomholt Christensen",
                        origin="University of Copenhagen")


file = ms.McStas_file("./test.instr")

file.add_to_instr(instr)




