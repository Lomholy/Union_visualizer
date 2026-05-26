import mcstasscript as ms



def add_2D_space_logger(instrument, ref, extra_name=None,
                        dim1="z", D1_min=-0.1, D1_max=0.1, n1=300,
                        dim2="x", D2_min=-0.1, D2_max=0.1, n2=300,
                        order=None):
    
    if extra_name is None:
        extra_name = ""
    else:
        extra_name = str(extra_name) + "_"
    
    name = f"logger_2D_space_{extra_name}{dim1}{dim2}_"
    if order is None:
        name += "all"
    else:
        name += str(order)
    
    log = instrument.add_component(name, "Union_logger_2D_space")
    log.set_AT([0,0,0], RELATIVE=ref)
    log.set_parameters(D_direction_1='"' + dim1 + '"', D1_min=D1_min, D1_max=D1_max, n1=n1,
                       D_direction_2='"' + dim2 + '"', D2_min=D2_min, D2_max=D2_max, n2=n2,
                       filename='"' + name + '.dat"')
    if order is not None:
        log.set_parameters(order_total=order)


def add_2D_space_loggers(instrument, ref, zx=True, zy=True, xy=True, order=None,
                         xmin=-0.1, xmax=0.1, ymin=-0.1, ymax=0.1, zmin=-0.1, zmax=0.1,
                         nx=300, ny=300, nz=300, extra_name=None):
    
    if not isinstance(order, list):
        order = [order]
    
    # Have all zx monitors in a row for neater visualization
    for this_order in order:    
        if zx:
            add_2D_space_logger(instrument=instrument, ref=ref, order=this_order,
                                dim1="z", D1_min=zmin, D1_max=zmax, n1=nz,
                                dim2="x", D2_min=xmin, D2_max=xmax, n2=nx,
                                extra_name=extra_name)
    
    # Then zy together
    for this_order in order:   
        if zy:
            add_2D_space_logger(instrument=instrument, ref=ref, order=this_order,
                                dim1="z", D1_min=zmin, D1_max=zmax, n1=nz,
                                dim2="y", D2_min=ymin, D2_max=ymax, n2=ny,
                                extra_name=extra_name)

    # Then xy together            
    for this_order in order:            
        if xy:
            add_2D_space_logger(instrument=instrument, ref=ref, order=this_order,
                                dim1="x", D1_min=xmin, D1_max=xmax, n1=nx,
                                dim2="y", D2_min=ymin, D2_max=ymax, n2=ny,
                                extra_name=extra_name)                                
    
def add_2D_Q_logger(instrument, ref, extra_name=None,
                        dim1="z", D1_min=-5, D1_max=5, n1=300,
                        dim2="x", D2_min=-5, D2_max=5, n2=300,
                        order=None):
        
    if extra_name is None:
        extra_name = ""
    else:
        extra_name = str(extra_name) + "_"
    
    name = f"logger_2DQ_{extra_name}{dim1}{dim2}_"
    if order is None:
        name += "all"
    else:
        name += str(order)
    
    log = instrument.add_component(name, "Union_logger_2DQ")
    log.set_AT([0,0,0], RELATIVE=ref)
    log.set_parameters(Q_direction_1='"' + dim1 + '"', Q1_min=D1_min, Q1_max=D1_max, n1=n1,
                       Q_direction_2='"' + dim2 + '"', Q2_min=D2_min, Q2_max=D2_max, n2=n2,
                       filename='"' + name + '.dat"')
    if order is not None:
        log.set_parameters(order_total=order)


def add_2D_Q_loggers(instrument, ref, zx=True, zy=True, xy=True, order=None,
                         xmin=-0.1, xmax=0.1, ymin=-0.1, ymax=0.1, zmin=-0.1, zmax=0.1,
                         nx=300, ny=300, nz=300, max_E_var=None, extra_name=None):
    
    if max_E_var is not None:
        # q = 2k sin(theta)
        try:
            instrument.add_declare_var("double", "q_min")
            instrument.add_declare_var("double", "q_max")
        except:
            # Already defined
            pass
        
        instrument.append_initialize(f"q_max = 1.05*2*V2K*SE2V*sqrt({max_E_var});")
        instrument.append_initialize(f"q_min = -q_max;")
        
        xmin = ymin = zmin = "q_min"
        xmax = ymax = zmax = "q_max"
        
    if not isinstance(order, list):
        order = [order]
    
    # Have all zx monitors in a row for neater visualization
    for this_order in order:    
        if zx:
            add_2D_Q_logger(instrument=instrument, ref=ref, order=this_order,
                                dim1="z", D1_min=zmin, D1_max=zmax, n1=nz,
                                dim2="x", D2_min=xmin, D2_max=xmax, n2=nx,
                                extra_name=extra_name)                            
    
    # Then zy together
    for this_order in order:   
        if zy:
            add_2D_Q_logger(instrument=instrument, ref=ref, order=this_order,
                                dim1="z", D1_min=zmin, D1_max=zmax, n1=nz,
                                dim2="y", D2_min=ymin, D2_max=ymax, n2=ny,
                                extra_name=extra_name)                            

    # Then xy together            
    for this_order in order:            
        if xy:
            add_2D_Q_logger(instrument=instrument, ref=ref, order=this_order,
                                dim1="x", D1_min=xmin, D1_max=xmax, n1=nx,
                                dim2="y", D2_min=ymin, D2_max=ymax, n2=ny,
                                extra_name=extra_name)          
            

time_loggers = False
high_resolution = True

instrument = ms.McStas_instr("cryostat_test")

progress = instrument.add_component("progress", "Progress_bar")

source = instrument.add_component("source", "Source_simple")
source.xwidth = 0.01
source.yheight = 0.01
source.focus_xw = 0.01
source.focus_yh = 0.01
source.dist = 2
source.E0 = instrument.add_parameter("E0", value=5)
source.dE = instrument.add_parameter("dE", value=0.1)

init = instrument.add_component("init", "Union_init")

orange_cryostat = ms.Cryostat("orange", instrument)
orange_cryostat.set_AT([0, 0, 2], source)

orange_cryostat.add_layer(inner_radius=70E-3/2, outer_radius=75.45E-3/2,
                          origin_to_bottom=82.57E-3, bottom_thickness=5.53E-3,
                          #sample_to_top=198.9E-3, top_thickness=5.63E-3,
                          origin_to_top=198.9E-3, top_thickness=-1E-3,
                          material="Al", p_interact=0.2)
orange_cryostat.last_layer.add_window(outer_radius=73E-3/2, origin_to_top=44.42E-3, origin_to_bottom=88.2E-3)

orange_cryostat.add_layer(inner_radius=80E-3/2, outer_radius=81E-3/2,
                          origin_to_bottom=89.8E-3, bottom_thickness=1.65E-3,
                          origin_to_top=237.62E-3, top_thickness=8.19E-3, p_interact=0.2)

orange_cryostat.add_layer(inner_radius=95E-3/2, outer_radius=99.5E-3/2,
                          origin_to_bottom=92.76E-3, bottom_thickness=5.86E-3,
                          origin_to_top=223.65E-3, top_thickness=8.64E-3, p_interact=0.2)
orange_cryostat.last_layer.add_window(outer_radius=97E-3/2, origin_to_top=51.79E-3, origin_to_bottom=98.63E-3)

orange_cryostat.add_layer(inner_radius=120E-3/2, outer_radius=126.63E-3/2,
                          origin_to_bottom=108.57E-3, bottom_thickness=10.82E-3,
                          origin_to_top=200.4E-3, top_thickness=21.74E-3, p_interact=0.2)
orange_cryostat.last_layer.add_window(outer_radius=123E-3/2, origin_to_top=55.7E-3, origin_to_bottom=93.54E-3)

#orange_cryostat.add_spatial_loggers()
orange_cryostat.add_time_histogram(t_min=0.0018, t_max=0.004)
orange_cryostat.add_spatial_loggers(n_x=500, n_y=500, n_z=500)
orange_cryostat.build(include_master=False)

if high_resolution:
    add_2D_space_loggers(instrument=instrument, ref="orange", order=[None, 1, 2, 3, 4, 5, 6],
                         xmin=-0.07, xmax=0.07, zmin=-0.07, zmax=0.07, ymin=-0.13, ymax=0.13,
                         nx=1000, ny=1000, nz=1000)

    add_2D_Q_loggers(instrument=instrument, ref="orange", order=[None, 1, 2, 3, 4, 5, 6], max_E_var="E0+dE",
                     nx=1000, ny=1000, nz=1000)
else:
    add_2D_space_loggers(instrument=instrument, ref="orange", order=[None, 1, 2, 3, 4, 5, 6],
                         xmin=-0.07, xmax=0.07, zmin=-0.07, zmax=0.07, ymin=-0.13, ymax=0.13)

    add_2D_Q_loggers(instrument=instrument, ref="orange", order=[None, 1, 2, 3, 4, 5, 6], max_E_var="E0+dE")


# conditional
add_2D_space_loggers(instrument=instrument, ref="orange", order=[None, 1, 2, 3, 4, 5, 6], extra_name="con",
                     xmin=-0.07, xmax=0.07, zmin=-0.07, zmax=0.07, ymin=-0.13, ymax=0.13)

add_2D_Q_loggers(instrument=instrument, ref="orange", order=[None, 1, 2, 3, 4, 5, 6],
                 extra_name="con", max_E_var="E0+dE")

    
# Set up YBaCuO with incoherent and single crystal
YBaCuO_incoherent = instrument.add_component("YBaCuO_incoherent", "Incoherent_process")
YBaCuO_incoherent.sigma = 2.105
YBaCuO_incoherent.unit_cell_volume = 173.28

YBaCuO_crystal = instrument.add_component("YBaCuO_crystal", "Single_crystal_process")
YBaCuO_crystal.set_parameters(
{"ax" : 3.816, "ay" : 0, "az" : 0,
 "bx" : 0, "by" : 3.886, "bz" : 0,
 "cx" : 0, "cy" : 0, "cz" : 11.677,
 "delta_d_d" : 5E-4, "mosaic" : 30, "barns" : 1,
 "reflections" : '"YBaCuO.lau"'})

YBaCuO = instrument.add_component("YBaCuO", "Union_make_material")
YBaCuO.process_string = '"YBaCuO_incoherent,YBaCuO_crystal"'
YBaCuO.my_absorption = 100*14.82/173.28

A3_angle = instrument.add_parameter("A3_angle", value=0)
phi = instrument.add_parameter("phi", value=0)

sample_position_y_rotated = instrument.add_component("sample_position_y_rotated", "Arm")
sample_position_y_rotated.set_AT(0, "orange")
sample_position_y_rotated.set_ROTATED([0, A3_angle, 0], RELATIVE="orange")

sample_position = instrument.add_component("sample_position", "Arm")
sample_position.set_AT(0, sample_position_y_rotated)
sample_position.set_ROTATED([phi, 0, 0], RELATIVE=sample_position_y_rotated)

sample = instrument.add_component("sample", "Union_box", RELATIVE=sample_position)
# first version
#sample.xwidth = 0.015
#sample.yheight = 0.032
#sample.zdepth = 0.012

# Version sent to Kim
#sample.set_parameters(xwidth = 0.0075, yheight = 0.016, zdepth = 0.006)
# Kim wanted to reduce sample size with about 30%
sample.set_parameters(xwidth=6E-3, yheight = 12E-3, zdepth = 5E-3)




sample.material_string = '"YBaCuO"'
sample.priority = 200
#sample.set_ROTATED([phi, 0, 0], RELATIVE=sample_position)


# sample_holder
vertical = instrument.add_component("vertical", "Union_box")
vertical.set_parameters(zdepth=7E-3, xwidth=2.5E-3, yheight=sample.yheight, priority=300, material_string='"Al"')
#vertical.set_AT([0, 2E-3, 0.5*sample.zdepth + 0.5*vertical.zdepth + 1E-4], sample_position_y_rotated)
vertical.set_AT([0.5*sample.xwidth + 0.5*vertical.xwidth + 1E-4, 6.5E-3, 0], sample_position_y_rotated)

top = instrument.add_component("top", "Union_cylinder")
top.set_parameters(radius=0.6*sample.xwidth, yheight=2E-3, priority=290, material_string='"Al"')
top.set_AT([0, 0.5*sample.yheight + 0.5*top.yheight + 5E-3, 0], sample_position_y_rotated)

sample_rod = instrument.add_component("sample_rod", "Union_cylinder")
sample_rod.set_parameters(radius=6E-3, yheight=1.2, priority=310, material_string='"Al"')
sample_rod.set_AT([0, 0.5*sample_rod.yheight + 0.49*top.yheight, 0], top)


instrument.add_declare_var("double", "max_speed")
instrument.add_declare_var("double", "min_speed")
instrument.append_initialize("max_speed = sqrt(E0 + dE)*SE2V;")
instrument.append_initialize("min_speed = sqrt(E0 - dE)*SE2V;")

instrument.add_declare_var("double", "tmin_cryo")
instrument.add_declare_var("double", "tmax_cryo")
instrument.append_initialize("tmin_cryo = (2.0-0.15)/max_speed;")
instrument.append_initialize("tmax_cryo = (2.0+0.45)/min_speed;")

if time_loggers:
    log_2D_st = instrument.add_component("logger_2D_space_time_zx", "Union_logger_2D_space_time")
    log_2D_st.set_AT([0,0,0], RELATIVE="orange")
    log_2D_st.time_bins = 100
    log_2D_st.time_min = "tmin_cryo"
    log_2D_st.time_max = "tmax_cryo"
    log_2D_st.D_direction_1 = '"z"'
    log_2D_st.D1_min = -0.07
    log_2D_st.D1_max = 0.07
    log_2D_st.n1 = 300
    log_2D_st.D_direction_2 = '"x"'
    log_2D_st.D2_min = -0.07
    log_2D_st.D2_max = 0.07
    log_2D_st.n2 = 300
    log_2D_st.filename = '"logger_2D_space_time_zx.dat"'

    log_2D_st = instrument.add_component("logger_2D_space_time_zy", "Union_logger_2D_space_time")
    log_2D_st.set_AT([0,0,0], RELATIVE="orange")
    log_2D_st.time_bins = 100
    log_2D_st.time_min = "tmin_cryo"
    log_2D_st.time_max = "tmax_cryo"
    log_2D_st.D_direction_1 = '"z"'
    log_2D_st.D1_min = -0.07
    log_2D_st.D1_max = 0.07
    log_2D_st.n1 = 300
    log_2D_st.D_direction_2 = '"y"'
    log_2D_st.D2_min = -0.07
    log_2D_st.D2_max = 0.07
    log_2D_st.n2 = 300
    log_2D_st.filename = '"logger_2D_space_time_zy.dat"'

    log_2D_st = instrument.add_component("logger_2D_space_time_zy_sample", "Union_logger_2D_space_time")
    log_2D_st.set_AT([0,0,0], RELATIVE="orange")
    log_2D_st.time_bins = 100
    log_2D_st.time_min = "tmin_cryo"
    log_2D_st.time_max = "tmax_cryo"
    log_2D_st.D_direction_1 = '"z"'
    log_2D_st.D1_min = -0.006
    log_2D_st.D1_max = 0.006
    log_2D_st.n1 = 300
    log_2D_st.D_direction_2 = '"y"'
    log_2D_st.D2_min = -0.0075
    log_2D_st.D2_max = 0.0075
    log_2D_st.n2 = 300
    log_2D_st.filename = '"logger_2D_space_time_zy_sample.dat"'
    
    
# Set up instrument parameters describing what spot to investigate
instrument.add_parameter("tag_angle", value=-95)
instrument.add_parameter("tag_time", value=0.00188)
instrument.add_parameter("tag_interval", value=9E-5)

# Set up an arm pointing to the relevant spot
spot_dir = instrument.add_component("spot_dir", "Arm",
                                    RELATIVE="orange")
spot_dir.set_ROTATED([0, "tag_angle", 0], RELATIVE="orange")

# Set up a conditional component targeting all our loggers
PSD_conditional = instrument.add_component("space_all_PSD_conditional", "Union_conditional_PSD")

target_loggers = []
for comp in instrument.component_list:
    if "_con_" in comp.name:
        target_loggers.append(comp.name)

PSD_conditional.target_loggers = '"' + ",".join(target_loggers) + '"'
PSD_conditional.xwidth = instrument.add_parameter("double", "conditional_xwidth", value=0.05)
PSD_conditional.yheight = 0.2
PSD_conditional.time_min = "tag_time-0.5*tag_interval"
PSD_conditional.time_max = "tag_time+0.5*tag_interval"
# Ensure the position of the conditional rectangle is on the detector surface
PSD_conditional.set_AT([0, 0, 0.5], RELATIVE=spot_dir) 


master = instrument.add_component("master", "Union_master")
master.verbal = 0

# Add a monitor with flag that is only active when the condition in the conditional is true
instrument.add_declare_var("int", "flag1")
instrument.add_declare_var("int", "n_scattering")
logger_con = instrument.get_component("logger_2D_space_con_zx_all")
logger_con.logger_conditional_extend_index = 1
master.append_EXTEND("flag1 = logger_conditional_extend_array[1];")
master.append_EXTEND("n_scattering = number_of_scattering_events;")

stop = instrument.add_component("stop", "Union_stop")

banana_detector = instrument.add_component("banana_detector", "Monitor_nD")
banana_detector.set_RELATIVE("orange")
banana_detector.radius = 0.5
banana_detector.yheight = 0.2
banana_detector.restore_neutron = 1

instrument.add_declare_var("double", "tmin")
instrument.add_declare_var("double", "tmax")
instrument.append_initialize("tmin = (2.5-0.15)/max_speed;")
instrument.append_initialize("tmax = (2.5+0.65)/min_speed;")

#instrument.append_initialize('printf("tmin = %lf tmax= %lf \\n", tmin, tmax);')

instrument.add_declare_var("char", "option_string", array=1024)
if high_resolution:
    instrument.append_initialize('sprintf(option_string, "banana, theta limits=[-180,180] bins=721, t limits=[%lf %lf] bins=1000", tmin, tmax);')
else:
    instrument.append_initialize('sprintf(option_string, "banana, theta limits=[-180,180] bins=361, t limits=[%lf %lf] bins=500", tmin, tmax);')

options = '"banana, theta limits=[-180,180] bins=361, t limits=[0.0 0.0025] bins=500"'
banana_detector.options = "option_string"
banana_detector.filename = '"tof_b.dat"'

# Copy of our banana detector, but with WHEN condition to verify we are investigating the right peak
for order in range(1,7):
    banana_detector = instrument.copy_component("banana_detector_" + str(order), "banana_detector")
    banana_detector.filename = f'"tof_b_{order}.dat"'
    banana_detector.set_WHEN(f"n_scattering == {order}")


# Copy of our banana detector, but with WHEN condition to verify we are investigating the right peak
banana_detector = instrument.add_component("banana_detector_limited", "Monitor_nD")
banana_detector.set_RELATIVE("orange")
banana_detector.radius = 0.5
banana_detector.yheight = 0.2
banana_detector.restore_neutron = 1
banana_detector.options = "option_string"
banana_detector.filename = '"tof_b_limited.dat"'
banana_detector.set_WHEN("flag1 > 0")

emon = instrument.add_component("Emon", "E_monitor", AT=0.5, RELATIVE="orange")
emon.set_parameters(Emin="E0 - dE", Emax="E0 + dE", xwidth=0.04, yheight=0.04, nE=300, restore_neutron=1)