# Author: Daniel Lomholt Christensen
# From University of Copenhagen, Niels Bohr Institute, 10/10//2025 October
from time import perf_counter

import plotly.graph_objects as go
import numpy as np
import mcstasscript as ms
import matplotlib.pyplot as plt
from mcstasscript.helper.mcstas_objects import Component
from matplotlib.widgets import Slider
import ipywidgets as widgets
from ipywidgets import interact

# ChatGPT utilities
# ---------- Rotation utils ----------
def _rot_x(a):  # radians
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, ca, -sa],
                     [0, sa,  ca]])

def _rot_y(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ ca, 0, sa],
                     [  0, 1,  0],
                     [-sa, 0, ca]])

def _rot_z(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0],
                     [sa,  ca, 0],
                     [ 0,   0, 1]])

def euler_to_matrix(angles_deg, order="xyz"):
    """
    Convert Euler angles (deg) to rotation matrix.
    Default order = 'xyz' (rotate about x, then y, then z).
    Adjust 'order' if your pipeline uses a different convention.
    """
    ax, ay, az = np.radians(angles_deg)
    rot_map = {'x': _rot_x, 'y': _rot_y, 'z': _rot_z}
    R = np.eye(3)
    for axis, angle in zip(order, [ax, ay, az]):
        R = R @  rot_map[axis](angle)
    return R

def ensure_rotation_matrix(rot):
    """
    Accepts either:
      - a 3-vector of Euler angles in degrees, or
      - a 3x3 rotation matrix
    Returns a valid 3x3 rotation matrix.
    """
    rot = np.asarray(rot)
    if rot.shape == (3, 3):
        return rot
    if rot.shape == (3,):
        return euler_to_matrix(rot, order="xyz")
    if rot.size == 0 or rot is None:
        return np.eye(3)
    raise ValueError(f"Unrecognized ROTATED_data shape: {rot.shape}")

# ---------- Passive transform ----------
def world_to_local(X, Y, Z, t, R):
    """
    Vectorized passive transform: returns (x', y', z') in component-local coords.
    p_local = R^T @ (p_world - t)
    X, Y, Z can be broadcastable arrays (same shape).
    """
    t = np.asarray(t).reshape(3)
    R = np.asarray(R).reshape(3, 3)

    # Stack, subtract translation
    P = np.stack([X - t[0], Y - t[1], Z - t[2]], axis=-1)  # shape (..., 3)
    # Apply R^T
    P_local = P @ R  # since P is row-vectors, multiplying by R (not R^T) equals R^T @ column
    return P_local[..., 0], P_local[..., 1], P_local[..., 2]





# First load in the union components
def find_union_geometries(instr: ms.McStas_instr):
    instr_geometries = {}
    for comp in instr.component_list:
        if comp.component_name in instr.geometry_types.values():
            instr_geometries[comp.name] = comp.component_name
    instr.union_geometries = instr_geometries



def calculate_absolute_pos_and_rot(instr: ms.McStas_instr, comp: Component):
    # Loop over all relations
    # TODO: Make this function to begin with.
    relation = False
    # TODO: Make position be the absolute position.
    # Current fix is to just use the given position and rotation
    # TODO: Rotation is also to absolute, not specific coordinate system


    # if comp.AT_reference != None:
    #     relation =  True
    #     relation_pipe = [comp.AT_reference]
    # print(relation_pipe)
    # while relation == True:
    #     # Get a list of relations
    #     related_comp = instr.get_component(relation_pipe[-1])
    #     if related_comp == None:
    #         relation = False
    #         break
    #     if related_comp.AT_reference == "PREVIOUS":
    #         index = 0
    #         for i in range(len(instr.component_list)):
    #             if instr.component_list[i]==related_comp:
    #                 break
    #             index += 1
    #         print(index)
    #         related_comp.AT_reference = instr.get_component(instr.component_list[index].name)
    #     print(related_comp.AT_reference)

    #     relation_pipe.append(related_comp.AT_reference)

def max_min_val_cylinder(instr: ms.McStas_instr, comp: Component):
    # max/min is the the largest dimension of the cylinder in all directions
    extreme = comp.yheight/2 if comp.yheight/2>comp.radius else comp.radius
    extreme += np.array(comp.AT_data)
    # Added to the position
    # Find max val in absolute units
    instr.max = np.where(instr.max>extreme, instr.max, extreme)
    instr.min = np.where(instr.min<-extreme, instr.min, -extreme)


def max_min_val_box(instr: ms.McStas_instr, comp: Component):
    # max/min is the the largest dimension of the box in all directions
    extreme = max([comp.yheight, comp.xwidth, comp.zdepth])/2
    extreme += np.array(comp.AT_data)
    # Added to the position
    # Find max val in absolute units
    instr.max = np.where(instr.max>extreme, instr.max, extreme)
    instr.min = np.where(instr.min<-extreme, instr.min, -extreme)

def max_min_val_sphere(instr: ms.McStas_instr, comp: Component):
    # max/min is the the largest dimension of the box in all directions
    extreme = comp.radius
    extreme += np.array(comp.AT_data)
    # Added to the position
    # Find max val in absolute units
    instr.max = np.where(instr.max>extreme, instr.max, extreme)
    instr.min = np.where(instr.min<-extreme, instr.min, -extreme)



def calculate_bounding_box(instr:ms.McStas_instr):
    # Find the size of the bounding box:
    instr.max = np.zeros(3)
    instr.min = np.zeros(3)
    for name, geometry in instr.union_geometries.items():
        # Calculate positions and rotations in absolute units
        calculate_absolute_pos_and_rot(instr, instr.get_component(name)) 
        if geometry == instr.geometry_types['cylinder']:
            max_min_val_cylinder(instr, instr.get_component(name))
        if geometry == instr.geometry_types['box']:
            max_min_val_box(instr, instr.get_component(name))
        if geometry == instr.geometry_types['sphere']:
            max_min_val_sphere(instr, instr.get_component(name))
    instr.max += instr.max*0.05
    instr.min += instr.min*0.05




def point_inside_cylinder(x, y, z, radius, yheight):
    """
    Cylinder aligned with local y-axis:
      - radial: x^2 + z^2 <= r^2
      - axial: |y| <= yheight/2
    xl, yl, zl are arrays (same shape).
    Returns boolean mask with the same shape.
    """
    radial = (x**2 + z**2) <= (radius**2)
    axial  = np.abs(y) <= (0.5 * yheight)
    return radial & axial

def point_inside_box(x, y, z, xwidth, yheight, zdepth):
    """
    Box aligned with local axes:
      - x: |x| <= xwidth/2
      - y: |y| <= yheight/2
      - z: |z| <= zdepth/2
    xl, yl, zl are arrays (same shape).
    Returns boolean mask with the same shape.
    """
    inside_x = np.abs(x) <= (0.5 * xwidth)
    inside_y = np.abs(y) <= (0.5 * yheight)
    inside_z = np.abs(z) <= (0.5 * zdepth)
    return inside_x & inside_y & inside_z



def point_inside_sphere(x, y, z, radius):
    """
    Sphere centered at origin:
      - x, y, z are arrays (same shape)
      - radius is scalar
    Returns boolean mask with the same shape.
    """
    distance_squared = x**2 + y**2 + z**2
    return distance_squared <= radius**2

def point_inside_cone(x, y, z, radius_top, radius_bottom, yheight):
    """
    Checks if a point (x, y, z) is inside a truncated cone (frustum) aligned along the y-axis.
    
    Parameters:
    - x, y, z: coordinates of the point(s), can be scalars or arrays of the same shape.
    - radius_bottom: radius at y = -yheight/2 (bottom).
    - radius_top: radius at y = +yheight/2 (top).
    - yheight: total height of the cone along the y-axis.
    
    Returns:
    - Boolean mask of the same shape as x, y, z indicating whether each point is inside the cone.
    """
    # Axial check: point must be within the height of the cone
    axial = np.abs(y) <= (0.5 * yheight)
    
    # Linear interpolation of radius at given y
    # Shift y so that y = 0 corresponds to the bottom of the cone
    y_shifted = y + (0.5 * yheight)
    radius_at_y = radius_bottom + (radius_top - radius_bottom) * (y_shifted / yheight)
    
    # Radial check: point must be within the interpolated radius
    radial = (x**2 + z**2) <= radius_at_y**2
    
    return axial & radial



def create_point_cloud(instr: ms.McStas_instr, zval, yval, res):
    # Create the point cloud for each of the objects:
    x = np.linspace(instr.min[0], instr.max[0], res[0])
    y = np.linspace(instr.min[1], instr.max[1], res[1])
    z = np.linspace(instr.min[2], instr.max[2], res[1])
    X, Y = np.meshgrid(x,y)
    X, Z = np.meshgrid(x,z)
    xy_plane = np.zeros((len(instr.component_list),res[0], res[1]), dtype=object)
    xz_plane = np.zeros((len(instr.component_list),res[0],res[1]), dtype=object)
    index = 0
    # For every component, jump the planes into their coordinate systems, and
    # check if the points are inside the shapes.
    for name, geometry in instr.union_geometries.items():
        comp:Component = instr.get_component(name)
        
        # --- Get transforms ---
        t = np.array(comp.AT_data, dtype=float).reshape(3)
        R = ensure_rotation_matrix(comp.ROTATED_data)

        # --- Build plane coordinates (world) ---

        Z_fixed = np.full_like(X, zval, dtype=float)
        xy_x, xy_y, xy_z = world_to_local(X, Y, Z_fixed, t, R)
        Y_fixed = np.full_like(X, yval, dtype=float)
        xz_x, xz_y, xz_z = world_to_local(X, Y_fixed, Z, t, R)


        if geometry == instr.geometry_types['cylinder']:
            r = comp.radius
            yh = comp.yheight
            xy_cloud = point_inside_cylinder(xy_x, xy_y, xy_z, r, yh) 
            xz_cloud = point_inside_cylinder(xz_x, xz_y, xz_z, r, yh)
        
        elif geometry == instr.geometry_types['box']:
            xw = comp.xwidth
            yh = comp.yheight
            zd = comp.zdepth
            xy_cloud = point_inside_box(xy_x, xy_y, xy_z, xw, yh, zd)
            xz_cloud = point_inside_box(xz_x, xz_y, xz_z, xw, yh, zd)
        
        elif geometry == instr.geometry_types['sphere']:
            r = comp.radius
            xy_cloud = point_inside_sphere(xy_x, xy_y, xy_z, r)
            xz_cloud = point_inside_sphere(xz_x, xz_y, xz_z, r)
        
        elif geometry == instr.geometry_types['cone']:
            r_top = comp.radius_top
            r_bot = comp.radius_bottom
            yh = comp.yheight
            xy_cloud = point_inside_cone(xy_x, xy_y, xy_z, r_top, r_bot, yh) 
            xz_cloud = point_inside_cone(xz_x, xz_y, xz_z, r_top, r_bot, yh) 

        else:
            raise ValueError('Geometry is neither box nor cylinder or sphere or cone.')
        xy_plane[index] = xy_cloud*comp.priority
        xz_plane[index] = xz_cloud*comp.priority
        index += 1
    return xy_plane, xz_plane, X, Y, Z

def map_priority_to_material(instr, xy_plane, xz_plane):
    """
    Map per-pixel priorities to material IDs.
    
    Parameters
    ----------
    instr : ms.McStas_instr
        Has instr.union_geometries and instr.get_component(name)
    xy_plane : np.ndarray
        2D array of priorities for the XY slice.
    xz_plane : np.ndarray or None
        2D array of priorities for the XZ slice (optional).

    Returns
    -------
    xy_mat : np.ndarray
        2D array of material IDs for XY.
    xz_mat : np.ndarray or None
        2D array of material IDs for XZ (if xz_plane provided).
    material_enum : dict[str, int]
        Mapping from material_string -> material_id.
    priority_to_mat : dict[int, int]
        Mapping from priority -> material_id actually used for the planes.
    """
    # Build material enum and priority→material_id map
    priority_material_map = {}
    material_enum = {}
 
    for name, geometry in instr.union_geometries.items():
        comp:Component = instr.get_component(name)
        # Check if material already has a number
        if comp.material_string not in material_enum.keys():
            if (comp.material_string=="Vacuum"):
                material_enum[comp.material_string] = 0 
            else:
                material_enum[comp.material_string] = len(material_enum.keys())+1

        priority_material_map[comp.priority] = material_enum[comp.material_string]
    xy_planed = np.max(xy_plane, axis=0)
    xz_planed = np.max(xz_plane, axis=0)
    xy_plane = np.copy(xy_planed)
    xz_plane = np.copy(xz_planed)
    
    for priority, mat_id in priority_material_map.items():
            xy_plane[xy_planed==priority] = mat_id
            xz_plane[xz_planed==priority] = mat_id
    return xy_plane, xz_plane, material_enum





def plotly_slices(xy_plane:np.ndarray, xz_plane, X, Y, Z, yval, zval, instr, res, mat_enum):
    # Surface 1: XY slice at constant z = z0 (use surfacecolor for colormap)
    # ticktext, tickvals = mat_enum.items()
    xy = go.Surface(
        x=X, y=np.full_like(X,zval), z=Y,
        surfacecolor=xy_plane,
        colorscale="RdBu",
        opacity=0.6
    )
    # Surface 2: XZ slice at constant y = y0
    xz = go.Surface(
        x=X, y=Z, z=np.full_like(X, yval),
        surfacecolor=xz_plane, showscale=False,
        colorscale="RdBu",        
        opacity=0.6
    )
    
    fig = go.FigureWidget(data=[xy, xz])
    fig.update_scenes(xaxis_title='X', yaxis_title='Z', zaxis_title='Y')
    fig.update_layout(width=800, height=600, autosize=True,
                      scene_camera=dict(eye=dict(x=2.0, y=2.0, z=0.75)))
    # Sliders
    y_slider = widgets.FloatSlider(
        value=yval,
        min=Y.min(), 
        max=Y.max(),
        step=0.00001,
        description='Y value of XZ plane:',
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='.3f',
    )

    z_slider = widgets.FloatSlider(
        value=zval,
        min=Z.min(), 
        max=Z.max(),
        step=0.00001,
        description='Z value of XY plane:',
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='.3f',
    )

    def on_change(_):
        yval = y_slider.value
        zval = z_slider.value


        xy_p, xz_p, Xn, Yn, Zn = create_point_cloud(instr, zval, yval, res)
        xy_p, xz_p, _ = map_priority_to_material(instr, xy_p, xz_p)

        with fig.batch_update():
            fig.data[0].y = np.full_like(Xn, zval)         # move XY slice
            fig.data[0].surfacecolor = xy_p              # recolor
            fig.data[1].z = np.full_like(Xn, yval)         # move XZ slice
            fig.data[1].surfacecolor = xz_p              # recolor
    y_slider.observe(on_change, names='value')
    z_slider.observe(on_change, names='value')

    display(widgets.VBox([z_slider, y_slider,fig]))
    # display(widgets.VBox([fig]))
    # return fig

def slice_union(instr: ms.McStas_instr, zval=0, yval=0, res=(100,100)):
    instr.geometry_types = {"cylinder":"Union_cylinder", "box":"Union_box", "cone":"Union_cone", "sphere":"Union_sphere"}
    find_union_geometries(instr)
    calculate_bounding_box(instr)
    xy_plane, xz_plane, X, Y, Z = create_point_cloud(instr,zval, yval, res)
    
    # TODO: masking with mask string and mask settings
    
    
    # Do priority merging:

    xy_plane, xz_plane, mat_enum = map_priority_to_material(instr, xy_plane, xz_plane)
    # print(mat_enum)
    # # Plotting:
    plotly_slices(xy_plane, xz_plane, X, Y, Z, yval, zval, instr, res, mat_enum)
    return (xy_plane, xz_plane)
    # fig.show()