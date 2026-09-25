# Visualize your McStas model with unviz!

unviz both visualizes your McStas instrument (including the union environment) and allows you to convert your McStas instrument into a CAD model. 

## Reasons to use unviz:

1. The unviz GUI allows for live update of your instrument. I.e while you have unviz open and you change your mcstas instrument, the visualisation will update to reflect the changed instrument.

2. unviz performs the proper pseudo CSG operations that union does for all of its geometrical components. This allows for proper visualization of your union environments.

3. The unviz GUI allows for advanced inspection of the instrument, via toggling on and off the view of each component, and allowing for clipping planes on each coordinate system in the instrument.

### Installation

In order to use this package, you must use the conda package manager.

First clone this repository:

`git clone https://github.com/Lomholy/Union_visualizer`

Then you make a union visualizer environment by using the two commands below:

`conda env create -f unviz_env.yml`

`conda activate unviz`

The environment installs this repository in editable mode, which makes the
`unviz` command available while keeping it connected to your checkout. If the
environment already existed before the command-line entry point was added, run:

```console
python -m pip install -e .
```

## Usage

Once installed, start the interactive viewer directly from the command line:

```console
unviz
```

You can open a mcstas instrument immediately by passing it as a positional argument:

```console
unviz my_mcstas_instr.py
```

This instrument file can either be a python file (.py) or a "classic" McStas instrument.

To export the mcstas instrument without opening the interface, add `--export`:

```console
unviz my_mcstas_instr.py --export --out-file YOUR_FILE_NAME
```

And see your mcstas instrument become a CAD model! The export covers the
whole instrument, not just the Union sample environment: every other
component (sources, guides, slits, monitors, ...) is included too, drawn
the same way the GUI's "Show McStas components" option draws them, so it
needs a working McStas install and compiler.


## GUI options

### Settings

The "Group by material" button groups the Union geometries into their material allowing you to show/hide entire materials at a time.

The "Hide Vacuum" button ensures that the vacuum union geometries are not shown.

The "Force mcstas-pygen preprocessing" box forces the visualizer to use the mcstas-pygen engine to preprocess the instrument which may resolve some WARNING messages for instruments.

With "Show McStas components" ticked (the default), union_viewer.py also draws every other component (sources, guides, slits, monitors, ...) the way McStas's own `MCDISPLAY` draws it. To get that drawing it compiles and runs the instrument with `mcrun --trace`, so it needs a working McStas install and compiler (launch it from the activated `unviz` environment). Like a normal `mcrun`, this leaves `<instrument>.c` and `<instrument>.out` next to the instrument file.

Furthermore it is possible to show all the arms of the mcstas instrument using the Show Arms checkbox.

The "show neutron rays" checkbox opens the neutron rays settings, and runs the instrument to start the rays.

The Reset view button focuses the view on the union samples.

The Fit whole instrument focuses the view on the whole instrument.

The Export STL button allows you to export the current geometries (i.e clipping is taken into account and the hidden geometries are also taken into account).

### Instrument parameters

The Instrument Parameters panel (a scrollable list, since an instrument can have many) has one field per instrument parameter, showing its default. Values you type there are used both for the Union geometry and for the McStas run; parameters without a default must be filled in there. When a parameter is changed the visualized instrument should update.

### Mesher options

You can choose between using Boundary representations (brep), Marching cubes (mc), or Dual Contouring (dc). For the brep you can choose the surface deflection (The coarseness of the polygonization). For the mc and dc you can change the input resolution.

### Clipping planes

The clipping plane can be placed in any component's coordinate system (e.g. the sample's Arm) with the Clipping panel's "Coordinate system" choice. "Export STL..." writes the visible Union meshes and McStas components as one file, cut by the clipping plane; McStas lines are exported as thin tubes.

### Neutron rays

Tick "Show neutron rays" (off by default) to also trace a number of neutrons (50 by default, set in the Neutron Rays panel) with the same `mcrun --trace` run and draw their paths through the instrument and your Union sample. Rays can be coloured by speed, weight or time (a colourbar appears under "Colour by" whenever a mode other than Uniform is chosen), limited to those reaching a chosen component, and marked where they scatter or are absorbed (each with a colour swatch next to its checkbox). A monitor with `restore_neutron=1` makes McStas report the neutron's state jumping back to where it entered that component; that jump is real, and is drawn too, but in light grey so it reads as a jump rather than a normal continuation of the path. "Show rays" and "Show teleports" and scattering/absorption markers toggle the ordinary path the teleport jumps, the scattering markers, and the absorption markers independently.

### Detector counts

The detector counts panel allows you to load in a mcstas run of your instrument, and it then plots the spatial monitors on top of your instrument.

### Visible geometries

The visible geometries panel lists all the geometries and their colours. Each geometry can be toggled off and on, and all union geometries are as a default grouped together into their materials, but this can be changed by the "group material definitions" check box.

---
### Dependencies:

- pyqt (For the interface of union_viewer)

- mcstas (To load in the instrument, and get the mcstasscript parser)

- trimesh (For loading and outputting meshes)

- scikit-image (For applying the marching cubes algorithm)

- pygfx (For the renderer inside union_viewer)

- pythonocc-core (For the boundary representation math exposed by the Open Cascade Kernel)

- shapely and mapbox_earcut (Optional: close the cut faces when exporting a clipped instrument; without them the cut is left open)



### Author

The union visualizer software was authored by
Daniel Lomholt Christensen as part of his Phd. project at the Niels Bohr Institute at the University of Copenhagen

