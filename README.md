# Visualize your McStas Union Sample environment with unviz!

unviz converts your McStas union sample into a CAD model.

It provides two different ways to do this. 

The first is to use the mcstas-to-cad script like:

`python3 mcstas_to_cad.py --input_file my_mcstas_instr.py --export --out_file {YOUR_FILE_NAME}`

The second is to use the union_viewer.py script to launch an interactive interface with live updates of your chosen mcstas instrument:

`union_viewer.py`

And see your union environment become a CAD model!

#### The rest of the instrument

With "Show McStas components" ticked (the default), union_viewer.py also draws every other component (sources, guides, slits, monitors, ...) the way McStas's own `MCDISPLAY` draws it. To get that drawing it compiles and runs the instrument with `mcrun --trace`, so it needs a working McStas install and compiler (launch it from the activated `unviz` environment). Like a normal `mcrun`, this leaves `<instrument>.c` and `<instrument>.out` next to the instrument file.

The Instrument Parameters panel (a scrollable list, since an instrument can have many) has one field per instrument parameter, showing its default. Values you type there are used both for the Union geometry and for the McStas run; parameters without a default must be filled in there.

The clipping plane can be placed in any component's coordinate system (e.g. the sample's Arm) with the Clipping panel's "Coordinate system" choice. "Export STL..." writes the visible Union meshes and McStas components as one file, cut by the clipping plane; McStas lines are exported as thin tubes.

#### Neutron rays

Tick "Show neutron rays" (off by default) to also trace a number of neutrons (50 by default, set in the Neutron Rays panel) with the same `mcrun --trace` run and draw their paths through the instrument and your Union sample. Rays can be coloured by speed, weight or time (a colourbar appears under "Colour by" whenever a mode other than Uniform is chosen), limited to those reaching a chosen component, and marked where they scatter or are absorbed (each with a colour swatch next to its checkbox). A monitor with `restore_neutron=1` makes McStas report the neutron's state jumping back to where it entered that component; that jump is real, and is drawn too, but in light grey so it reads as a jump rather than a normal continuation of the path. "Show rays" and "Show teleports" (each with its own colour swatch, like the scattering/absorption markers) toggle the ordinary path and these jumps independently. Trace mode is single-threaded and verbose, so keep the ray count modest for interactive use even though the spinner itself allows up to 1e8.


---
### Dependencies:

- pyqt (For the interface of union_viewer)

- mcstas (To load in the instrument, and get the mcstasscript parser)

- trimesh (For loading and outputting meshes)

- scikit-image (For applying the marching cubes algorithm)

- pygfx (For the renderer inside union_viewer)

- pythonocc-core (For the boundary representation math exposed by the Open Cascade Kernel)

- shapely and mapbox_earcut (Optional: close the cut faces when exporting a clipped instrument; without them the cut is left open)


### Installation

In order to use this package, you must use the conda package manager.

First clone this repository:

`git clone https://github.com/Lomholy/Union_visualizer`

Then you make a union visualizer environment by using the two commands below:

`conda env create -f unviz_env.yml`

`conda activate unviz`

### Author

The union visualizer software was authored by
Daniel Lomholt Christensen as part of his Phd. project at the Niels Bohr Institute at the University of Copenhagen


