# Visualize your McStas Union Sample environment with unviz!

unviz converts your McStas union sample into a CAD model.

It provides two different ways to do this. 

The first is to use the mcstas-to-cad script like:

`python3 mcstas_to_cad.py --input_file my_mcstas_instr.py --export --out_file {YOUR_FILE_NAME}`

The second is to use the union_viewer.py script to launch an interactive interface with live updates of your chosen mcstas instrument:

`union_viewer.py`

And see your union environment become a CAD model!

#### The rest of the instrument

With "Show McStas components" ticked (the default), union_viewer.py also draws every other component (sources, guides, slits, monitors, ...) the way McStas's own `MCDISPLAY` draws it. To get that drawing it compiles and runs the instrument with `mcrun --trace`, so it needs a working McStas install and compiler (launch it from the activated `unviz` environment). Like a normal `mcrun`, this leaves `<instrument>.c` and `<instrument>.out` next to the instrument file. Parameters without a default can be given in the "Instrument parameters" field as `name=value` pairs.


---
### Dependencies:

- pyqt (For the interface of union_viewer)

- mcstas (To load in the instrument, and get the mcstasscript parser)

- trimesh (For loading and outputting meshes)

- scikit-image (For applying the marching cubes algorithm)

- pygfx (For the renderer inside union_viewer)

- pythonocc-core (For the boundary representation math exposed by the Open Cascade Kernel)


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


