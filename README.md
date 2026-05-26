# Visualize your McStas Union Sample environment with unviz!

unviz converts your McStas union sample into a CAD model.

Simply use the mcstas-to-cad.py script like:
`mcstas-to-cad.py --input_file=my_mcstas_instr.py --out=my_mcstas`


And see your union environment become a CAD model!
---

### cmdline options:
plot_point_cloud | plots a point cloud over all the geometries
dont_save_vacuum | A flag that can be set, in order to not save the geometries made from the vacuum material.

---
### Dependencies:
numpy (For general math)
trimesh (For loading and outputting meshes)
plotly.graph_objects (For debugging currently)
mcstasscript (For loading the mcstas files)
scikit-image (For applying the marching cubes algorithm)
