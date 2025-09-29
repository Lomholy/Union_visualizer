/*
 * Function to import an instrument file into Union Visualizer.
 * Only relies on three and spawnObject to work.
*/
import * as THREE from 'three';
import {spawnObject } from '../utils/ObjectHandling.js'


export function loadInstrumentFile(context) {
  // Set up the file input change handler
  document.getElementById('importFile').addEventListener('change', () => {
    const fileInput = document.getElementById('importFile');
    if (fileInput.files.length>1){
      alert("Please select only 1 file.")
      return;
    }
    const file = fileInput.files[0]; // Get the first file

    if (!file) {
      alert("Please select a file.");
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target.result;
      const components = parseInstrumentFile(content);
      processComponents(components, context); // Process the components and integrate them into the context
    };

    reader.readAsText(file); // Read the file as text
    // reset input so same file can be chosen again
    fileInput.value = '';
  });
}

function parseInstrumentFile(fileContent) {
  const components = [];
  const shapeTypes = ['cylinder', 'cone', 'sphere', 'box'];  // The types of shapes you're interested in

  // Regex pattern to match component definitions
  const componentRegex = /COMPONENT\s+(\w+)\s*=\s*(Union_\w+)\(([^)]*)\)\s+AT\s+\(([^)]+)\)(?:\s+RELATIVE\s+\w+)?(?:\s+ROTATED\s+\(([^)]+)\)(?:\s+RELATIVE\s+\w+)?)?/g;
  let match;
  while ((match = componentRegex.exec(fileContent)) !== null) {
      const componentName = match[1];
      const componentType = match[2];
      if (componentType.startsWith('Union_') && shapeTypes.some(shape => componentType.toLowerCase().includes(shape))) {
        const parametersString = match[3];
        const positionString = match[4];
        const rotationString = match[5];
        
        const parameters = parseParameters(parametersString);
        const position = parseVector(positionString);
        const rotation = parseVector(rotationString);
      
      // Check if the component type starts with 'Union_' and matches one of the specified shapes
        
        components.push({
          name: componentName,
          type: componentType,
          parameters,
          position,
          rotation
        });
      }
     
    };
  return components;
}

function parseParameters(parametersString) {
  const params = {};
  const paramsArray = parametersString.split(',');

  paramsArray.forEach(param => {
    const [key, value] = param.split('=').map(s => s.trim());
    if (key && value !== undefined) {
      params[key] = parseFloat(value);
      if (key=="material_string"){
        params[key] = value;
      }
    }
  });

  return params;
}

function parseVector(vectorString) {
  if (!vectorString) return new THREE.Vector3(0, 0, 0);

  const parts = vectorString.split(',').map(s => parseFloat(s.trim()));
  const [x, y, z] = parts;

  return new THREE.Vector3(
    isNaN(x) ? 0 : x,
    isNaN(y) ? 0 : y,
    isNaN(z) ? 0 : z
  );
}



// =========================================================================
// Add component to scene 
// =========================================================================
// importExport.js

/* ... keep loadInstrumentFile, parseInstrumentFile, parseParameters, parseVector ... */

// Replace processComponents to use spawnObject
function processComponents(components, context) {
  components.forEach(component => {
    const spec = componentToSpec(component); // map parser output -> spawnObject spec
    spawnObject(context, spec);
  });
}

// Map parsed component -> spec for spawnObject()
function componentToSpec(component) {
  const p = component.parameters || {};
  const type = component.type;

  // Normalize geometry params for the four supported types
  let params = {};
  switch (type) {
    case 'Union_cylinder':
      params = {
        radius: finite(p.radius, p.radius_top, p.radius_bottom, 0.01),
        yheight: finite(p.height, p.yheight, 0.01)
      };
      break;
    case 'Union_cone':
      params = {
        radiusTop:    finite(p.radius_top, undefined, 0),
        radiusBottom: finite(p.radius_bottom, p.radius, 0.01),
        yheight:       finite(p.height, p.yheight, 0.01)
      };
      break;
    case 'Union_sphere':
      params = { radius: finite(p.radius, undefined, 0.01) };
      break;
    case 'Union_box':
      params = {
        xwidth:  finite(p.width,  p.xwidth,  0.01),
        yheight: finite(p.height, p.yheight, 0.01),
        zdepth:  finite(p.depth,  p.zdepth,  0.01)
      };
      break;
    default:
      console.warn(`Unknown component type: ${type}`);
      break;
  }

  // Material & priority (same keys your exporter uses)
  const materialName = (p.material_string ?? 'default');
  const priority = parseInt(p.priority) || 0;

  return {
    type,
    name: component.name,
    materialName,
    priority,
    params,
    position: component.position,       // THREE.Vector3 from parseVector
    rotationDegrees: component.rotation // parseVector returns Vector3 (degrees)
  };

  function finite(...vals) {
    for (const v of vals) if (Number.isFinite(v)) return v;
    return vals[vals.length - 1]; // last is default
  }
}
