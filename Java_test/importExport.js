import * as THREE from 'three';
import { showEditPanel } from './contextMenu';
import { rotate } from 'three/tsl';



export function loadInstrumentFile(context) {
  // Set up the file input change handler
  document.getElementById('importBtn').addEventListener('click', () => {
    const fileInput = document.getElementById('importFile');
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
      params[key] = value;
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


function processComponents(components, context) {
  // You can use this function to process the parsed components
  // For example, you can add them to the scene or handle them accordingly
  components.forEach(component => {
    console.log("Found component:", component);

    // Example of processing each component (modify as per your needs)
    addComponentToScene(component, context);
  });
}

function addComponentToScene(component, context) {
  console.log(`Adding component ${component.name} of type ${component.type} to the scene...`);

  // Create the geometry based on the component type
  let geometry;
  const params = component.parameters;  // Extracted parameters

  switch (component.type) {
    case 'Union_cylinder': {
      const radius = parseFloat(params.radius) || 1;  // Default to 1 if no value is provided
      const height = parseFloat(params.height) || 1;  // Default to 1 if no value is provided
      geometry = new THREE.CylinderGeometry(radius, radius, height);
      break;
    }
    case 'Union_cone': {
      const radius_top = parseFloat(params.radius_top) || 1;  // Default to 1 if no value is provided
      const height = parseFloat(params.yheight) || 1;  // Default to 1 if no value is provided
      const radius_bottom = parseFloat(params.radius_bottom) || 1;  // Default to 1 if no value is provided
      geometry = new THREE.CylinderGeometry(radius_top, radius_bottom, height);
      break;
    }
    case 'Union_sphere': {
      const radius = parseFloat(params.radius) || 1;  // Default to 1 if no value is provided
      geometry = new THREE.SphereGeometry(radius);
      break;
    }
    case 'Union_box': {
      const width = parseFloat(params.xwidth) || 1;  // Default to 1 if no value is provided
      const height = parseFloat(params.yheight) || 1;  // Default to 1 if no value is provided
      const depth = parseFloat(params.zdepth) || 1;  // Default to 1 if no value is provided
      geometry = new THREE.BoxGeometry(width, height, depth);
      break;
    }
    default:
      console.warn(`Unknown component type: ${component.type}`);
      return;  // If the component type is unknown, don't add it to the scene
  }

  // Material logic
  const materialName = params.material || 'default';
  const materialColor = getMaterialColor(materialName); // Optional: map material name to color
  const material = new THREE.MeshBasicMaterial({ 
    color: materialColor,
    transparent: true, opacity: 0.5  });

  // Create mesh
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = component.name || `Unnamed_${Date.now()}`;

  // Store extra data
  mesh.userData = {
    type: component.type,
    materialName: materialName,
    priority: parseInt(params.priority) || 0,
    ...params  // Store all other parameters for shape editing later
  };



  // Optionally, set position, rotation, or scale for the mesh
  if (component.position) {
    mesh.position.copy(component.position);
  } else {
    mesh.position.set(0, 0, 0);
  }  if (component.rotation) {
    const rot = component.rotation;
    mesh.rotation.set(
      THREE.MathUtils.degToRad(rot.x),
      THREE.MathUtils.degToRad(rot.y),
      THREE.MathUtils.degToRad(rot.z)
    );
  } else {
    mesh.rotation.set(0, 0, 0);
  }
  // Add the mesh to the scene
  context.scene.add(mesh);  // Assuming context.scene is your THREE.js scene object

  // Store the component in the context (optional, for future reference)
  context.objects.push(mesh);
  // Add to the object selection list
  const objectSelect = document.getElementById("objectSelect");
  const option = document.createElement("option");
  option.value = mesh.name;  // Use the mesh name as the value
  option.text = `${mesh.name} (${mesh.userData.type})`;  // Display name and type in the dropdown
  objectSelect.appendChild(option);
  context.selectedObject = mesh;
  objectSelect.value = mesh.name;
  console.log(`Component ${component.name} added to the scene.`);
  showEditPanel(context);
};

function getMaterialColor(name) {
  if (name.toLowerCase() === 'vacuum') {
    return 0x000000;  // Color for vacuum (black)
  }

  // Generate a random color
  const randomColor = Math.floor(Math.random() * 16777215); // Random color in hex (0x000000 to 0xFFFFFF)
  return randomColor;
}


export function writeInstr(context){
    // Add the event listener for the download button
   
    // Prepare the content for the .instr file
    let fileContent = '';
    
    // Iterate over the objects and build the content
    context.objects.forEach(obj => {
        console.log(obj.userData);
        const parameters = buildParameters(obj);
        const position = `(${obj.position.x.toFixed(2)}, ${obj.position.y.toFixed(2)}, ${obj.position.z.toFixed(2)})`;
        const rotation = `(${obj.rotation.x.toFixed(2)}, ${obj.rotation.y.toFixed(2)}, ${obj.rotation.z.toFixed(2)})`;

        // Construct the object description
        fileContent += `COMPONENT ${obj.name} = ${obj.userData.type}(\n${parameters}\n) AT ${position} RELATIVE sample_arm\nROTATED ${rotation} RELATIVE sample_arm\n\n`;
    });

    // Create a Blob from the content
    const blob = new Blob([fileContent], { type: 'text/plain' });

    // Create an anchor element to trigger the download
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'sample.instr';  // The file extension will be .instr

    // Trigger the download by programmatically clicking the link
    link.click();

    // Clean up the object URL after downloading
    URL.revokeObjectURL(link.href);
}

// Helper function to build the parameters string based on geometry type
function buildParameters(obj) {
    let params = '';
    // Get the material and priority from userData, defaulting to "default" for material and 0 for priority
    const material = obj.userData.material || 'default';
    const priority = obj.userData.priority || 0;

    switch (obj.userData.type) {
        case 'Union_cylinder':
            params += `radius=${obj.userData.radiusTop ?? obj.geometry.parameters.radiusTop ?? obj.userData.radiusBottom ?? obj.geometry.parameters.radiusBottom ?? 1}, ` +
                      `yheight=${obj.userData.height ?? obj.geometry.parameters.height ?? 1}, ` +
                      `material=${material}, priority=${priority}`;
            break;
    
        case 'Union_box':
            params += `xwidth=${obj.userData.width ?? obj.geometry.parameters.width ?? 1}, ` +
                      `yheight=${obj.userData.height ?? obj.geometry.parameters.height ?? 1}, ` +
                      `zdepth=${obj.userData.depth ?? obj.geometry.parameters.depth ?? 1}, ` +
                      `material=${material}, priority=${priority}`;
            break;
    
        case 'Union_sphere':
            params += `radius=${obj.userData.radius ?? obj.geometry.parameters.radius ?? 1}, ` +
                      `material=${material}, priority=${priority}`;
            break;
    
        case 'Union_cone':
            params += `radius_top=${obj.userData.radiusTop ?? obj.geometry.parameters.radiusTop ?? 1}, ` +
                      `radius_bottom=${obj.userData.radiusBottom ?? obj.geometry.parameters.radiusBottom ?? 1}, ` +
                      `yheight=${obj.userData.height ?? obj.geometry.parameters.height ?? 1}, ` +
                      `material=${material}, priority=${priority}`;
            break;
    
        default:
            console.warn(`Unsupported geometry type: ${obj.type}`);
            break;
    }
    return params;
}