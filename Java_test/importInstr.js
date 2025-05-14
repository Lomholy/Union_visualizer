import * as THREE from 'three';
import { showEditPanel } from './editPanel';



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
  const componentRegex = /COMPONENT\s+(\w+)\s*=\s*(Union_\w+)\(([^)]*)\)/g;

  let match;
  while ((match = componentRegex.exec(fileContent)) !== null) {
      const componentName = match[1];
      const componentType = match[2];
      const parametersString = match[3];

      // Check if the component type starts with 'Union_' and matches one of the specified shapes
      if (componentType.startsWith('Union_') && shapeTypes.some(shape => componentType.toLowerCase().includes(shape))) {
          const parameters = parseParameters(parametersString);
          components.push({
              name: componentName,
              type: componentType,
              parameters: parameters
          });
      }
  }

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
    wireframe: true });

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
  mesh.position.set(0, 0, 0);  // You can modify this based on your component parameters
  mesh.rotation.set(0, 0, 0);  // Similarly, you can modify rotation if needed

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