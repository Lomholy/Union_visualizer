
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

// function parseInstrumentFile(content) {
//   // Regex to match all COMPONENT definitions and their content
//   const componentRegex = /COMPONENT\s+([A-Za-z0-9_]+)\s*=\s*([A-Za-z0-9_]+(?:\([^\)]*\))?)(.*?)(?=COMPONENT|\bEND\b)/gs;

//   let components = [];
//   let match;

//   while ((match = componentRegex.exec(content)) !== null) {
//     const componentName = match[1];
//     const componentType = match[2];
//     const componentDetails = match[3].trim();
    
//     components.push({
//       name: componentName,
//       type: componentType,
//       details: componentDetails
//     });
//   }

//   return components;
// }
function parseInstrumentFile(fileContent) {
  const components = [];
  const shapeTypes = ['cylinder', 'cube', 'sphere', 'box'];  // The types of shapes you're interested in

  // Regex pattern to match component definitions
  const componentRegex = /COMPONENT\s+(\w+)\s*=\s*(Union_\w+)\(([^)]*)\)/g;

  let match;
  while ((match = componentRegex.exec(fileContent)) !== null) {
    console.log(match);
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
  const paramsArray = parametersString.split(','); // Split by commas

  paramsArray.forEach(param => {
      const [key, value] = param.split('=').map(s => s.trim());
      if (key && value) {
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
  // Handle adding the component to your 3D scene, or other necessary processing
  // This is just a placeholder to process the component
  console.log(`Adding component ${component.name} of type ${component.type} to the scene...`);

  // You could store it in the context.objects or add it directly to the scene depending on your requirements
  context.objects.push(component);
}
