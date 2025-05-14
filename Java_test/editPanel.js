import * as THREE from 'three';

import { linkSliderAndInput } from './linksAndSliders.js'; // Import if it's now separated

const shapeParameters = {
    Union_cylinder: ['radius', 'yheight'],
    Union_cube: ['xwidth', 'yheight', 'zdepth'],
    Union_sphere: ['radius'],
    Union_box: ['xwidth', 'yheight', 'zdepth']
};

  
export function showEditPanel(context) {
    const obj = context.selectedObject;
    if (!obj) return;

    document.getElementById('editPanel').style.display = 'block';

    document.getElementById('objectName').value = obj.name || '';

    document.getElementById('posX').value = obj.position.x.toFixed(2);
    document.getElementById('posY').value = obj.position.y.toFixed(2);
    document.getElementById('posZ').value = obj.position.z.toFixed(2);

    document.getElementById('posXSlider').value = obj.position.x;
    document.getElementById('posYSlider').value = obj.position.y;
    document.getElementById('posZSlider').value = obj.position.z;

    // Sync sliders and inputs live
    linkSliderAndInput(context, 'posXSlider', 'posX', val => obj.position.x = val);
    linkSliderAndInput(context, 'posYSlider', 'posY', val => obj.position.y = val);
    linkSliderAndInput(context, 'posZSlider', 'posZ', val => obj.position.z = val);

    document.getElementById('rotX').value = obj.rotation.x.toFixed(2);
    document.getElementById('rotY').value = obj.rotation.y.toFixed(2);
    document.getElementById('rotZ').value = obj.rotation.z.toFixed(2);

    document.getElementById('rotXSlider').value = obj.rotation.x;
    document.getElementById('rotYSlider').value = obj.rotation.y;
    document.getElementById('rotZSlider').value = obj.rotation.z;

    // Sync sliders and inputs for rotation
    linkSliderAndInput(context, 'rotXSlider', 'rotX', val => obj.rotation.x = val);
    linkSliderAndInput(context, 'rotYSlider', 'rotY', val => obj.rotation.y = val);
    linkSliderAndInput(context, 'rotZSlider', 'rotZ', val => obj.rotation.z = val);

    document.getElementById('editPriority').value = obj.userData.priority || 0;
    document.getElementById('materialName').value = obj.userData.materialName || '';
     // ⬇️ SHAPE PARAMETERS SECTION ⬇️
     const paramContainer = document.getElementById('shapeParams');
     paramContainer.innerHTML = ''; // Clear previous parameters
 
     // Get the geometry type (e.g., CylinderGeometry, BoxGeometry, etc.)
     const geometryType = obj.geometry?.type;
 
     // Define which parameters to show for which shapes
     const shapeParamsMap = {
         CylinderGeometry: ['radiusTop', 'radiusBottom', 'height'],
         BoxGeometry: ['width', 'height', 'depth'],
         SphereGeometry: ['radius'],
         ConeGeometry: ['radius', 'height']
     };
 
     // Get the parameters for the current shape type
     const shapeParams = shapeParamsMap[geometryType];
     if (!shapeParams) return;
 
     // Loop through each shape parameter and create the UI
     shapeParams.forEach(param => {
         const wrapper = document.createElement('div');
         wrapper.style.marginBottom = '10px';
 
         const label = document.createElement('label');
         label.textContent = param;
         label.htmlFor = `shape_${param}`;
         label.style.display = 'block';
 
         const numberInput = document.createElement('input');
         numberInput.type = 'number';
         numberInput.step = '0.01';
         numberInput.min = '0';
         numberInput.id = `shape_${param}`;
         numberInput.value = obj.userData[param] ?? obj.geometry.parameters[param] ?? 0;
 
         const sliderInput = document.createElement('input');
         sliderInput.type = 'range';
         sliderInput.min = '0';
         sliderInput.max = '10';  // Adjust the max as needed
         sliderInput.step = '0.01';
         sliderInput.value = numberInput.value;
 
         // Sync number input with slider
         numberInput.addEventListener('input', () => {
             sliderInput.value = numberInput.value;
             if (isValidValue(numberInput.value)) {
                 obj.userData[param] = parseFloat(numberInput.value);
                 updateGeometry(obj);
             } else {
                 numberInput.style.backgroundColor = 'red'; // Indicate invalid input
             }
         });
 
         // Sync slider with number input
         sliderInput.addEventListener('input', () => {
             numberInput.value = sliderInput.value;
             if (isValidValue(sliderInput.value)) {
                 obj.userData[param] = parseFloat(sliderInput.value);
                 updateGeometry(obj);
             } else {
                 sliderInput.style.backgroundColor = 'red'; // Indicate invalid input
             }
         });
 
         // Append the label, number input, and slider to the wrapper
         wrapper.appendChild(label);
         wrapper.appendChild(numberInput);
         wrapper.appendChild(sliderInput);
         paramContainer.appendChild(wrapper);
     });
}


function updateGeometry(object) {
    const type = object.geometry?.type;
    const params = object.userData;

    let newGeometry;
    switch (type) {
        case 'CylinderGeometry':
            newGeometry = new THREE.CylinderGeometry(
                params.radiusTop ?? 1,
                params.radiusBottom ?? 1,
                params.height ?? 1,
                32 // segments
            );
            break;
        case 'BoxGeometry':
            newGeometry = new THREE.BoxGeometry(
                params.width ?? 1,
                params.height ?? 1,
                params.depth ?? 1
            );
            break;
        case 'SphereGeometry':
            newGeometry = new THREE.SphereGeometry(
                params.radius ?? 1,
                32, 32
            );
            break;
        case 'ConeGeometry':
            newGeometry = new THREE.ConeGeometry(
                params.radius ?? 1,
                params.height ?? 1,
                32
            );
            break;
        default:
            return;
    }

    object.geometry.dispose(); // Clean up old geometry
    object.geometry = newGeometry;
}


function isValidValue(value) {
    // Check if the value is a valid positive number and not NaN
    return !isNaN(value) && value > 0;
}
