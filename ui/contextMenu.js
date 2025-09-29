// This file handles most of the info displayed in the context menu of the
// GUI


import * as THREE from 'three';
import {setupLiveUpdateHandlers} from '../utils/objectHandling.js';



// The function to update the list of objects
export function updateObjectList(context) {
  if (!context || !Array.isArray(context.objects)) {
    console.warn("updateObjectList: Invalid context or missing objects array");
    return;
  }
  //console.log("objectList element:", document.getElementById("objectList"));
  
  const list = document.getElementById("objectList");
  list.innerHTML = ""; // Clear the list
  
  context.objects.forEach(obj => {
    const li = document.createElement("li");

    const {r,g,b} = obj.material.color;
    li.style.color = `rgb(${Math.round(r*255)}, ${Math.round(g*255)}, ${Math.round(b*255)})`; // Convert rgb values to 0-255 range
    li.textContent = `${obj.name} (Priority: ${obj.userData.priority}, Material: ${obj.userData.materialName})`;
    
    // Add the data-obj-name attribute
    li.dataset.objName = obj.name;

    // Optional: make items clickable to select
    li.style.cursor = "pointer";

    li.addEventListener("click", () => {
        console.log("Clicked object:", obj);
        context.selectedObject = obj;  // Update selectedObject in context if you store it there
        document.getElementById("objectSelect").value = obj.name;
        showEditPanel(context);  // Show the edit panel for the selected object

         // Highlight selected item
        [...list.children].forEach(item => item.classList.remove("selected"));
        li.classList.add("selected");
    });

    // Apply highlight if this object is already selected
    if (context.selectedObject === obj) {
      li.classList.add("selected");
    }
    
    list.appendChild(li);
  });
}

export function updateObjectOrder(context) {
  // Sort objects by priority (ascending, lowest priority first)
  context.objects.sort((a, b) => a.userData.priority - b.userData.priority);

  // Update the renderOrder based on priority
  context.objects.forEach((obj, index) => {
    obj.renderOrder =context.objects.length -index;  // Objects with higher priority will be rendered first
  });
}


export function linkSliderAndInput(context, sliderId, inputId, onChange) {
    const { sliderListeners, inputListeners, objects } = context;

    const slider = document.getElementById(sliderId);
    const input = document.getElementById(inputId);

    if (sliderListeners[sliderId]) slider.removeEventListener('input', sliderListeners[sliderId]);
    if (inputListeners[inputId]) input.removeEventListener('input', inputListeners[inputId]);

    const sliderHandler = () => {
        input.value = slider.value;
        onChange(parseFloat(slider.value));
    };

    const inputHandler = () => {
        slider.value = input.value;
        onChange(parseFloat(input.value));
    };

    sliderListeners[sliderId] = sliderHandler;
    inputListeners[inputId] = inputHandler;

    slider.addEventListener('input', sliderHandler);
    input.addEventListener('input', inputHandler);
}


  
export function showEditPanel(context) {
    const obj = context.selectedObject;
    if (!obj) return;

    document.getElementById('editPanel').style.display = 'block';

    document.getElementById('objectName').value = obj.name || '';
    document.getElementById('materialName').value = obj.userData.materialName;

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

 
    document.getElementById('rotX').value = (obj.rotation.x * 180 / Math.PI).toFixed(2);
    document.getElementById('rotY').value = (obj.rotation.y * 180 / Math.PI).toFixed(2);
    document.getElementById('rotZ').value = (obj.rotation.z * 180 / Math.PI).toFixed(2);
    
    document.getElementById('rotXSlider').value = obj.rotation.x * 180 / Math.PI;
    document.getElementById('rotYSlider').value = obj.rotation.y * 180 / Math.PI;
    document.getElementById('rotZSlider').value = obj.rotation.z * 180 / Math.PI;
    // Sync sliders and inputs for rotation
    linkSliderAndInput(context, 'rotXSlider', 'rotX', val => obj.rotation.x = val * Math.PI / 180);
    linkSliderAndInput(context, 'rotYSlider', 'rotY', val => obj.rotation.y = val * Math.PI / 180);
    linkSliderAndInput(context, 'rotZSlider', 'rotZ', val => obj.rotation.z = val * Math.PI / 180);
    
    document.getElementById('editPriority').value = obj.userData.priority || 0;

    // ⬇️ SHAPE PARAMETERS SECTION ⬇️
    const paramContainer = document.getElementById('shapeParams');
    console.log(obj.userData.type);
    paramContainer.innerHTML = ''; // Clear previous parameters

    // Define which parameters to show for which shapes
    const shapeParamsMap = {
        'Union_cylinder': ['radius', 'height'],
        'Union_box': ['width', 'height', 'depth'],
        'Union_sphere': ['radius'],
        'Union_cone': ['radius_top', 'radius_bottom', 'height']
    };

    // Get the parameters for the current shape type
    const shapeParams = shapeParamsMap[obj.userData.type];
    if (!shapeParams){ 
        console.log('Error: ShapeParams not existing');
        return;
    }
    console.log(obj)
    // Loop through each shape parameter and create the UI
    shapeParams.forEach(param => {
        const wrapper = document.createElement('div');
        wrapper.style.marginBottom = '10px';

        const label = document.createElement('label');
        label.textContent = param;
        label.htmlFor = `shape_${param}`;
        wrapper.style.display = 'flex';
        wrapper.style.alignItems = 'center';
        wrapper.style.gap = '8px';
        label.style.display = 'inline-block';
        label.style.minWidth = '80px'; // optional, keeps labels aligned

        const numberInput = document.createElement('input');
        numberInput.type = 'number';
        numberInput.step = '0.01';
        numberInput.min = '0';
        numberInput.id = `shape_${param}`;

        // Dynamically set the initial value from obj.userData or obj.geometry.parameters
        const paramValue = obj.userData[param] ?? obj.geometry.parameters[param];
        numberInput.value = paramValue;

        const sliderInput = document.createElement('input');
        sliderInput.type = 'range';
        sliderInput.min = '0';
        sliderInput.max = '10';  // Adjust the max as needed
        sliderInput.step = '0.01';
        sliderInput.value = paramValue;

        // Sync number input with slider
        numberInput.addEventListener('input', () => {
            sliderInput.value = numberInput.value;
            if (isValidValue(numberInput.value)) {
                obj.userData[param] = parseFloat(numberInput.value); // Update only the modified parameter
                updateGeometry(obj, param); // Update geometry based on modified parameter
            } else {
                numberInput.style.backgroundColor = 'red'; // Indicate invalid input
            }
        });

        // Sync slider with number input
        sliderInput.addEventListener('input', () => {
            numberInput.value = sliderInput.value;
            if (isValidValue(sliderInput.value)) {
                obj.userData[param] = parseFloat(sliderInput.value); // Update only the modified parameter
                updateGeometry(obj, param); // Update geometry based on modified parameter
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
    const type = object.userData.type;
    const params = object.userData;
    const geometryParams = object.geometry?.parameters || {};  // Failsafe: Use geometry parameters if available

    let newGeometry;
    switch (type) {
        case 'Union_cylinder':
            newGeometry = new THREE.CylinderGeometry(
                params.radius ?? geometryParams.radiusTop ?? 1,   // Failsafe first for userData, then for geometry parameters
                params.radius ?? geometryParams.radiusBottom ?? 1,
                params.height ?? geometryParams.height ?? 1,
                32 // segments
            );
            break;
        case 'Union_box':
            newGeometry = new THREE.BoxGeometry(
                params.width ?? geometryParams.width ?? 1,  // Failsafe for userData, then for geometry parameters
                params.height ?? geometryParams.height ?? 1,
                params.depth ?? geometryParams.depth ?? 1
            );
            break;
        case 'Union_sphere':
            newGeometry = new THREE.SphereGeometry(
                params.radius ?? geometryParams.radius ?? 1,  // Failsafe for userData, then for geometry parameters
                32, 32
            );
            break;
        case 'Union_cone':
            newGeometry = new THREE.CylinderGeometry(
                params.radius_top ?? geometryParams.radius ?? 1,  // Failsafe for userData, then for geometry parameters
                params.radius_bottom ?? geometryParams.radius ?? 1,
                params.height ?? geometryParams.height ?? 1
            );
            break;
        default:
            return;
    }

    // Clean up old geometry and replace it with the new one
    object.geometry.dispose();
    object.geometry = newGeometry;
}



function isValidValue(value) {
    // Check if the value is a valid positive number and not NaN
    return !isNaN(value) && value > 0;
}
