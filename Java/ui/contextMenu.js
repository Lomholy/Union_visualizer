// This file handles most of the info displayed in the context menu of the
// GUI


import * as THREE from 'three';

// The function to update the list of objects
export function updateObjectList(context) {
  if (!context || !Array.isArray(context.objects)) {
    console.warn("updateObjectList: Invalid context or missing objects array");
    return;
  }
  console.log("objectList element:", document.getElementById("objectList"));
  
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
    // li.onclick = () => {
    //   context.selectedObject = obj;  // Update selectedObject in context if you store it there
    //   document.getElementById("objectSelect").value = obj.name;
    //   showEditPanel(context);  // Show the edit panel for the selected object
    // };
    
    list.appendChild(li);
  });
}

export function updateObjectOrder(context) {
  // Sort objects by priority (ascending, lowest priority first)
  context.objects.sort((a, b) => b.userData.priority - a.userData.priority);

  // Update the renderOrder based on priority
  context.objects.forEach((obj, index) => {
    obj.renderOrder = index;  // Objects with higher priority will be rendered first
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

    // Initialize color picker with current color
    const currentColor = '#' + obj.material.color.getHexString();
    document.getElementById('editColor').value = currentColor;

    // Update color on button click
    document.getElementById('updateColorBtn').onclick = () => {
        const newColor = document.getElementById('editColor').value;
        obj.material.color.set(newColor);
        obj.userData.materialName = newColor;
    };

    // ⬇️ SHAPE PARAMETERS SECTION ⬇️
    const paramContainer = document.getElementById('shapeParams');
    console.log(obj.geometry);
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
    const type = object.geometry?.type;
    const params = object.userData;
    const geometryParams = object.geometry?.parameters || {};  // Failsafe: Use geometry parameters if available

    let newGeometry;
    switch (type) {
        case 'CylinderGeometry':
            newGeometry = new THREE.CylinderGeometry(
                params.radiusTop ?? geometryParams.radiusTop ?? 1,   // Failsafe first for userData, then for geometry parameters
                params.radiusBottom ?? geometryParams.radiusBottom ?? 1,
                params.height ?? geometryParams.height ?? 1,
                32 // segments
            );
            break;
        case 'BoxGeometry':
            newGeometry = new THREE.BoxGeometry(
                params.width ?? geometryParams.width ?? 1,  // Failsafe for userData, then for geometry parameters
                params.height ?? geometryParams.height ?? 1,
                params.depth ?? geometryParams.depth ?? 1
            );
            break;
        case 'SphereGeometry':
            newGeometry = new THREE.SphereGeometry(
                params.radius ?? geometryParams.radius ?? 1,  // Failsafe for userData, then for geometry parameters
                32, 32
            );
            break;
        case 'ConeGeometry':
            newGeometry = new THREE.ConeGeometry(
                params.radius ?? geometryParams.radius ?? 1,  // Failsafe for userData, then for geometry parameters
                params.height ?? geometryParams.height ?? 1,
                32
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
