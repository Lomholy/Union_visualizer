import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

let scene, camera, renderer, controls;
const objects = []; // Store spawned objects
const sliderListeners = {};
const inputListeners = {};



initScene();
setupLighting();

document.getElementById("spawnBtn").addEventListener("click", spawnObject);

function initScene() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff); // White color
  // Add an Axes Helper to visualize the coordinate system
  const axesHelper = new THREE.AxesHelper(5); // Size of the axes (5 units long)
  scene.add(axesHelper);

  camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; // Optional, for smoother movement
  controls.dampingFactor = 0.25; // Optional, speed of damping
  controls.screenSpacePanning = false; // Optional, prevents camera from panning along the screen space
  controls.target.set(0, 0, 0); // Camera will always orbit around (0, 0, 0)
  camera.position.set(5, 5, 5); // Set it a bit far from the center to see the objects

  animate();
}

function setupLighting() {
  // Hemisphere light: gives a soft ambient glow from sky and ground
  const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.6);
  hemiLight.position.set(0, 1, 0);
  scene.add(hemiLight);

  // Main directional light (like a sun)
  const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight1.position.set(5, 5, 5);
  scene.add(dirLight1);

  // Fill directional light from the opposite side
  const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
  dirLight2.position.set(-5, 5, -5);
  scene.add(dirLight2);

  // Optional: point light at the center of the scene
  const pointLight = new THREE.PointLight(0xffffff, 0.3);
  pointLight.position.set(0, 2, 2);
  scene.add(pointLight);
}


function spawnObject() {
  const shapeType = document.getElementById("shape").value;
  const priority = parseInt(document.getElementById("priority").value) || 0;
  const color = document.getElementById("material").value;

  let geometry;
  switch (shapeType) {
    case "Box": geometry = new THREE.BoxGeometry(); break;
    case "Sphere": geometry = new THREE.SphereGeometry(0.5, 32, 32); break;
    case "Cylinder": geometry = new THREE.CylinderGeometry(0.5, 0.5, 1, 32); break;
    case "Cone": geometry = new THREE.ConeGeometry(0.5, 1, 32); break;
  }

  const material = new THREE.MeshStandardMaterial({ color });
  const mesh = new THREE.Mesh(geometry, material);

  mesh.position.set(0,0,0);
  mesh.userData.priority = priority;
  mesh.userData.materialName = ''; // Initialize material name

  // Assign a name or ID for the dropdown
  const id = `Object${objects.length + 1}`;
  mesh.name = id;
  objects.push(mesh);
  scene.add(mesh);

  // Add to dropdown
  const option = document.createElement("option");
  option.value = id;
  option.text = `${id} (${shapeType})`;
  document.getElementById("objectSelect").appendChild(option);
  updateObjectList();

}


function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
let selectedObject = null;
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

// Click handler for selecting object
renderer.domElement.addEventListener('click', onMouseClick);

function onMouseClick(event) {
  // Convert mouse to normalized device coordinates (-1 to +1)
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

  // // Raycast from camera to scene
  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(scene.children);

  if (intersects.length > 0) {
    selectedObject = intersects[0].object;
    document.getElementById("objectSelect").value = selectedObject.name;
    showEditPanel(selectedObject);
  }
}

function showEditPanel(object) {
  document.getElementById('editPanel').style.display = 'block';

  document.getElementById('objectName').value = object.name || '';

  document.getElementById('posX').value = object.position.x.toFixed(2);
  document.getElementById('posY').value = object.position.y.toFixed(2);
  document.getElementById('posZ').value = object.position.z.toFixed(2);
  
  document.getElementById('posXSlider').value = object.position.x;
  document.getElementById('posYSlider').value = object.position.y;
  document.getElementById('posZSlider').value = object.position.z;

  // Sync sliders and inputs live
  linkSliderAndInput('posXSlider', 'posX', val => object.position.x = val);
  linkSliderAndInput('posYSlider', 'posY', val => object.position.y = val);
  linkSliderAndInput('posZSlider', 'posZ', val => object.position.z = val);
    
  document.getElementById('rotX').value = selectedObject.rotation.x.toFixed(2);
  document.getElementById('rotY').value = selectedObject.rotation.y.toFixed(2);
  document.getElementById('rotZ').value = selectedObject.rotation.z.toFixed(2);

  document.getElementById('rotXSlider').value = selectedObject.rotation.x;
  document.getElementById('rotYSlider').value = selectedObject.rotation.y;
  document.getElementById('rotZSlider').value = selectedObject.rotation.z;

  // Sync sliders and inputs for rotation
  linkSliderAndInput('rotXSlider', 'rotX', val => selectedObject.rotation.x = val);
  linkSliderAndInput('rotYSlider', 'rotY', val => selectedObject.rotation.y = val);
  linkSliderAndInput('rotZSlider', 'rotZ', val => selectedObject.rotation.z = val);

  document.getElementById('editPriority').value = object.userData.priority || 0;
  document.getElementById('materialName').value = object.userData.materialName || '';
}


document.getElementById('updateBtn').addEventListener('click', () => {
  if (!selectedObject) return;

  // Update name
  const newName = document.getElementById('objectName').value.trim();
  if (newName && newName !== selectedObject.name) {
    // Update the object name
    const oldName = selectedObject.name;
    selectedObject.name = newName;

    // Update dropdown display text
    const dropdown = document.getElementById("objectSelect");
    const option = [...dropdown.options].find(opt => opt.value === oldName);
    if (option) {
      option.value = newName;
      option.text = `${newName} (${selectedObject.geometry.type.replace('Geometry', '')})`;
    }
  }
  if (objects.some(obj => obj !== selectedObject && obj.name === newName)) {
    alert("Name already in use. Please choose a unique name.");
    return;
  }
  

  // Update position
  selectedObject.position.set(
    parseFloat(document.getElementById('posX').value),
    parseFloat(document.getElementById('posY').value),
    parseFloat(document.getElementById('posZ').value)
  );

  // Update rotation
  selectedObject.rotation.set(
    parseFloat(document.getElementById('rotX').value),
    parseFloat(document.getElementById('rotY').value),
    parseFloat(document.getElementById('rotZ').value)
  );

  // Priority and material
  selectedObject.userData.priority = parseInt(document.getElementById('editPriority').value) || 0;
  selectedObject.userData.materialName = document.getElementById('materialName').value;
  updateObjectList();

  console.log('Updated object:', selectedObject);
});


function updateObjectList() {
  const list = document.getElementById("objectList");
  list.innerHTML = ""; // Clear the list

  objects.forEach(obj => {
    const li = document.createElement("li");
    li.textContent = `${obj.name} (Priority: ${obj.userData.priority}, Material: ${obj.userData.materialName || "none"})`;
    
    // Optional: make items clickable to select
    li.style.cursor = "pointer";
    li.onclick = () => {
      selectedObject = obj;
      showEditPanel(obj);
      document.getElementById("objectSelect").value = obj.name;
    };

    list.appendChild(li);
  });
}
function linkSliderAndInput(sliderId, inputId, onChange) {
  const slider = document.getElementById(sliderId);
  const input = document.getElementById(inputId);

  // Remove previous listeners if they exist
  if (sliderListeners[sliderId]) slider.removeEventListener('input', sliderListeners[sliderId]);
  if (inputListeners[inputId]) input.removeEventListener('input', inputListeners[inputId]);

  // Create new listeners
  const sliderHandler = () => {
    input.value = slider.value;
    onChange(parseFloat(slider.value));
    updateObjectList();
  };

  const inputHandler = () => {
    slider.value = input.value;
    onChange(parseFloat(input.value));
    updateObjectList();
  };

  // Store them for future cleanup
  sliderListeners[sliderId] = sliderHandler;
  inputListeners[inputId] = inputHandler;

  // Add the listeners
  slider.addEventListener('input', sliderHandler);
  input.addEventListener('input', inputHandler);
}

document.getElementById("loadInstrBtn").addEventListener("click", () => {
  document.getElementById("instrFile").click();
});

document.getElementById("instrFile").addEventListener("change", handleInstrFile);

function handleInstrFile(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = e => {
    const content = e.target.result;
    parseInstrFile(content);
  };
  reader.readAsText(file);
}

function parseInstrFile(text) {
  const componentRegex = /COMPONENT\s+(\w+)\s*=\s*(\w+)\(([\s\S]*?)\)\s*AT\s*\(([^)]+)\)/g;

  let match;
  while ((match = componentRegex.exec(text)) !== null) {
    const [_, name, type, paramsBlock, positionStr] = match;

    const params = parseParams(paramsBlock);
    const position = positionStr.split(',').map(s => parseFloat(s.trim()));
    if (!type.startsWith("Union_")) {continue;}
    spawnFromInstr({
      name,
      type,
      params,
      position
    });
  }
}

function parseParams(block) {
  const paramRegex = /(\w+)\s*=\s*("?[^",\n]+?"?)/g;
  const params = {};
  let match;
  while ((match = paramRegex.exec(block)) !== null) {
    let key = match[1];
    let val = match[2].replace(/"/g, '');
    val = isNaN(val) ? val : parseFloat(val);
    params[key] = val;
  }
  return params;
}

function spawnFromInstr({ name, type, params, position }) {
  let geometry;
  const color = 0xff0000; // Default red
  const material = new THREE.MeshStandardMaterial({ color });

  // Match shape by known types or parameter patterns
  if (params.radius) {
    geometry = new THREE.SphereGeometry(params.radius, 32, 32);
  } else if (params.xwidth && params.yheight && (params.zdepth || params.zlength)) {
    geometry = new THREE.BoxGeometry(
      params.xwidth,
      params.yheight,
      params.zdepth || params.zlength
    );
  } else {
    console.warn(`Unknown shape for component ${name}, skipping.`);
    return;
  }

  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = name;
  mesh.position.set(...position);
  mesh.userData.materialName = type;
  mesh.userData.priority = 0;

  objects.push(mesh);
  scene.add(mesh);

  // Add to dropdown
  const option = document.createElement("option");
  option.value = mesh.name;
  option.text = `${mesh.name} (${geometry.type.replace('Geometry', '')})`;
  document.getElementById("objectSelect").appendChild(option);

  updateObjectList();
}
