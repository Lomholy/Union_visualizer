import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { setupLighting } from './lighting.js';
import { spawnObject } from './spawnObject.js';
import { updateObjectList } from './updateObjectList.js';
import context from './appContext.js';
import { onMouseClick } from './clickOnObjects.js';
import { setupUpdateHandler } from './updateObject.js';


initScene(context);
setupLighting(context.scene);
document.getElementById("spawnBtn").addEventListener("click", function() {
  spawnObject(context);  // Call spawnObject with scene and objects when the button is clicked
});

function initScene(context) {
  context.scene = new THREE.Scene();
  context.scene.background = new THREE.Color(0xffffff); // White color
  // Add an Axes Helper to visualize the coordinate system
  const axesHelper = new THREE.AxesHelper(5); // Size of the axes (5 units long)
  context.scene.add(axesHelper);

  context.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);

  context.renderer = new THREE.WebGLRenderer({ antialias: true });
  context.renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(context.renderer.domElement);

  context.controls = new OrbitControls(context.camera, context.renderer.domElement);
  context.controls.enableDamping = true; // Optional, for smoother movement
  context.controls.dampingFactor = 0.25; // Optional, speed of damping
  context.controls.screenSpacePanning = false; // Optional, prevents camera from panning along the screen space
  context.controls.target.set(0, 0, 0); // Camera will always orbit around (0, 0, 0)
  context.camera.position.set(5, 5, 5); // Set it a bit far from the center to see the objects

  context.raycaster = new THREE.Raycaster();
  context.mouse = new THREE.Vector2();
  context.renderer.domElement.addEventListener('click', function(event) {
    onMouseClick(event, context);
  });
  setupUpdateHandler(context);
  animate(context);
}

function animate(context) {
  requestAnimationFrame(() => animate(context));  // Pass context explicitly
  context.controls.update();
  context.renderer.render(context.scene, context.camera);
}


// document.getElementById("loadInstrBtn").addEventListener("click", () => {
//   document.getElementById("instrFile").click();
// });

// document.getElementById("instrFile").addEventListener("change", handleInstrFile);

// function handleInstrFile(event) {
//   const file = event.target.files[0];
//   if (!file) return;

//   const reader = new FileReader();
//   reader.onload = e => {
//     const content = e.target.result;
//     parseInstrFile(content);
//   };
//   reader.readAsText(file);
// }



// function parseParams(block) {
//   const paramRegex = /(\w+)\s*=\s*("?[^",\n]+?"?)/g;
//   const params = {};
//   let match;
//   while ((match = paramRegex.exec(block)) !== null) {
//     let key = match[1];
//     let val = match[2].replace(/"/g, '');
//     val = isNaN(val) ? val : parseFloat(val);
//     params[key] = val;
//   }
//   return params;
// }

// function spawnFromInstr({ name, type, params, position }) {
//   let geometry;
//   const color = 0xff0000; // Default red
//   const material = new THREE.MeshStandardMaterial({ color });

//   // Match shape by known types or parameter patterns
//   if (params.radius) {
//     geometry = new THREE.SphereGeometry(params.radius, 32, 32);
//   } else if (params.xwidth && params.yheight && (params.zdepth || params.zlength)) {
//     geometry = new THREE.BoxGeometry(
//       params.xwidth,
//       params.yheight,
//       params.zdepth || params.zlength
//     );
//   } else {
//     console.warn(`Unknown shape for component ${name}, skipping.`);
//     return;
//   }

//   const mesh = new THREE.Mesh(geometry, material);
//   mesh.name = name;
//   mesh.position.set(...position);
//   mesh.userData.materialName = type;
//   mesh.userData.priority = 0;

//   objects.push(mesh);
//   scene.add(mesh);

//   // Add to dropdown
//   const option = document.createElement("option");
//   option.value = mesh.name;
//   option.text = `${mesh.name} (${geometry.type.replace('Geometry', '')})`;
//   document.getElementById("objectSelect").appendChild(option);

//   updateObjectList();
// }
