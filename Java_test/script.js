import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { setupLighting } from './lighting.js';
import { spawnObject } from './spawnObject.js';
import { updateObjectList } from './updateObjectList.js';
import context from './appContext.js';
import { onMouseClick } from './clickOnObjects.js';
import { setupUpdateHandler,setupDeleteHandler } from './updateObject.js';
import { loadInstrumentFile } from './importInstr.js';



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
  setupDeleteHandler(context);
  loadInstrumentFile(context);
  animate(context);
}

function animate(context) {
  requestAnimationFrame(() => animate(context));  // Pass context explicitly
  context.controls.update();
  context.renderer.render(context.scene, context.camera);
}