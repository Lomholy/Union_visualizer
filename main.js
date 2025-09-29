import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { init_slicePanel } from './ui/slicePanel.js';
import { showEditPanel} from './ui/contextMenu.js';

import {loadInstrumentFile} from './InputOutput/Import.js';
import {writeInstr} from './InputOutput/Export.js';

import {spawnObjectFromUI} from './objectHandling/spawnObject.js';
import {setSelectedObject,setupLiveUpdateHandlers} from './objectHandling/selectedObject.js';
import {updateAllMaterials} from './objectHandling/changeRenderStyle.js';
import {setupDeleteHandler} from './objectHandling/deleteObject.js';

import {updateMaterialList } from './continuousUpdating/updateMaterialList.js';
import {updateObjectList} from './continuousUpdating/objectList.js';
import {updateObjectOrder} from './continuousUpdating/prioritysort.js';

const context = {
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    objects: [],
    sliderListeners: {},
    inputListeners: {},
    selectedObject: null,
    raycaster: null,
    mouse: null,
    materials: {},
    old_materials: {}
};
 

initScene(context);
setupLighting(context.scene);
let frameCount = 0;
let previousSnapshot = null;
animate(context);

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
  if (typeof showEditPanel === 'function') showEditPanel(context);
  context.controls.target.set(0, 0, 0); // Camera will always orbit around (0, 0, 0)
  context.camera.position.set(5, 5, 5); // Set it a bit far from the center to see the objects
  
  context.mouse = new THREE.Vector2();
  document.getElementById("spawnBtn").addEventListener("click", function() {
    spawnObjectFromUI(context);  // Call spawnObject with scene and objects when the button is clicked
  });
  document.getElementById('renderStyle').addEventListener('change', function() {
    updateAllMaterials(context); // Or wherever your context is stored
  });
  setupLiveUpdateHandlers(context);
  setupDeleteHandler(context);
  loadInstrumentFile(context);
  init_slicePanel(context);
  document.getElementById('downloadBtn').addEventListener('click', function() {
  writeInstr(context);
  });

  // Assuming the 'objectSelect' is the dropdown for object selection
  document.getElementById('objectSelect').addEventListener('change', function () {
    const selectedObjectName = this.value;
    const selectedObject = context.objects.find(obj => obj.name === selectedObjectName);

    if (selectedObject) {
      context.selectedObject = selectedObject; // Update selected object in context
      setSelectedObject(context, selectedObject);
      showEditPanel(context); // Show the edit panel for the newly selected object
    }
  }); 

}


function setupLighting(scene) {
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





function animate(context) {
  requestAnimationFrame(() => animate(context));  // Pass context explicitly
  
  frameCount++;
   if (frameCount%200==0){
     // console.log(frameCount);
      

         updateMaterialList(context);
         updateObjectList(context);
   }
  updateObjectOrder(context);
  
  
  context.controls.update();
  context.renderer.render(context.scene, context.camera);
}
