// Initializing the scene
import * as THREE from 'three';

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
  
  export default context;




export function setupLighting(scene) {
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