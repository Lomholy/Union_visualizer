// spawnObject.js
import * as THREE from 'three';
import { showEditPanel } from './editPanel.js';


export function spawnObject(context) {
  const { scene, objects } = context;

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

  mesh.position.set(0, 0, 0);
  mesh.userData.priority = priority;
  mesh.userData.materialName = '';

  const id = `Object${objects.length + 1}`;
  mesh.name = id;
  objects.push(mesh);
  scene.add(mesh);

  const option = document.createElement("option");
  option.value = id;
  option.text = `${id} (${shapeType})`;
  document.getElementById("objectSelect").appendChild(option);
  // Set as selected object
  context.selectedObject = mesh;
  document.getElementById("objectSelect").value = id;

  showEditPanel(context);
  console.log(context);
}