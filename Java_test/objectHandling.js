// This file contains all the functions for handling the objects in the scene.

// This means, it spawns them, it destroys them, it updates all of them, ....
// updateObject.js

import { showEditPanel } from './contextMenu.js';
import * as THREE from 'three';
import { VertexNormalsHelper } from 'three/examples/jsm/helpers/VertexNormalsHelper.js';



//==============================================================================
//================================= spawnObject ================================
//==============================================================================
export function spawnObject(context) {
  const { scene, objects } = context;

  const shapeType = document.getElementById("shape").value;
  const priority = parseInt(document.getElementById("priority").value) || 0;
  const color = document.getElementById("material").value;

  let geometry;
  let unionType = ""; // <- Track the Union_type string

  switch (shapeType) {
    case "Box":
      geometry = new THREE.BoxGeometry();
      unionType = "Union_box";
      break;
    case "Sphere":
      geometry = new THREE.SphereGeometry(0.5, 32, 32);
      unionType = "Union_sphere";
      break;
    case "Cylinder":
      geometry = new THREE.CylinderGeometry(0.5, 0.5, 1, 32);
      unionType = "Union_cylinder";
      break;
    case "Cone":
      geometry = new THREE.ConeGeometry(0.5, 1, 32);
      unionType = "Union_cone";
      break;
    default:
      console.warn("Unknown shape type:", shapeType);
      return;
  }

  const material = new THREE.MeshStandardMaterial({ color });
  const mesh = new THREE.Mesh(geometry, material);

  mesh.position.set(0, 0, 0);
  mesh.userData.priority = priority;
  mesh.userData.materialName = '';
  mesh.userData.type = unionType; // <- Set the correct type here


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


//==============================================================================
//=========================== Update button logic ==============================
//==============================================================================

export function setupUpdateHandler(context) {
  document.getElementById('updateBtn').addEventListener('click', () => {
    const { selectedObject, objects } = context;
    if (!selectedObject) return;
    

    const newName = document.getElementById('objectName').value.trim();
    if (newName && newName !== selectedObject.name) {
      const oldName = selectedObject.name;
      selectedObject.name = newName;

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

    selectedObject.position.set(
      parseFloat(document.getElementById('posX').value),
      parseFloat(document.getElementById('posY').value),
      parseFloat(document.getElementById('posZ').value)
    );

    selectedObject.rotation.set(
      parseFloat(document.getElementById('rotX').value),
      parseFloat(document.getElementById('rotY').value),
      parseFloat(document.getElementById('rotZ').value)
    );

    selectedObject.userData.priority = parseInt(document.getElementById('editPriority').value) || 0;
    selectedObject.userData.materialName = document.getElementById('materialName').value;

    console.log('Updated object:', selectedObject);
  });
}


//==============================================================================
//=========================== Delete Button logic ==============================
//==============================================================================

export function setupDeleteHandler(context) {
  document.getElementById('deleteBtn').addEventListener('click', () => {
    const { selectedObject, scene, objects } = context;
    if (!selectedObject) return;

    // Remove from scene
    scene.remove(selectedObject);

    // Remove from objects list
    const index = objects.indexOf(selectedObject);
    if (index !== -1) objects.splice(index, 1);

    // Remove from dropdown
    const dropdown = document.getElementById("objectSelect");
    const option = [...dropdown.options].find(opt => opt.value === selectedObject.name);
    if (option) dropdown.removeChild(option);

    // Remove from object list UI
    const list = document.getElementById("objectList");
    const item = [...list.children].find(li => li.textContent.includes(selectedObject.name));
    if (item) list.removeChild(item);

    // Update selection: select first remaining object, if any
    context.selectedObject = objects[0] || null;
    if (context.selectedObject) {
      document.getElementById("objectSelect").value = context.selectedObject.name;
      showEditPanel(context);
    } else {
      document.getElementById('editPanel').style.display = 'none'; // Hide panel if nothing left
    }

    console.log("Object deleted.");
  });
}

//==============================================================================
//========== Update of all materials when switching view styles ================
//==============================================================================

export function updateAllMaterials(context) {
  const style = document.getElementById('renderStyle').value;
  
  context.objects.forEach(obj => {
    if (!obj.isMesh) return;
    const baseColor = obj.material.color || new THREE.Color('#808080');

    let material;

    switch (style) {
      case 'solid':
        material = new THREE.MeshStandardMaterial({ color: baseColor, wireframe: false, transparent: false });
        break;

      case 'wireframe':
        material = new THREE.MeshBasicMaterial({ color: baseColor, wireframe: true });
        break;

      case 'transparent':
        material = new THREE.MeshStandardMaterial({ color: baseColor, transparent: true, opacity: 0.5 });
        break;

      case 'edges':
        // Keep solid mesh + add edges
        material = new THREE.MeshStandardMaterial({ color: baseColor });
        if (!obj.userData.edgeHelper) {
          const edgeHelper = new THREE.EdgesGeometry(obj.geometry);
          const line = new THREE.LineSegments(edgeHelper, new THREE.LineBasicMaterial({ color: 0x000000 }));
          line.name = 'edgeHelper';
          obj.add(line);
          obj.userData.edgeHelper = line;
        }
        break;

      case 'normals':
        // Visualize normals
        if (!obj.userData.normalHelper) {
          const normalHelper = new VertexNormalsHelper(obj, 0.2, 0x0000ff);
          obj.userData.normalHelper = normalHelper;
          context.scene.add(normalHelper);
        }
        break;
    }

    // Remove extra helpers if switching back
    if (style !== 'edges' && obj.userData.edgeHelper) {
      obj.remove(obj.userData.edgeHelper);
      delete obj.userData.edgeHelper;
    }

    if (style !== 'normals' && obj.userData.normalHelper) {
      context.scene.remove(obj.userData.normalHelper);
      delete obj.userData.normalHelper;
    }

    obj.material = material;
  });
  console.log(context.objects[0].userData);
}
