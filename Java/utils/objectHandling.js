// This file contains all the functions for handling the objects in the scene.

// This means, it spawns them, it destroys them, it updates all of them, ....
// updateObject.js

import { showEditPanel, updateObjectList } from '../ui/contextMenu.js';
import * as THREE from 'three';
import { VertexNormalsHelper } from 'three/examples/jsm/helpers/VertexNormalsHelper.js';



//==============================================================================
//================================= spawnObject ================================
//==============================================================================
export function spawnObject(context) {
  const { scene, objects } = context;

  const shapeType = document.getElementById("shape").value;
  const priority = parseInt(document.getElementById("priority").value) || 0;
  const mat_name = document.getElementById("material").value;

  let geometry;
  let unionType = "";
  let params = {}; // geometric parameters to store in userData

  switch (shapeType) {
    case "Box": {
      const width = 1, height = 1, depth = 1;
      geometry = new THREE.BoxGeometry(width, height, depth);
      unionType = "Union_box";
      params = { width, height, depth };
      break;
    }
    case "Sphere": {
      const radius = 0.5;
      geometry = new THREE.SphereGeometry(radius);
      unionType = "Union_sphere";
      params = { radius };
      break;
    }
    case "Cylinder": {
      const radiusTop = 0.5, radiusBottom = 0.5, height = 1;
      geometry = new THREE.CylinderGeometry(radiusTop, radiusBottom, height);
      unionType = "Union_cylinder";
      params = { radius: radiusTop, height }; // cylinder has equal radii
      break;
    }
    case "Cone": {
      const radius = 0.5, height = 1;
      geometry = new THREE.CylinderGeometry(0,radius, height);
      unionType = "Union_cone";
      params = { radius_bottom: radius, radius_top: 0, height }; // default to pointed cone
      break;
    }
    default:
      console.warn("Unknown shape type:", shapeType);
      return;
  }
  
  // Create mesh
  if (!context.materials[mat_name]){
      const randomColor = Math.floor(Math.random() * 0xffffff); // 0x000000 to 0xFFFFFF
      context.materials[mat_name] = new THREE.MeshStandardMaterial({ color: randomColor, name: mat_name ,transparent: true, opacity: 0.5}); 
  } else {
  
  }
  const material = context.materials[mat_name]
  const mesh = new THREE.Mesh(geometry, material);

  mesh.position.set(0, 0, 0);

  // Attach metadata
  mesh.userData = {
    priority,
    materialName: mat_name,
    type: unionType,
    ...params   // ✅ include shape parameters
  };

  // Assign unique name
  const id = `Object${objects.length + 1}`;
  mesh.name = id;

  // Add to context and scene
  objects.push(mesh);
  scene.add(mesh);

  // Set as selected object
  context.selectedObject = mesh;
  document.getElementById("objectSelect").value = id;
  const objectSelect = document.getElementById("objectSelect");
  const option = document.createElement("option");
  option.value = mesh.name;  // Use the mesh name as the value
  option.text = `${mesh.name} (${mesh.userData.type})`;  // Display name and type in the dropdown
  objectSelect.appendChild(option);
  context.selectedObject = mesh;
  objectSelect.value = mesh.name;
  // ✅ Update UI through the centralized function
  updateObjectList(context);

  // Show the edit panel
  showEditPanel(context);

  console.log("Spawned object:", mesh);
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
    console.log(context.materials)
    console.log(context.old_materials)

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
// Update the UI if materials were added, removed, or replaced
export function updateMaterialUI(context) {
  const newEntries = Object.entries(context.materials);
  const oldEntries = context.old_materials || {};

  let changed = false;

  // Compare keys and object references
  if (newEntries.length !== Object.keys(oldEntries).length) {
    changed = true;
  } else {
    for (const [name, mat] of newEntries) {
      if (!(name in oldEntries) || oldEntries[name] !== mat) {
        changed = true;
        break;
      }
    }
  }

  if (changed) {
    console.log('Materials changed');
    populateMaterialList(context);

    // Save snapshot of references
    context.old_materials = { ...context.materials };
  }
}

// Populate the list from context.materials
function populateMaterialList(context) {
  const materialListEl = document.getElementById('materialList');
  materialListEl.innerHTML = '';

  Object.entries(context.materials).forEach(([name, mat]) => {
    const li = document.createElement('li');
    li.className = 'material-item';
    li.textContent = name;

    const colorInput = document.createElement('input');
    colorInput.type = 'color';

    // Use the material's current color
    if (mat.color) {
      colorInput.value = `#${mat.color.getHexString()}`;
    } else {
      colorInput.value = '#ffffff';
      colorInput.disabled = true;
    }

    // When the user changes the color
    colorInput.addEventListener('input', (e) => {
      if (mat.color) {
        mat.color.set(e.target.value); // update the THREE.Material color
        context.renderer.render(context.scene, context.camera); // redraw scene
      }
    });

    li.appendChild(colorInput);
    materialListEl.appendChild(li);
  });
}
