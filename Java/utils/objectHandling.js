// This file contains all the functions for handling the objects in the scene.

// This means, it spawns them, it destroys them, it updates all of them, ....
// updateObject.js

import { showEditPanel, updateObjectList } from '../ui/contextMenu.js';
import * as THREE from 'three';
import { VertexNormalsHelper } from 'three/examples/jsm/helpers/VertexNormalsHelper.js';



//==============================================================================
//================================= spawnObject ================================
//==============================================================================

/**
 * Generalized object creation used by both the UI and the import pipeline.
 * It adds the mesh to scene/context, updates the object dropdown, selects it, and refreshes UI.
 *
 * @param {object} context - { scene, objects, materials, selectedObject, ... }
 * @param {object} spec - Declarative description of the object:
 *   {
 *     type: 'Union_box'|'Union_sphere'|'Union_cylinder'|'Union_cone',
 *     name?: string,
 *     materialName?: string,
 *     priority?: number,
 *     params?: object,                 // geometry params (see cases below)
 *     position?: {x:number,y:number,z:number}|THREE.Vector3|[x,y,z],
 *     rotation?: {x:number,y:number,z:number}|[x,y,z], // radians
 *     rotationDegrees?: {x:number,y:number,z:number}|[x,y,z] // degrees (import)
 *   }
 * @returns {THREE.Mesh}
 */
export function spawnObject(context, spec) {
  const { scene, objects } = context;
  if (!spec || !spec.type) {
    throw new Error('spawnObject: spec.type is required (e.g., "Union_box").');
  }

  const type = spec.type;
  const paramsIn = spec.params || {};
  const materialName = spec.materialName ?? 'default';

  // Ensure material exists (color stable across uses)
  ensureMaterial(context, materialName);

  // --- Build geometry and normalize params for userData/export ---
  let geometry;
  /** @type {object} */
  let params = {};

  switch (type) {
    case 'Union_box': {
      const width  = val(paramsIn.width,  paramsIn.xwidth, 1);
      const height = val(paramsIn.height, paramsIn.yheight, 1);
      const depth  = val(paramsIn.depth,  paramsIn.zdepth, 1);
      geometry = new THREE.BoxGeometry(width, height, depth);
      params = { width, height, depth, ...paramsIn };
      break;
    }
    case 'Union_sphere': {
      const radius = val(paramsIn.radius, undefined, 0.5);
      geometry = new THREE.SphereGeometry(radius);
      params = { radius, ...paramsIn };
      break;
    }
    case 'Union_cylinder': {
      // Accept {radius, height} or {radiusTop,radiusBottom,height}
      const radius = val(paramsIn.radius, paramsIn.radiusTop ?? paramsIn.radius_bottom, 0.5);
      const height = val(paramsIn.height, paramsIn.yheight, 1);
      geometry = new THREE.CylinderGeometry(radius, radius, height);
      params = { radius, height, ...paramsIn };
      break;
    }
    case 'Union_cone': {
      // CylinderGeometry with different top/bottom
      const radiusTop    = val(paramsIn.radius_top, paramsIn.radiusTop, 0);
      const radiusBottom = val(paramsIn.radius_bottom, paramsIn.radiusBottom, paramsIn.radius ?? 0.5);
      const height       = val(paramsIn.height, paramsIn.yheight, 1);
      geometry = new THREE.CylinderGeometry(radiusTop, radiusBottom, height);
      params = { radiusTop, radiusBottom, height, ...paramsIn };
      break;
    }
    default:
      console.warn('spawnObject: unknown type', type);
      return null;
  }

  const material = context.materials[materialName];
  const mesh = new THREE.Mesh(geometry, material);

  // --- Name (ensure uniqueness if needed) ---
  let name = spec.name ?? `Object${objects.length + 1}`;
  if (objects.some(o => o.name === name)) {
    let i = 2, base = name;
    while (objects.some(o => o.name === `${base}_${i}`)) i++;
    name = `${base}_${i}`;
  }
  mesh.name = name;

  // --- Metadata expected by writeInstr/buildParameters ---
  // Keep keys compatible with your export code (width/height/depth, radius,
  // radiusTop/radiusBottom, height, materialName, priority, type, etc.)
  mesh.userData = {
    priority: spec.priority ?? 0,
    materialName,
    type,
    ...params
  };

  // --- Transform (position/rotation) ---
  if (spec.position) {
    const p = toXYZ(spec.position);
    mesh.position.set(p.x, p.y, p.z);
  }
  if (spec.rotation) {
    const r = toXYZ(spec.rotation);
    mesh.rotation.set(r.x, r.y, r.z);
  } else if (spec.rotationDegrees) {
    const r = toXYZ(spec.rotationDegrees);
    mesh.rotation.set(
      THREE.MathUtils.degToRad(r.x),
      THREE.MathUtils.degToRad(r.y),
      THREE.MathUtils.degToRad(r.z)
    );
  }

  // --- Add to scene and context ---
  scene.add(mesh);
  objects.push(mesh);

  // --- UI: add to dropdown, select, and refresh list/panel ---
  addToObjectSelect(mesh);
  context.selectedObject = mesh;
  if (typeof updateObjectList === 'function') updateObjectList(context);
  if (typeof showEditPanel === 'function') showEditPanel(context);

  return mesh;

  // ---------- helpers ----------
  function val(...candidates) {
    for (const c of candidates) if (Number.isFinite(c)) return c;
    return candidates[candidates.length - 1]; // last is default
  }
  function toXYZ(v) {
    if (Array.isArray(v)) return { x: v[0] ?? 0, y: v[1] ?? 0, z: v[2] ?? 0 };
    if ('isVector3' in (v || {})) return { x: v.x ?? 0, y: v.y ?? 0, z: v.z ?? 0 };
    return { x: v?.x ?? 0, y: v?.y ?? 0, z: v?.z ?? 0 };
  }
  function ensureMaterial(ctx, name) {
    if (!ctx.materials[name]) {
      const randomColor = Math.floor(Math.random() * 0xffffff);
      ctx.materials[name] = new THREE.MeshStandardMaterial({
        color: randomColor, name, transparent: true, opacity: 0.5
      });
    }
  }
  function addToObjectSelect(m) {
    const sel = /** @type {HTMLSelectElement} */(document.getElementById('objectSelect'));
    if (!sel) return;
    // If option already exists (same name), don't duplicate
    const existing = [...sel.options].find(o => o.value === m.name);
    if (!existing) {
      const opt = document.createElement('option');
      opt.value = m.name;
      opt.text = `${m.name} (${m.userData.type})`;
      sel.appendChild(opt);
    } else {
      existing.text = `${m.name} (${m.userData.type})`;
    }
    sel.value = m.name;
  }
}

/**
 * Thin wrapper to preserve your existing UI flow:
 * reads DOM fields and calls the generalized spawnObject().
 * This can replace your old spawnObject(context).
 */
export function spawnObjectFromUI(context) {
  const shapeType = document.getElementById("shape").value;
  const priority = parseInt(document.getElementById("priority").value) || 0;
  const materialName = document.getElementById("material").value;

  let type = '';
  let params = {};

  switch (shapeType) {
    case "Box":      type = 'Union_box';     params = { width: 1,  height: 1, depth: 1 }; break;
    case "Sphere":   type = 'Union_sphere';  params = { radius: 0.5 };                    break;
    case "Cylinder": type = 'Union_cylinder';params = { radius: 0.5, height: 1 };         break;
    case "Cone":     type = 'Union_cone';    params = { radiusTop: 0, radiusBottom: 0.5, height: 1 }; break;
    default: console.warn("Unknown shape type:", shapeType); return null;
  }

  return spawnObject(context, {
    type,
    materialName,
    priority,
    params
  });
}


//==============================================================================
//=========================== Update selected object============================
/* In main.js when the scene is made, setupLiveUpdateHandlers is called once.
 * After that, setSelectedObject is set whenever a new object is selected.
 * This is done in main as an event listener to objectSelect.*/
//==============================================================================

// helpers: cache DOM elements so we don't repeatedly query them
function cacheUI(context) {
  if (context._ui) return context._ui;

  const get = id => /** @type {HTMLInputElement|null} */ (document.getElementById(id));

  const ui = {
    objectSelect: document.getElementById('objectSelect'), // may be a <select>
    objectName: get('objectName'),

    posX: get('posX'),
    posY: get('posY'),
    posZ: get('posZ'),

    rotX: get('rotX'),
    rotY: get('rotY'),
    rotZ: get('rotZ'),

    editPriority: get('editPriority'),
    materialName: get('materialName'),
  };

  ui.allInputs = [
    ui.objectName,
    ui.posX, ui.posY, ui.posZ,
    ui.rotX, ui.rotY, ui.rotZ,
    ui.editPriority,
    ui.materialName
  ].filter(Boolean);

  context._ui = ui;
  return ui;
}

function disableInputs(context, disabled) {
  const { allInputs } = cacheUI(context);
  allInputs.forEach(el => { el.disabled = !!disabled; });
}

function getNum(el) {
  if (!el) return 0;
  const v = parseFloat(el.value);
  return Number.isFinite(v) ? v : 0;
}

function getStr(el) {
  return el ? el.value.trim() : '';
}

/**
 * Call this once at app init. It attaches one set of listeners that
 * always operate on `context.selectedObject`.
 */
export function setupLiveUpdateHandlers(context) {
  if (context._liveUpdateAttached) return; // idempotent
  context._liveUpdateAttached = true;

  const ui = cacheUI(context);

  const update = () => {
    const obj = context.selectedObject;
    if (!obj) return;

    // --- Name update & dropdown sync ---
    const objectSelect = ui.objectSelect;
    const oldName = obj.name;
    const newName = getStr(ui.objectName);

    if (newName && newName !== oldName) {
      obj.name = newName;

      // Update dropdown option to reflect the new name
      if (objectSelect && objectSelect.options) {
        const option = [...objectSelect.options].find(opt => opt.value === oldName);
        if (option) {
          option.value = newName;
          option.text = `${newName} (${obj.userData.type ?? ''})`.trim();
        }
        // Ensure the selection points at the renamed object
        objectSelect.value = newName;
      }
    } else {
      // keep obj.name in sync if user typed same/empty
      if (ui.objectName) obj.name = getStr(ui.objectName) || obj.name;
    }

    // --- Transform updates ---
    const px = getNum(ui.posX);
    const py = getNum(ui.posY);
    const pz = getNum(ui.posZ);
    obj.position.set(px, py, pz);

    const rx = getNum(ui.rotX) * Math.PI / 180;
    const ry = getNum(ui.rotY) * Math.PI / 180;
    const rz = getNum(ui.rotZ) * Math.PI / 180;
    obj.rotation.set(rx, ry, rz);

    // --- Priority in userData ---
    if (!obj.userData) obj.userData = {};
    obj.userData.priority = Number.isFinite(getNum(ui.editPriority))
      ? parseInt(ui.editPriority.value, 10) || 0
      : 0;

    // --- Material handling ---
    const mat_name = getStr(ui.materialName);
    if (mat_name) {
      if (!context.materials) context.materials = {};
      if (!context.materials[mat_name]) {
        const randomColor = Math.floor(Math.random() * 0xffffff);
        context.materials[mat_name] = new THREE.MeshStandardMaterial({
          color: randomColor,
          name: mat_name,
          transparent: true,
          opacity: 0.5
        });
      }
      obj.userData.materialName = mat_name;
      obj.material = context.materials[mat_name];
    }

    // --- CLEANUP UNUSED MATERIALS ---
    if (context.materials && Array.isArray(context.objects)) {
      const usedMaterials = new Set(
        context.objects
          .map(o => o.material?.name)
          .filter(Boolean)
      );
      for (const matName of Object.keys(context.materials)) {
        if (!usedMaterials.has(matName)) {
          delete context.materials[matName];
          console.log(`Material "${matName}" removed from materials dict.`);
        }
      }
    }

    // --- Keep your list UI in sync ---
    if (typeof updateObjectList === 'function') {
      updateObjectList(context);
    }
  };

  // Attach ONE set of listeners that always reads/writes current selection.
  ui.allInputs.forEach(el => {
    el.addEventListener('input', update);
  });

}

export function setSelectedObject(context, obj) {
  context.selectedObject = obj;

  const ui = cacheUI(context);

  if (!obj) return;

  // Reflect object values in the UI (programmatic .value set does not fire 'input')
  if (ui.objectName) ui.objectName.value = obj.name ?? '';

  if (ui.posX) ui.posX.value = (obj.position?.x ?? 0).toFixed(3);
  if (ui.posY) ui.posY.value = (obj.position?.y ?? 0).toFixed(3);
  if (ui.posZ) ui.posZ.value = (obj.position?.z ?? 0).toFixed(3);

  if (ui.rotX) ui.rotX.value = (((obj.rotation?.x ?? 0) * 180 / Math.PI) || 0).toFixed(2);
  if (ui.rotY) ui.rotY.value = (((obj.rotation?.y ?? 0) * 180 / Math.PI) || 0).toFixed(2);
  if (ui.rotZ) ui.rotZ.value = (((obj.rotation?.z ?? 0) * 180 / Math.PI) || 0).toFixed(2);

  if (ui.editPriority) ui.editPriority.value = String(obj.userData?.priority ?? 0);

  if (ui.materialName) {
    const matName = obj.userData?.materialName || obj.material?.name || '';
    ui.materialName.value = matName;
  }

  // Keep dropdown in sync with the current object's name (optional)
  if (ui.objectSelect && obj.name) {
    ui.objectSelect.value = obj.name;
  }
}



//==============================================================================
//=========================== Delete Button logic ==============================
//==============================================================================

export function setupDeleteHandler(context) {
  document.getElementById('deleteBtn').addEventListener('click', () => {
    const { selectedObject, scene, objects, materials } = context;
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

    // --- CLEANUP UNUSED MATERIALS ---
    if (materials) {
      // Collect names of all materials currently used by remaining objects
      const usedMaterials = new Set(objects.map(obj => obj.material?.name).filter(Boolean));

      // Remove materials from the dict that are not in use
      for (const matName of Object.keys(materials)) {
        if (!usedMaterials.has(matName)) {
          delete materials[matName];
          console.log(`Material "${matName}" removed from materials dict.`);
        }
      }
    }

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
