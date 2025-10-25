
//==============================================================================
//=========================== Update selected object============================
/* In main.js when the scene is made, setupLiveUpdateHandlers is called once.
 * After that, setSelectedObject is set whenever a new object is selected.
 * This is done in main as an event listener to objectSelect.*/
//==============================================================================
import * as THREE from 'three'


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


