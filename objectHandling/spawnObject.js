
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
import * as THREE from 'three'



export function spawnObject(context, spec) {
  const { scene, objects } = context;
  if (!spec || !spec.type) {
    throw new Error('spawnObject: spec.type is required (e.g., "Union_box").');
  }

  const type = spec.type;
  const materialName = spec.materialName ?? 'default';
  const params = spec.params;
  // Ensure material exists (color stable across uses)
  ensureMaterial(context, materialName);

  // --- Build geometry and normalize params for userData/export ---
  let geometry;
  /** @type {object} */

  switch (type) {
    case 'Union_box': {
      geometry = new THREE.BoxGeometry(params.xwidth, 
            params.yheight, 
            params.zdepth);
      break;
    }
    case 'Union_sphere': {
      geometry = new THREE.SphereGeometry(params.radius);
      break;
    }
    case 'Union_cylinder': {
      geometry = new THREE.CylinderGeometry(params.radius, 
            params.radius, 
            params.yheight);
      break;
    }
    case 'Union_cone': {
      geometry = new THREE.CylinderGeometry(
            params.radiusTop, params.radiusBottom, params.yheight);
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
  
   let highestPriority = -Infinity;
   let highestObject = null;
   // Find highest priority and assign that as default if no priority is set.
   for (let i = 0; i < objects.length; i++) {
     if (objects[i].priority > highestPriority) {
       highestPriority = objects[i].priority;
     }
   }


  mesh.name = name;

  // --- Metadata expected by writeInstr/buildParameters ---
  // Keep keys compatible with your export code (width/height/depth, radius,
  // radiusTop/radiusBottom, height, materialName, priority, type, etc.)
  mesh.userData = {
    priority: spec.priority ?? highestPriority + 1,
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
  console.log(context.objects);
  // --- UI: add to dropdown, select, and refresh list/panel ---
  addToObjectSelect(mesh);
  context.selectedObject = mesh;

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


