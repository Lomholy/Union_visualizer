import * as THREE from 'three';

import { showEditPanel } from "./editPanel";

export function onMouseClick(event, context) {
  const { renderer, mouse, camera, scene, raycaster } = context;

  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(scene.children);
  // ✅ Only update if we hit a valid (non-helper) object
  const hit = intersects.find(obj => !(obj.object instanceof THREE.AxesHelper));
  if (hit) {
    if (intersects.length > 0) {
      context.selectedObject = intersects[0].object;
      document.getElementById("objectSelect").value = context.selectedObject.name;
      showEditPanel(context);
    }
  }
}