import * as THREE from 'three';
import { showEditPanel } from "./editPanel";

export function onMouseClick(event, context) {
  const { renderer, mouse, camera, scene, raycaster } = context;

  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(scene.children, true);

  // ✅ Only use the first valid (non-AxesHelper) hit
  const hit = intersects.find(obj => !(obj.object instanceof THREE.AxesHelper));

  if (hit) {
    context.selectedObject = hit.object;
    document.getElementById("objectSelect").value = context.selectedObject.name;
    showEditPanel(context);
  }
}