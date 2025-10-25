

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
    } else {
      document.getElementById('editPanel').style.display = 'none'; // Hide panel if nothing left
    }

    console.log("Object deleted.");
  });
}

