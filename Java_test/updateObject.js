
// updateObject.js
import { showEditPanel } from './editPanel.js';
import { updateObjectList } from './updateObjectList.js';


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

    updateObjectList(context);
    console.log('Updated object:', selectedObject);
  });
}

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
