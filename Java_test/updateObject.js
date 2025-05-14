
// updateObject.js
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
