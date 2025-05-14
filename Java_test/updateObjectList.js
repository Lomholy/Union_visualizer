import { showEditPanel } from './editPanel';


// The function to update the list of objects
export function updateObjectList(context) {
  if (!context || !Array.isArray(context.objects)) {
    console.warn("updateObjectList: Invalid context or missing objects array");
    return;
  }
  
  const list = document.getElementById("objectList");
  list.innerHTML = ""; // Clear the list
  
  context.objects.forEach(obj => {
    const li = document.createElement("li");
    li.textContent = `${obj.name} (Priority: ${obj.userData.priority}, Material: ${obj.userData.materialName || "none"})`;
    
    // Add the data-obj-name attribute
    li.dataset.objName = obj.name;

    // Optional: make items clickable to select
    li.style.cursor = "pointer";
    li.onclick = () => {
      context.selectedObject = obj;  // Update selectedObject in context if you store it there
      document.getElementById("objectSelect").value = obj.name;
      showEditPanel(context);  // Show the edit panel for the selected object
    };
    
    list.appendChild(li);
  });
}

export function updateObjectOrder(context) {
  // Sort objects by priority (ascending, lowest priority first)
  context.objects.sort((a, b) => b.userData.priority - a.userData.priority);

  // Update the renderOrder based on priority
  context.objects.forEach((obj, index) => {
    obj.renderOrder = index;  // Objects with higher priority will be rendered first
  });
}