import { showEditPanel } from './editPanel';


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
    
    // Optional: make items clickable to select
    li.style.cursor = "pointer";
    li.onclick = () => {
      context.selectedObject = obj;  // Update selectedObject in context if you store it there
      showEditPanel(context);
      document.getElementById("objectSelect").value = obj.name;
    };
    list.appendChild(li);
  });
}