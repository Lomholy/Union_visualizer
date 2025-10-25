import {showEditPanel} from '../ui/contextMenu.js'
// The function to update the list of objects
export function updateObjectList(context) {
  if (!context || !Array.isArray(context.objects)) {
    console.warn("updateObjectList: Invalid context or missing objects array");
    return;
  }
  //console.log("objectList element:", document.getElementById("objectList"));
  
  const list = document.getElementById("objectList");
  list.innerHTML = ""; // Clear the list
  
  context.objects.forEach(obj => {
    const li = document.createElement("li");

    const {r,g,b} = obj.material.color;
    li.style.color = `rgb(${Math.round(r*255)}, ${Math.round(g*255)}, ${Math.round(b*255)})`; // Convert rgb values to 0-255 range
    li.textContent = `${obj.name} (Priority: ${obj.userData.priority}, Material: ${obj.userData.materialName})`;
    
    // Add the data-obj-name attribute
    li.dataset.objName = obj.name;

    // Optional: make items clickable to select
    li.style.cursor = "pointer";

    li.addEventListener("click", () => {
        console.log("Clicked object:", obj);
        context.selectedObject = obj;  // Update selectedObject in context if you store it there
        document.getElementById("objectSelect").value = obj.name;
        showEditPanel(context); 
         // Highlight selected item
        [...list.children].forEach(item => item.classList.remove("selected"));
        li.classList.add("selected");
    });

    // Apply highlight if this object is already selected
    if (context.selectedObject === obj) {
      li.classList.add("selected");
    }
    
    list.appendChild(li);
  });
}

