// Update the UI if materials were added, removed, or replaced
export function updateMaterialList(context) {
  const newEntries = Object.entries(context.materials);
  const oldEntries = context.old_materials || {};

  let changed = false;

  // Compare keys and object references
  if (newEntries.length !== Object.keys(oldEntries).length) {
    changed = true;
  } else {
    for (const [name, mat] of newEntries) {
      if (!(name in oldEntries) || oldEntries[name] !== mat) {
        changed = true;
        break;
      }
    }
  }

  if (changed) {
    console.log('Materials changed');
    populateMaterialList(context);

    // Save snapshot of references
    context.old_materials = { ...context.materials };
  }
}

// Populate the list from context.materials
function populateMaterialList(context) {
  const materialListEl = document.getElementById('materialList');
  materialListEl.innerHTML = '';

  Object.entries(context.materials).forEach(([name, mat]) => {
    const li = document.createElement('li');
    li.className = 'material-item';
    li.textContent = name;

    const colorInput = document.createElement('input');
    colorInput.type = 'color';

    // Use the material's current color
    if (mat.color) {
      colorInput.value = `#${mat.color.getHexString()}`;
    } else {
      colorInput.value = '#ffffff';
      colorInput.disabled = true;
    }

    // When the user changes the color
    colorInput.addEventListener('input', (e) => {
      if (mat.color) {
        mat.color.set(e.target.value); // update the THREE.Material color
        context.renderer.render(context.scene, context.camera); // redraw scene
      }
    });

    li.appendChild(colorInput);
    materialListEl.appendChild(li);
  });
}
