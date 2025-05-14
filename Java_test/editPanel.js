import { linkSliderAndInput } from './linksAndSliders.js'; // Import if it's now separated

export function showEditPanel(context) {
    
    document.getElementById('editPanel').style.display = 'block';

    document.getElementById('objectName').value = context.selectedObject.name || '';

    document.getElementById('posX').value = context.selectedObject.position.x.toFixed(2);
    document.getElementById('posY').value = context.selectedObject.position.y.toFixed(2);
    document.getElementById('posZ').value = context.selectedObject.position.z.toFixed(2);

    document.getElementById('posXSlider').value = context.selectedObject.position.x;
    document.getElementById('posYSlider').value = context.selectedObject.position.y;
    document.getElementById('posZSlider').value = context.selectedObject.position.z;

    // Sync sliders and inputs live
    linkSliderAndInput(context, 'posXSlider', 'posX', val => context.selectedObject.position.x = val);
    linkSliderAndInput(context, 'posYSlider', 'posY', val => context.selectedObject.position.y = val);
    linkSliderAndInput(context, 'posZSlider', 'posZ', val => context.selectedObject.position.z = val);

    document.getElementById('rotX').value = context.selectedObject.rotation.x.toFixed(2);
    document.getElementById('rotY').value = context.selectedObject.rotation.y.toFixed(2);
    document.getElementById('rotZ').value = context.selectedObject.rotation.z.toFixed(2);

    document.getElementById('rotXSlider').value = context.selectedObject.rotation.x;
    document.getElementById('rotYSlider').value = context.selectedObject.rotation.y;
    document.getElementById('rotZSlider').value = context.selectedObject.rotation.z;

    // Sync sliders and inputs for rotation
    linkSliderAndInput(context, 'rotXSlider', 'rotX', val => context.selectedObject.rotation.x = val);
    linkSliderAndInput(context, 'rotYSlider', 'rotY', val => context.selectedObject.rotation.y = val);
    linkSliderAndInput(context, 'rotZSlider', 'rotZ', val => context.selectedObject.rotation.z = val);

    document.getElementById('editPriority').value = context.selectedObject.userData.priority || 0;
    document.getElementById('materialName').value = context.selectedObject.userData.materialName || '';
}
