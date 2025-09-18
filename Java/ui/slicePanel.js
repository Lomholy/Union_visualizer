import * as THREE from 'three';



export function init_slicePanel(context){// Add slice panel 
    // Assuming you already have these references
    const toggleBtn = document.getElementById('toggleSlicePanel');
    const slicePanel = document.getElementById('slicePanel');
    const planeHeightSlider = document.getElementById('planeHeight');
    // Track if the plane has been added
    let xyPlane = null;

    toggleBtn.addEventListener('click', () => {
    // Toggle panel visibility
    slicePanel.style.display = (slicePanel.style.display === 'none') ? 'block' : 'none';

    // Add XY plane if it doesn't exist
    if (!xyPlane) {
        // Create a plane
        const size = 20; // adjust as needed
        const geometry = new THREE.PlaneGeometry(size, size);
        const material = new THREE.MeshStandardMaterial({
        color: 0xaaaaaa,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.5
        });
        xyPlane = new THREE.Mesh(geometry, material);

        xyPlane.rotation.x = -Math.PI / 2; // Make it XY plane
        xyPlane.position.set(0, 0, 0);
    } else {
        xyPlane.visible = !xyPlane.visible; // toggle visibility
    }
        // Make sure 'scene' is in scope
    if (typeof context.scene !== 'undefined') {
        context.scene.add(xyPlane);
        console.log('XY plane added to scene');
    } else {
        console.warn('scene is not defined');
    }
    });
    // Move plane up/down when slider changes
    planeHeightSlider.addEventListener('input', () => {
        if (xyPlane) {
            xyPlane.position.y = parseFloat(planeHeightSlider.value);
        }
    });
}

