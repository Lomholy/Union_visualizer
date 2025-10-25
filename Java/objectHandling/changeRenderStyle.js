

//==============================================================================
//========== Update of all materials when switching view styles ================
//==============================================================================

export function updateAllMaterials(context) {
  const style = document.getElementById('renderStyle').value;
  
  context.objects.forEach(obj => {
    if (!obj.isMesh) return;
    const baseColor = obj.material.color || new THREE.Color('#808080');

    let material;

    switch (style) {
      case 'solid':
        material = new THREE.MeshStandardMaterial({ color: baseColor, wireframe: false, transparent: false });
        break;

      case 'wireframe':
        material = new THREE.MeshBasicMaterial({ color: baseColor, wireframe: true });
        break;

      case 'transparent':
        material = new THREE.MeshStandardMaterial({ color: baseColor, transparent: true, opacity: 0.5 });
        break;

      case 'edges':
        // Keep solid mesh + add edges
        material = new THREE.MeshStandardMaterial({ color: baseColor });
        if (!obj.userData.edgeHelper) {
          const edgeHelper = new THREE.EdgesGeometry(obj.geometry);
          const line = new THREE.LineSegments(edgeHelper, new THREE.LineBasicMaterial({ color: 0x000000 }));
          line.name = 'edgeHelper';
          obj.add(line);
          obj.userData.edgeHelper = line;
        }
        break;

      case 'normals':
        // Visualize normals
        if (!obj.userData.normalHelper) {
          const normalHelper = new VertexNormalsHelper(obj, 0.2, 0x0000ff);
          obj.userData.normalHelper = normalHelper;
          context.scene.add(normalHelper);
        }
        break;
    }

    // Remove extra helpers if switching back
    if (style !== 'edges' && obj.userData.edgeHelper) {
      obj.remove(obj.userData.edgeHelper);
      delete obj.userData.edgeHelper;
    }

    if (style !== 'normals' && obj.userData.normalHelper) {
      context.scene.remove(obj.userData.normalHelper);
      delete obj.userData.normalHelper;
    }

    obj.material = material;
  });
  console.log(context.objects[0].userData);
}
