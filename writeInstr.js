

export function writeInstr(context){
    // Add the event listener for the download button
   
    // Prepare the content for the .instr file
    let fileContent = '';
    
    // Iterate over the objects and build the content
    context.objects.forEach(obj => {
        console.log(obj.userData);
        const parameters = buildParameters(obj);
        const position = `(${obj.position.x.toFixed(2)}, ${obj.position.y.toFixed(2)}, ${obj.position.z.toFixed(2)})`;
        const rotation = `(${obj.rotation.x.toFixed(2)}, ${obj.rotation.y.toFixed(2)}, ${obj.rotation.z.toFixed(2)})`;

        // Construct the object description
        fileContent += `COMPONENT ${obj.name} = ${obj.type}(\n${parameters}\n) AT ${position} RELATIVE sample_arm\nROTATED ${rotation} RELATIVE sample_arm\n\n`;
    });

    // Create a Blob from the content
    const blob = new Blob([fileContent], { type: 'text/plain' });

    // Create an anchor element to trigger the download
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'sample.instr';  // The file extension will be .instr

    // Trigger the download by programmatically clicking the link
    link.click();

    // Clean up the object URL after downloading
    URL.revokeObjectURL(link.href);
}

// Helper function to build the parameters string based on geometry type
function buildParameters(obj) {
    let params = '';
    // Get the material and priority from userData, defaulting to "default" for material and 0 for priority
    const material = obj.userData.material || 'default';
    const priority = obj.userData.priority || 0;

    switch (obj.userData.type) {
        case 'Union_cylinder':
            params += `radiusTop=${obj.userData.radiusTop || 1}, radiusBottom=${obj.userData.radiusBottom || 1}, height=${obj.userData.height || 1}, material=${material}, priority=${priority}`;
            break;
        case 'Union_box':
            params += `width=${obj.userData.width || 1}, height=${obj.userData.height || 1}, depth=${obj.userData.depth || 1}, material=${material}, priority=${priority}`;
            break;
        case 'Union_sphere':
            params += `radius=${obj.userData.radius || 1}, material=${material}, priority=${priority}`;
            break;
        case 'Union_cone':
            params += `radius=${obj.userData.radius || 1}, height=${obj.userData.height || 1}, material=${material}, priority=${priority}`;
            break;
        default:
            console.warn(`Unsupported geometry type: ${obj.type}`);
            break;
    }
    return params;
}