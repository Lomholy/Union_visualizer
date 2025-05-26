

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
        fileContent += `COMPONENT ${obj.name} = ${obj.userData.type}(\n${parameters}\n) AT ${position} RELATIVE sample_arm\nROTATED ${rotation} RELATIVE sample_arm\n\n`;
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
            params += `radius=${obj.userData.radiusTop ?? obj.geometry.parameters.radiusTop ?? obj.userData.radiusBottom ?? obj.geometry.parameters.radiusBottom ?? 1}, ` +
                      `yheight=${obj.userData.height ?? obj.geometry.parameters.height ?? 1}, ` +
                      `material=${material}, priority=${priority}`;
            break;
    
        case 'Union_box':
            params += `xwidth=${obj.userData.width ?? obj.geometry.parameters.width ?? 1}, ` +
                      `yheight=${obj.userData.height ?? obj.geometry.parameters.height ?? 1}, ` +
                      `zdepth=${obj.userData.depth ?? obj.geometry.parameters.depth ?? 1}, ` +
                      `material=${material}, priority=${priority}`;
            break;
    
        case 'Union_sphere':
            params += `radius=${obj.userData.radius ?? obj.geometry.parameters.radius ?? 1}, ` +
                      `material=${material}, priority=${priority}`;
            break;
    
        case 'Union_cone':
            params += `radius_top=${obj.userData.radiusTop ?? obj.geometry.parameters.radiusTop ?? 1}, ` +
                      `radius_bottom=${obj.userData.radiusBottom ?? obj.geometry.parameters.radiusBottom ?? 1}, ` +
                      `yheight=${obj.userData.height ?? obj.geometry.parameters.height ?? 1}, ` +
                      `material=${material}, priority=${priority}`;
            break;
    
        default:
            console.warn(`Unsupported geometry type: ${obj.type}`);
            break;
    }
    return params;
}