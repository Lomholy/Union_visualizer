
export function updateObjectOrder(context) {
  // Sort objects by priority (ascending, lowest priority first)
  context.objects.sort((a, b) => a.userData.priority - b.userData.priority);

  // Update the renderOrder based on priority
  context.objects.forEach((obj, index) => {
    obj.renderOrder =context.objects.length -index;  // Objects with higher priority will be rendered first
  });
}
