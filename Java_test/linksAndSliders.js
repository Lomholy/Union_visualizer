import { updateObjectList } from "./updateObjectList.js";


export function linkSliderAndInput(context, sliderId, inputId, onChange) {
    const { sliderListeners, inputListeners, objects } = context;

    const slider = document.getElementById(sliderId);
    const input = document.getElementById(inputId);

    if (sliderListeners[sliderId]) slider.removeEventListener('input', sliderListeners[sliderId]);
    if (inputListeners[inputId]) input.removeEventListener('input', inputListeners[inputId]);

    const sliderHandler = () => {
        input.value = slider.value;
        onChange(parseFloat(slider.value));
        updateObjectList(context);
    };

    const inputHandler = () => {
        slider.value = input.value;
        onChange(parseFloat(input.value));
        updateObjectList(context);
    };

    sliderListeners[sliderId] = sliderHandler;
    inputListeners[inputId] = inputHandler;

    slider.addEventListener('input', sliderHandler);
    input.addEventListener('input', inputHandler);
}
