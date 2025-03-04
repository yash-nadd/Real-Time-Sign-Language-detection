function switchModel(model) {
    fetch('/switch_model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: `model=${model}`
    }).then(() => {
        // Update the UI to reflect the change while keeping the style
        const modelText = document.getElementById("currentModel").querySelector("span");
        modelText.textContent = model;
        modelText.className = "badge bg-success"; // Ensure style remains
    });
}
