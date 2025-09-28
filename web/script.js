const video = document.getElementById("webcam");
const snapBtn = document.getElementById("snap");
const binLabel = document.getElementById("bin");
const factLabel = document.getElementById("fact");

// Start webcam
navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => { video.srcObject = stream; })
.catch(err => {
    console.error("Error accessing webcam: ", err);
    alert("Cannot access webcam. Please allow camera access.");
});

// Capture image and show placeholder
snapBtn.onclick = () => {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    // Placeholder: show dummy result
    binLabel.innerText = "Put this item in the PLASTIC bin";
    factLabel.innerText = "♻️ Fun Fact: Recycling one plastic bottle saves enough energy to power a light bulb for 3 hours!";
};
