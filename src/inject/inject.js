chrome.extension.sendMessage({}, function (response) {
  var readyStateCheckInterval = setInterval(function () {
    if (document.readyState === "complete") {
      clearInterval(readyStateCheckInterval);

      const bodyEl = document.querySelector("body");

      const videoEl = document.createElement("video");
      videoEl.setAttribute("id", "videoEl");
      videoEl.setAttribute("autoplay", "true");
      bodyEl.appendChild(videoEl);

      if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices
          .getUserMedia({ video: true })
          .then(function (stream) {
            videoEl.srcObject = stream;
          })
          .catch(function (err0r) {
            console.log("Something went wrong!");
          });
      }

      videoEl.onloadeddata = async (e) => {
        console.log("Yay!", e);

        if (faceLandmarksDetection) {
          // Load the faceLandmarksDetection model assets.
          const model = await faceLandmarksDetection.load(
            faceLandmarksDetection.SupportedPackages.mediapipeFacemesh
          );

          // Pass in a video stream to the model to obtain an array of detected faces from the MediaPipe graph.
          // For Node users, the `estimateFaces` API also accepts a `tf.Tensor3D`, or an ImageData object.
          const faces = await model.estimateFaces({ input: e.target });
          console.log(faces);

          for (let i = 0; i < faces.length; i++) {
            const keypoints = faces[i].scaledMesh;

            // Log facial keypoints.
            for (let i = 0; i < keypoints.length; i++) {
              const [x, y, z] = keypoints[i];

              console.log(`Keypoint ${i}: [${x}, ${y}, ${z}]`);
            }
          }
        }
      };
    }
  }, 10);
});
