chrome.extension.sendMessage({}, function (response) {
  var readyStateCheckInterval = setInterval(function () {
    if (document.readyState === "complete") {
      clearInterval(readyStateCheckInterval);

      // Constants
      const NUM_KEYPOINTS = 468;
      const NUM_IRIS_KEYPOINTS = 5;
      const VIDEO_WIDTH = 400;
      const VIDEO_HEIGHT = 300;
      const BLACK = "#000";
      const GREEN = "#28CF75";

      // Setup video stream
      async function setupVideoStream(videoEl) {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            width: VIDEO_WIDTH,
            height: VIDEO_HEIGHT,
          },
        });

        videoEl.srcObject = stream;

        return new Promise((resolve) => {
          videoEl.onloadeddata = () => {
            resolve(videoEl);
          };
        });
      }

      async function addHTMLElementsToDOM(container, canvasEl, videoEl) {
        // Assign ids
        container.setAttribute("id", "container");
        canvasEl.setAttribute("id", "canvasEl");
        videoEl.setAttribute("id", "videoEl");

        // Add elements to DOM
        document.querySelector("body").appendChild(container);
        container.appendChild(canvasEl);
        container.appendChild(videoEl);
      }

      function distance(a, b) {
        return Math.sqrt(Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2));
      }

      async function renderPrediction(ctx, model) {
        const predictions = await model.estimateFaces({
          input: videoEl,
          returnTensors: false,
          flipHorizontal: false,
          predictIrises: true,
        });

        ctx.drawImage(
          videoEl,
          0,
          0,
          videoEl.width,
          videoEl.height,
          0,
          0,
          canvasEl.width,
          canvasEl.height
        );

        if (predictions.length > 0) {
          predictions.forEach((prediction) => {
            const keypoints = prediction.scaledMesh;

            // Draw dots
            ctx.fillStyle = BLACK;

            for (let i = 0; i < NUM_KEYPOINTS; i++) {
              const x = keypoints[i][0];
              const y = keypoints[i][1];

              ctx.beginPath();
              ctx.arc(x, y, 1 /* radius */, 0, 2 * Math.PI);
              ctx.fill();
            }

            if (keypoints.length > NUM_KEYPOINTS) {
              ctx.strokeStyle = GREEN;
              ctx.lineWidth = 1;

              const leftCenter = keypoints[NUM_KEYPOINTS];
              const leftDiameterY = distance(
                keypoints[NUM_KEYPOINTS + 4],
                keypoints[NUM_KEYPOINTS + 2]
              );
              const leftDiameterX = distance(
                keypoints[NUM_KEYPOINTS + 3],
                keypoints[NUM_KEYPOINTS + 1]
              );

              ctx.beginPath();
              ctx.ellipse(
                leftCenter[0],
                leftCenter[1],
                leftDiameterX / 2,
                leftDiameterY / 2,
                0,
                0,
                2 * Math.PI
              );

              ctx.stroke();

              if (keypoints.length > NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS) {
                const rightCenter =
                  keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS];
                const rightDiameterY = distance(
                  keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 2],
                  keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 4]
                );
                const rightDiameterX = distance(
                  keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 3],
                  keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 1]
                );

                ctx.beginPath();
                ctx.ellipse(
                  rightCenter[0],
                  rightCenter[1],
                  rightDiameterX / 2,
                  rightDiameterY / 2,
                  0,
                  0,
                  2 * Math.PI
                );
                ctx.stroke();
              }
            }
          });
        }

        requestAnimationFrame(() => {
          renderPrediction(ctx, model);
        });
      }

      async function main() {
        const container = document.createElement("div");
        const canvasEl = document.createElement("canvas");
        const videoEl = document.createElement("video");

        await addHTMLElementsToDOM(container, canvasEl, videoEl);
        await setupVideoStream(videoEl);

        videoEl.play();
        videoEl.width = VIDEO_WIDTH;
        videoEl.height = VIDEO_HEIGHT;
        canvasEl.width = VIDEO_WIDTH;
        canvasEl.height = VIDEO_HEIGHT;

        const ctx = canvasEl.getContext("2d");
        ctx.translate(canvasEl.width, 0);
        ctx.scale(-1, 1);
        ctx.fillStyle = BLACK;
        ctx.strokeStyle = BLACK;
        ctx.lineWidth = 0.25;

        const model = await faceLandmarksDetection.load(
          faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
          { maxFaces: 1 }
        );

        renderPrediction(ctx, model);
      }

      main();
    }
  }, 10);
});
