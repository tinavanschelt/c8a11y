chrome.extension.sendMessage({}, function (response) {
  var readyStateCheckInterval = setInterval(function () {
    if (document.readyState === "complete") {
      clearInterval(readyStateCheckInterval);

      let firstLoad = true;
      let shouldRenderOverlay = false;
      let isAlreadyTraining = false;
      let shouldPredict = true;

      const bodyEl = document.querySelector("body");
      let videoEl;
      let canvasEl;
      let overlayEl;

      const initialDotCountX = 8;
      const initialDotCountY = 6;
      const additionalDotCountX = 7;
      const additionalDotCountY = 5;
      const totalDataPointsForTraining =
        initialDotCountX +
        initialDotCountY +
        additionalDotCountX * additionalDotCountY;

      const dataSet = [];
      const inputsMin = [];
      const inputsMax = [];

      const TOTAL_EPOCHS = 800;
      const VIDEO_WIDTH = 220;
      const VIDEO_HEIGHT = 180;
      const BLACK = "#000";
      const GREEN = "#28CF75";
      const RED = "#FF0000";

      const model = tf.sequential();

      // MOUSE MOVEMENT + HIGHLIGHTER

      function getMousePosition(e) {
        return { x: e.clientX, y: e.clientY };
      }

      function resetMouseHighlighter() {
        const prevTarget = document.querySelector(".c8a11y-mouse-target");

        if (prevTarget) {
          bodyEl.removeChild(prevTarget);
        }
      }

      function mouseHighlighter(e) {
        resetMouseHighlighter();
        const mouseCoords = getMousePosition(e);
        const mouseTarget = document.createElement("div");
        mouseTarget.classList.add("c8a11y-mouse-target");
        mouseTarget.style.left = `${mouseCoords.x - 12}px`;
        mouseTarget.style.top = `${mouseCoords.y - 12}px`;
        bodyEl.appendChild(mouseTarget);
      }

      function initMouseHighlighter() {
        bodyEl.addEventListener("mousemove", mouseHighlighter);
      }

      function destroyMouseHighlighter() {
        resetMouseHighlighter();
        bodyEl.removeEventListener("mousemove", mouseHighlighter);
      }

      // CALCULATIONS

      function distance(a, b) {
        return Math.sqrt(Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2));
      }

      async function getFacialKeypointsPrediction(videoEl) {
        const faceLandmarksModel = await faceLandmarksDetection.load(
          faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
          { maxFaces: 1 }
        );

        const predictions = await faceLandmarksModel.estimateFaces({
          input: videoEl,
          returnTensors: false,
          flipHorizontal: false,
          predictIrises: true,
        });

        return predictions;
      }

      async function getCurrentCoordinates() {
        const predictions = await getFacialKeypointsPrediction(videoEl);
        const keypoints = predictions[0].annotations;

        const leftIrisCenter = keypoints.leftEyeIris[0];
        const rightIrisCenter = keypoints.rightEyeIris[0];
        const leftIrisCenterXY = [leftIrisCenter[0], leftIrisCenter[1]];
        const rightIrisCenterXY = [rightIrisCenter[0], rightIrisCenter[1]];

        const leftEyeUpper = keypoints["leftEyeUpper0"][0];
        const leftViewportXY = [leftEyeUpper[0], leftEyeUpper[1]];

        return {
          leftIrisCenterXY,
          rightIrisCenterXY,
          leftViewportXY,
        };
      }

      async function calculateDistances(coordinates) {
        const { leftIrisCenterXY, rightIrisCenterXY, leftViewportXY } =
          coordinates;

        const leftIrisXDistance = leftViewportXY[0] - leftIrisCenterXY[0];
        const leftIrisYDistance = leftViewportXY[1] - leftIrisCenterXY[1];
        const rightIrisXDistance = leftViewportXY[0] - rightIrisCenterXY[0];
        const rightIrisYDistance = leftViewportXY[1] - rightIrisCenterXY[1];

        return [
          leftIrisXDistance,
          leftIrisYDistance,
          rightIrisXDistance,
          rightIrisYDistance,
        ];
      }

      function calculateMinMaxValues(
        arrayOfValues,
        minValueInArray,
        maxValueInArray
      ) {
        const axis1 = arrayOfValues.map((i) => i[0]);
        const axis2 = arrayOfValues.map((i) => i[1]);
        const axis3 = arrayOfValues.map((i) => i[2]);
        const axis4 = arrayOfValues.map((i) => i[3]);

        minValueInArray.push(Math.min(...axis1));
        minValueInArray.push(Math.min(...axis2));
        minValueInArray.push(Math.min(...axis3));
        minValueInArray.push(Math.min(...axis4));

        maxValueInArray.push(Math.max(...axis1));
        maxValueInArray.push(Math.max(...axis2));
        maxValueInArray.push(Math.max(...axis3));
        maxValueInArray.push(Math.max(...axis4));
      }

      function normaliseInputs(inputs) {
        calculateMinMaxValues(inputs, inputsMin, inputsMax);

        return inputs.map((input) => {
          return input.map(
            (value, i) => (value - inputsMin[i]) / (inputsMax[i] - inputsMin[i])
          );
        });
      }

      async function cleanupData(data) {
        const inputs = [];
        const outputs = [];

        data.map((set) => {
          inputs.push(set.feature);
          outputs.push(set.label);
        });

        return {
          inputs: normaliseInputs(inputs),
          outputs: outputs,
        };
      }

      // INFO BANNER
      function initInfoBanner() {
        const infoBannerEl = document.createElement("div");
        infoBannerEl.classList.add("c8a11y-info-banner", "active");
        infoBannerEl.innerHTML =
          "Please wait for video stream to load. Video stream is loading...";
        bodyEl.appendChild(infoBannerEl);
      }

      function clearInfoBanner() {
        document
          .querySelector(".c8a11y-info-banner")
          .classList.remove("active");
        document.querySelector(".c8a11y-info-banner").innerHTML = "";
      }

      function updateInfoBanner(text) {
        document.querySelector(".c8a11y-info-banner").classList.add("active");
        document.querySelector(".c8a11y-info-banner").innerHTML = text;
      }

      function cleanupAnswerX(x) {
        if (x > bodyEl.clientWidth) {
          return bodyEl.clientWidth - 10;
        } else if (x < 0) {
          return 10;
        }

        return x;
      }

      function cleanupAnswerY(y) {
        if (y < 0) {
          return 10;
        }

        return y;
      }

      async function initModelPrediction() {
        const coordinates = await getCurrentCoordinates();
        const distances = await calculateDistances(coordinates);

        const testInput = distances.map(
          (value, i) => (value - inputsMin[i]) / (inputsMax[i] - inputsMin[i])
        );

        const inputTensor = tf.tensor2d([testInput]);
        const answer = model.predict(inputTensor);
        const answerAsArray = answer.dataSync();

        const xPrediction = cleanupAnswerX(answerAsArray[0]);
        const yPrediction = cleanupAnswerY(answerAsArray[1]);

        const predictedIrisTarget = document.createElement("div");
        predictedIrisTarget.classList.add("c8a11y-predicted-iris-target");
        predictedIrisTarget.style.left = `${xPrediction}px`;
        predictedIrisTarget.style.top = `${yPrediction}px`;
        overlayEl.appendChild(predictedIrisTarget);

        // cleanup
        answer.dispose();
        if (shouldPredict) {
          setTimeout(initModelPrediction, 100);
        }
      }

      function updateToggleButtonText(text) {
        document.querySelector(".c8a11y-predict-toggle-button").innerHTML =
          text;
      }

      async function toggleModelPrediction() {
        if (shouldPredict) {
          shouldPredict = false;
          updateToggleButtonText("Turn prediction on");
        } else {
          shouldPredict = true;
          updateToggleButtonText("Turn prediction off");
          initModelPrediction();
        }
      }

      function initPredictToggleButton() {
        const toggleButton = document.createElement("button");
        toggleButton.classList.add("c8a11y-predict-toggle-button");
        toggleButton.innerHTML = "Turn prediction off";
        toggleButton.addEventListener("click", toggleModelPrediction);
        bodyEl.querySelector(".c8a11y-info-banner").appendChild(toggleButton);
      }

      function updateProgress(progressAsPercentage) {
        bodyEl.querySelector(".c8a11y-progress").style.width = `${progressAsPercentage}%`;  
      }

      async function train() {
        // Unlock scroll on body
        bodyEl.style.overflow = 'auto';
        updateInfoBanner(
          "Training model...<div class='c8a11y-progress-bar'><div class='c8a11y-progress'></div></div>"
        );
        const { inputs, outputs } = await cleanupData(dataSet);
        isAlreadyTraining = true;

        const xs = tf.tensor2d(inputs);
        const ys = tf.tensor2d(outputs);

        const hiddenLayer = tf.layers.dense({
          activation: "relu",
          inputShape: [4],
          units: inputs.length,
        });

        const outputLayer = tf.layers.dense({
          units: 2,
        });

        model.add(hiddenLayer);
        model.add(outputLayer);

        model.compile({
          optimizer: tf.train.sgd(0.0001), // adam
          loss: "meanSquaredError",
        });

        model.summary();

        const printCallback = {
          onEpochEnd: (epoch, log) => {
            updateProgress(epoch / TOTAL_EPOCHS * 100);
            console.log(epoch, log);
          },
        };

        // train model
        model
          .fit(xs, ys, {
            epochs: TOTAL_EPOCHS,
            callbacks: printCallback,
            batchSize: 10,
          })
          .then((history) => {
            console.log("Training complete");
            updateInfoBanner(
              "Look around the screen to predict. You can toggle predictions on or off using the button 👉"
            );
            shouldPredict = true;
            initPredictToggleButton();
            initModelPrediction();
          });
      }

      function initInfoModal(
        innerHTML,
        confirmText,
        onConfirmation,
        dismissText,
        onDismissal
      ) {
        destroyMouseHighlighter();
        const infoModalEl = document.createElement("div");
        infoModalEl.classList.add("c8a11y-info-modal");
        infoModalEl.innerHTML = innerHTML;
        bodyEl.appendChild(infoModalEl);

        if (dismissText !== undefined) {
          const dismissButton = document.createElement("button");
          dismissButton.classList.add("c8a11y-info-modal-button", "dismiss");
          dismissButton.innerHTML = dismissText;
          dismissButton.addEventListener("click", function () {
            infoModalEl.remove();
            onDismissal();
          });

          infoModalEl.appendChild(dismissButton);
        }

        if (confirmText !== null) {
          const confirmButton = document.createElement("button");
          confirmButton.classList.add("c8a11y-info-modal-button", "confirm");
          confirmButton.innerHTML = confirmText;
          confirmButton.addEventListener("click", function () {
            infoModalEl.remove();
            onConfirmation();
          });

          infoModalEl.appendChild(confirmButton);
        }
      }

      function createDot(classes) {
        const dotEl = document.createElement("div");
        dotEl.classList.add(...classes);
        overlayEl.appendChild(dotEl);
        return dotEl;
      }

      function removeDot(e, dotEl) {
        dotEl.classList.add("animate");

        setTimeout(function () {
          overlayEl.removeChild(e.target);
        }, 300);
      }

      async function addAdditionalCalibrationDots() {
        initMouseHighlighter();

        const dots = [...Array(additionalDotCountX)]
          .map((_, xi) => {
            return [...Array(additionalDotCountY)].map((_, yi) => {
              return [`x${xi + 1}`, `y${yi + 1}`];
            });
          })
          .flat();

        dots.map((dot) => {
          const dotEl = createDot([
            "c8a11y-dot",
            "c8a11y-additional-dot",
            dot[0],
            dot[1],
          ]);

          dotEl.addEventListener("click", async function (e) {
            removeDot(e, dotEl);

            setTimeout(function () {
              if (
                !overlayEl.hasChildNodes() &&
                dataSet.length >= totalDataPointsForTraining &&
                !isAlreadyTraining
              ) {
                train();
              }
            }, 500);
          });
        });
      }

      async function addInitialCalibrationDots() {
        initMouseHighlighter();

        const dots = [...Array(initialDotCountX)]
          .map((_, xi) => {
            return [...Array(initialDotCountY)].map((_, yi) => {
              return [`x${xi + 1}`, `y${yi + 1}`];
            });
          })
          .flat();

        dots.map((dot) => {
          const dotEl = createDot([
            "c8a11y-dot",
            "c8a11y-initial-dot",
            dot[0],
            dot[1],
          ]);

          dotEl.addEventListener("click", async function (e) {
            removeDot(e, dotEl);

            setTimeout(function () {
              if (!overlayEl.hasChildNodes() && !isAlreadyTraining) {
                initInfoModal(
                  `<p>Great! You've collected ${dataSet.length} data points. For a more accurate prediction we recommend doing one more round of data collection before training your model.</p>`,
                  "OK!",
                  addAdditionalCalibrationDots,
                  "Train model using current data set",
                  train
                );
              }
            }, 500);
          });
        });
      }

      function initDataCollection() {
        bodyEl.addEventListener("click", async function (e) {
          const mousecoords = getMousePosition(e);
          const coordinates = await getCurrentCoordinates();
          const distances = await calculateDistances(coordinates);

          dataSet.push({
            feature: distances,
            label: [mousecoords.x, mousecoords.y],
          });
        });
      }

      function setShouldRenderOverlay(val) {
        shouldRenderOverlay = val;
      }

      function initOverlay() {
        overlayEl = document.createElement("div");
        overlayEl.classList.add("c8a11y-overlay");
        // Lock scroll on body
        bodyEl.style.overflow = 'hidden';
        bodyEl.appendChild(overlayEl);

        initDataCollection();
        addInitialCalibrationDots();
      }

      async function initVideoStream() {
        videoEl = document.createElement("video");
        canvasEl = document.createElement("canvas");
        const containerEl = document.createElement("div");

        containerEl.classList.add("c8a11y-stream-container");
        canvasEl.classList.add("c8a11y-canvas");
        videoEl.classList.add("c8a11y-video");

        bodyEl.appendChild(containerEl);
        containerEl.appendChild(canvasEl);
        containerEl.appendChild(videoEl);

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
            resolve();
          };
        });
      }

      function initCanvas(canvasEl) {
        const ctx = canvasEl.getContext("2d");
        ctx.translate(canvasEl.width, 0);
        ctx.scale(-1, 1);
        return ctx;
      }

      async function renderKeypoints(ctx, canvasEl, videoEl) {
        const predictions = await getFacialKeypointsPrediction(videoEl);

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
          const keypoints = predictions[0].annotations;

          const leftEyeEdgeKeypoint = {
            x: keypoints["leftEyeUpper0"][0][0],
            y: keypoints["leftEyeUpper0"][0][1],
          };
          const rightEyeEdgeKeypoint = {
            x: keypoints["rightEyeUpper0"][0][0],
            y: keypoints["rightEyeUpper0"][0][1],
          };

          const a = leftEyeEdgeKeypoint.x - rightEyeEdgeKeypoint.x;
          const b = leftEyeEdgeKeypoint.y - rightEyeEdgeKeypoint.y;
          const distance = Math.sqrt(a * a + b * b);

          // Draw "peephole" frame
          const padding = distance * 0.2; // add 20% x-padding
          const frameWidth = distance + padding;
          const frameHeight = distance / 3;
          const x = leftEyeEdgeKeypoint.x + padding / 2; // The x-coordinate of the upper-left corner
          const y = leftEyeEdgeKeypoint.y - frameHeight / 2; // The y-coordinate of the upper-left corner
          ctx.strokeStyle = BLACK;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.rect(x, y, -frameWidth, frameHeight);
          ctx.stroke();

          // Draw edge points
          ctx.fillStyle = GREEN;
          ctx.lineWidth = 2;
          // x1, y1
          ctx.beginPath();
          ctx.arc(x, leftEyeEdgeKeypoint.y, 1, 0, 2 * Math.PI);
          ctx.fill();
          // x2, y2
          ctx.beginPath();
          ctx.arc(x - frameWidth, leftEyeEdgeKeypoint.y, 1, 0, 2 * Math.PI);
          ctx.fill();

          // Draw irises
          const irisKeypoints = ["leftEyeIris", "rightEyeIris"];

          irisKeypoints.map((point) => {
            const iris = keypoints[point];
            ctx.fillStyle = RED;
            ctx.lineWidth = 0.25;

            iris.map((p) => {
              ctx.beginPath();
              ctx.arc(p[0], p[1], 1 /* radius */, 0, 2 * Math.PI);
              ctx.fill();
            });
          });

          ctx.fill();

          if (firstLoad) {
            firstLoad = false;
            clearInfoBanner();
          }
        }

        requestAnimationFrame(() => {
          renderKeypoints(ctx, canvasEl, videoEl);
        });
      }

      async function main() {
        initInfoBanner();
        initInfoModal(
          "<p>You’ve successfully activated <i>c8a11y</i>!</p><p><i>c8a11y</i> is an application that determines where you are looking in the browser, but first, the application needs to be trained. To do so, simply <ol><li>Ensure <b>access to your camera is enabled</b></li><li>Click away all of the <div class='c8a11y-dot inline'></div> green dots on the screen</li><li>Keep looking at the <div class='c8a11y-mouse-target inline'></div> bright blue dot whilst moving the cursor around</li></ol>Ready?</p>",
          "Let's go!",
          setShouldRenderOverlay(true)
        );

        await initVideoStream();

        if (videoEl && canvasEl) {
          videoEl.play();
          videoEl.width = VIDEO_WIDTH;
          videoEl.height = VIDEO_HEIGHT;
          canvasEl.width = VIDEO_WIDTH;
          canvasEl.height = VIDEO_HEIGHT;

          const ctx = initCanvas(canvasEl);
          await renderKeypoints(ctx, canvasEl, videoEl);
          if (shouldRenderOverlay) {
            initOverlay();
          }
        } else {
          // throw error
        }
      }

      main();
    }
  }, 10);
});
