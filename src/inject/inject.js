chrome.extension.sendMessage({}, function (response) {
  var readyStateCheckInterval = setInterval(function () {
    if (document.readyState === "complete") {
      clearInterval(readyStateCheckInterval);

      /* Define global application state */
      const state = {
        isFirstLoad: true,
        isAlreadyTraining: false,
        shouldRenderOverlay: false,
        isPredictMode: true,
        isTestMode: false,
      };

      /* Define global DOM elements */
      const bodyEl = document.querySelector("body");
      let videoEl;
      let canvasEl;
      let overlayEl;
      let testModeOverlayEl;

      /* Specify number of dots used for calibration */
      const dotCount = {
        x: 6,
        y: 4,
      };

      /* Define global data variables */
      const dataSet = [];
      const inputsMin = [];
      const inputsMax = [];

      /* Declare constants */
      const TOTAL_EPOCHS = 800;
      const VIDEO_WIDTH = 220;
      const VIDEO_HEIGHT = 180;
      const BLACK = "#000";
      const GREEN = "#28CF75";
      const RED = "#F44336";

      /* Initialise the tensorflow model */
      const model = tf.sequential();

      /* SECTION: MOUSE MOVEMENT AND HIGHLIGHTING */

      /* Get the current co-ordinates of the cursor */
      function getMousePosition(e) {
        return { x: e.clientX, y: e.clientY };
      }

      /* Clear (or reset) the existing mouse position */
      function resetMouseHighlighter() {
        const prevTarget = document.querySelector(".c8a11y-mouse-target");

        if (prevTarget) {
          bodyEl.removeChild(prevTarget);
        }
      }

      /* Highlight the mouse by rendering a div (circular) at the current position of the mouse on the screen */
      function mouseHighlighter(e) {
        resetMouseHighlighter(); // Always clear the existing position first
        const mouseCoords = getMousePosition(e);
        const mouseTarget = document.createElement("div");
        mouseTarget.classList.add("c8a11y-mouse-target");
        mouseTarget.style.left = `${mouseCoords.x - 12}px`;
        mouseTarget.style.top = `${mouseCoords.y - 12}px`;
        bodyEl.appendChild(mouseTarget); // Render a circle at the current mouse co-ordinates
      }

      /* Initialise the mouse highlighter */
      function initMouseHighlighter() {
        bodyEl.addEventListener("mousemove", mouseHighlighter);
      }

      /* Disable mouse highlighter */
      function destroyMouseHighlighter() {
        resetMouseHighlighter();
        bodyEl.removeEventListener("mousemove", mouseHighlighter);
      }

      /* SECTION: CALCULATIONS */

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

      /* SECTION: DATA NORMALISATION */

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

      async function splitDataSets(inputs, outputs) {
        const lengthOfTestData = Math.round(inputs.length * 0.15);
        const testIndexes = [];
        while (testIndexes.length < lengthOfTestData) {
          const i = Math.floor(Math.random() * inputs.length) + 1;
          if (testIndexes.indexOf(i) === -1) {
            testIndexes.push(i);
          }
        }

        const testInputs = testIndexes.map((i) => inputs[i]);
        const testOutputs = testIndexes.map((i) => outputs[i]);

        testIndexes.map((i) => {
          testInputs.push(inputs[i]);
          testOutputs.push(outputs[i]);
          inputs[i] = null;
          outputs[i] = null;
        });

        const trainInputs = inputs.filter((val) => val !== null);
        const trainOutputs = outputs.filter((val) => val !== null);

        return { trainInputs, trainOutputs, testInputs, testOutputs };
      }

      /* SECTION: POPUP INFO MODAL */

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

      /* SECTION: TOP INFO BANNER */

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

      /* SECTION: PREDICTION MODE */

      function cleanupOutputX(x) {
        if (x > bodyEl.clientWidth) {
          return bodyEl.clientWidth - 10;
        } else if (x < 0) {
          return 10;
        }

        return x;
      }

      function cleanupOutputY(y) {
        if (y < 0) {
          return 10;
        }

        return y;
      }

      async function getCurrentGazePrediction() {
        const coordinates = await getCurrentCoordinates();
        const distances = await calculateDistances(coordinates);

        const input = distances.map(
          (value, i) => (value - inputsMin[i]) / (inputsMax[i] - inputsMin[i])
        );

        const { xPrediction, yPrediction } = tf.tidy(() => {
          const inputTensor = tf.tensor2d([input]);
          const output = model.predict(inputTensor);
          const outputAsArray = output.dataSync();

          inputTensor.dispose();
          output.dispose();

          return {
            xPrediction: cleanupOutputX(outputAsArray[0]),
            yPrediction: cleanupOutputY(outputAsArray[1]),
          };
        });

        return { xPrediction, yPrediction };
      }

      async function initPredictMode() {
        const { xPrediction, yPrediction } = await getCurrentGazePrediction();

        const predictedIrisTarget = document.createElement("div");
        predictedIrisTarget.classList.add("c8a11y-predicted-iris-target");
        predictedIrisTarget.style.left = `${xPrediction}px`;
        predictedIrisTarget.style.top = `${yPrediction}px`;
        overlayEl.appendChild(predictedIrisTarget);

        if (state.isPredictMode) {
          setTimeout(initPredictMode, 50);
        }
      }

      /* SECTION: TEST MODE */

      /* Clear (or reset) the existing highlighted test block */
      function resetTestBlock() {
        const prevTestBlock = testModeOverlayEl.querySelector(".active");

        if (prevTestBlock) {
          prevTestBlock.classList.remove("active");
        }
      }

      async function highlightTestBlock() {
        const { xPrediction, yPrediction } = await getCurrentGazePrediction();

        const blockWidth =
          testModeOverlayEl.querySelector(".c8a11y-test-block").offsetWidth;
        const blockHeight =
          testModeOverlayEl.querySelector(".c8a11y-test-block").offsetHeight;

        let xBlock = 0;
        let yBlock = 0;

        for (let i = 1; i < 5; i++) {
          if (xPrediction < blockWidth * i) {
            xBlock = i;
            break;
          }
        }

        for (let i = 1; i < 5; i++) {
          if (yPrediction < blockHeight * i) {
            yBlock = i;
            break;
          }
        }

        if (xBlock !== 0 && yBlock !== 0) {
          resetTestBlock();
          const activeBlock = testModeOverlayEl.querySelector(
            `.c8a11y-test-block-x${xBlock}-y${yBlock}`
          );
          activeBlock.classList.add("active");
        }

        if (state.isTestMode) {
          setTimeout(highlightTestBlock, 50);
        }
      }

      async function initTestMode() {
        testModeOverlayEl = document.createElement("div");
        testModeOverlayEl.classList.add("c8a11y-test-mode-overlay");
        // Render 4 x 4 grid
        const testGrid = [...Array(4)]
          .map((_, yi) => {
            return [...Array(4)].map((_, xi) => {
              return [`x${xi + 1}`, `y${yi + 1}`];
            });
          })
          .flat();

        testGrid.map((block) => {
          const blockEl = document.createElement("div");
          blockEl.classList.add(
            "c8a11y-test-block",
            `c8a11y-test-block-${block[0]}-${block[1]}`
          );
          testModeOverlayEl.appendChild(blockEl);
        });

        bodyEl.appendChild(testModeOverlayEl);

        if (state.isTestMode) {
          highlightTestBlock();
        }
      }

      /* SECTION: MODE TOGGLE BUTTONS */

      function updatePredictButtonText(text) {
        document.querySelector(".c8a11y-predict-toggle-button").innerHTML =
          text;
      }

      function updateTestModeButtonText(text) {
        document.querySelector(".c8a11y-test-mode-toggle-button").innerHTML =
          text;
      }

      async function togglePredictMode() {
        if (state.isPredictMode) {
          // Toggle predict mode off
          state.isPredictMode = false;
          updatePredictButtonText("Turn predict mode on");
          overlayEl.style.visibility = "hidden"; // Hide predictions
        } else {
          // First turn off test mode
          if (state.isTestMode) {
            toggleTestMode();
          }

          // Toggle predict mode on
          state.isPredictMode = true;
          updatePredictButtonText("Turn predict mode off");
          overlayEl.style.visibility = "visible"; // Show predictions
          initPredictMode();
        }
      }

      async function toggleTestMode() {
        if (state.isTestMode) {
          // Toggle test mode off
          state.isTestMode = false;
          updateTestModeButtonText("Turn test mode on");
          testModeOverlayEl.style.visibility = "hidden"; // Hide test mode
        } else {
          // First turn off and reset predict mode
          if (state.isPredictMode) {
            togglePredictMode();
          }

          // Then toggle test mode on
          state.isTestMode = true;
          updateTestModeButtonText("Turn test mode off");
          testModeOverlayEl.style.visibility = "visible"; // Show test mode
          highlightTestBlock();
        }
      }

      function initToggleButtons() {
        const togglePredictButton = document.createElement("button");
        togglePredictButton.classList.add(
          "c8a11y-toggle-button",
          "c8a11y-predict-toggle-button"
        );
        togglePredictButton.innerHTML = "Turn predict mode off"; // Predict mode on by default
        togglePredictButton.addEventListener("click", togglePredictMode);

        const toggleTestModeButton = document.createElement("button");
        toggleTestModeButton.classList.add(
          "c8a11y-toggle-button",
          "c8a11y-test-mode-toggle-button"
        );
        toggleTestModeButton.innerHTML = "Turn test mode on"; // Test mode off by default
        toggleTestModeButton.addEventListener("click", toggleTestMode);

        const toggleButtons = document.createElement("div");
        toggleButtons.classList.add("c8a11y-toggle-buttons");
        toggleButtons.appendChild(togglePredictButton);
        toggleButtons.appendChild(toggleTestModeButton);

        bodyEl.querySelector(".c8a11y-info-banner").appendChild(toggleButtons);
      }

      /* Update the progress bar whilst the model is training */
      function updateProgress(progressAsPercentage) {
        bodyEl.querySelector(
          ".c8a11y-progress"
        ).style.width = `${progressAsPercentage}%`;
      }

      /* SECTION: TRAIN AND EVALUATE THE TENSORFLOW MODEL */

      async function evaluate(testInputs, testOutputs) {
        tf.tidy(() => {
          const xs = tf.tensor2d(testInputs);
          const ys = tf.tensor2d(testOutputs);

          const result = await model.evaluate(xs, ys); // Evaluate the model using test data

          // Cleanup tensors
          xs.dispose();
          ys.dispose();

          console.log("Model evaluation");
          result.print(); // Print the evaluation result to the browser console
        });
      }

      async function train() {
        state.isAlreadyTraining = true; // Update isAlreadyTraining state
        bodyEl.style.overflow = "auto"; // Unlock scroll on body

        const { inputs, outputs } = await cleanupData(dataSet);
        const { trainInputs, trainOutputs, testInputs, testOutputs } =
          await splitDataSets(inputs, outputs);

        // Update the info banner with the size of each training set and a progress bar
        updateInfoBanner(
          `Training model w/ ${trainInputs.length} training inputs and ${testInputs.length} testing inputs...<div class='c8a11y-progress-bar'><div class='c8a11y-progress'></div></div>`
        );

        // Create tensors from traning data
        const xs = tf.tensor2d(trainInputs);
        const ys = tf.tensor2d(trainOutputs);

        // Define model two model layers, hidden and output
        const hiddenLayer = tf.layers.dense({
          activation: "relu",
          inputShape: [4],
          units: trainInputs.length,
        });

        const outputLayer = tf.layers.dense({
          units: 2,
        });

        // Add layers to the model
        model.add(hiddenLayer);
        model.add(outputLayer);

        // Compile model
        model.compile({
          optimizer: tf.train.sgd(0.0001), // adam
          loss: "meanSquaredError",
        });

        // Print a summary of the compiled model in the browser console
        model.summary();

        const printCallback = {
          onEpochEnd: (epoch, log) => {
            updateProgress((epoch / TOTAL_EPOCHS) * 100); // Use current epoch value to determine how much training has done (as a percentage value)
          },
        };

        // Train the TS model
        await model
          .fit(xs, ys, {
            epochs: TOTAL_EPOCHS,
            callbacks: printCallback, // printCallback() is called after each tensor is passed into the model and provides an overview of the training process (and losses)
            batchSize: 10,
          })
          .then((history) => {
            console.log("Model history", history);
          });

        // Cleanup tensors
        xs.dispose();
        ys.dispose();

        updateInfoBanner(
          "Look around the screen to predict. You can toggle predictions on or off using the button 👉" // Update user instructions
        );
        state.isPredictMode = true;
        evaluate(testInputs, testOutputs); // Evaluate the now trained model with the test dataset
        initToggleButtons(); // Add buttons to the info banner to toggle prediction and test modes
        initTestMode(); // Setup test mode
        initPredictMode(); // Turn on predict mode by default
      }

      /* SECTION: SETUP CALIBRATION */

      /* Initialise dots by creating an array of x and y classnames based on the number of dots required */
      function initDots() {
        return [...Array(dotCount.x)]
          .map((_, xi) => {
            return [...Array(dotCount.y)].map((_, yi) => {
              return [`x${xi + 1}`, `y${yi + 1}`];
            });
          })
          .flat();
      }

      /* Create a single dot DOM element with the approriate class names and append it to the overlay */
      function createDot(dot, className) {
        const dotEl = document.createElement("div");
        dotEl.classList.add(...["c8a11y-dot", dot[0], dot[1]], ...[className]);
        overlayEl.appendChild(dotEl);
        return dotEl;
      }

      /* Remove a dot from the DOM */
      function removeDot(e, dotEl) {
        dotEl.classList.add("animate");

        setTimeout(function () {
          overlayEl.removeChild(e.target);
        }, 300);
      }

      /* Calibration is complete once all of the dots have been removed from the DOM.
       * Check if the model isn't already training and if not, call the specified function */
      function checkIfCalibrationComplete(onCalibrationCompletion) {
        setTimeout(function () {
          if (!overlayEl.hasChildNodes() && !state.isAlreadyTraining) {
            onCalibrationCompletion();
          }
        }, 500);
      }

      /* Add more calibration dots if prompted */
      async function addAdditionalCalibrationDots() {
        initMouseHighlighter();

        // Update dot count for second round of calibration
        dotCount.x = 5;
        dotCount.y = 3;

        initDots().map((dot) => {
          const dotEl = createDot(dot, "c8a11y-additional-dot");

          dotEl.addEventListener("click", async function (e) {
            removeDot(e, dotEl);
            checkIfCalibrationComplete(train);
          });
        });
      }

      /* Add calibration dots as child nodes of the overlay element */
      async function addInitialCalibrationDots() {
        initMouseHighlighter();

        initDots().map((dot) => {
          const dotEl = createDot(dot, "c8a11y-initial-dot");

          dotEl.addEventListener("click", async function (e) {
            removeDot(e, dotEl);
            checkIfCalibrationComplete(() => {
              initInfoModal(
                `<p>Great! You've collected ${dataSet.length} data points. For a more accurate prediction we recommend doing one more round of data collection before training your model.</p>`,
                "OK!",
                addAdditionalCalibrationDots,
                "Train model using current data set",
                train
              );
            });
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

      /* SECTION: ADD KEY ELEMENTS TO THE DOM */

      function initOverlay() {
        overlayEl = document.createElement("div");
        overlayEl.classList.add("c8a11y-overlay");
        // Lock scroll on body
        bodyEl.style.overflow = "hidden";
        bodyEl.appendChild(overlayEl);

        initDataCollection();
        addInitialCalibrationDots();
      }

      async function initVideoStream() {
        videoEl = document.createElement("video");
        canvasEl = document.createElement("canvas");
        const containerEl = document.createElement("div");
        peepholeEl = document.createElement("div");

        containerEl.classList.add("c8a11y-stream-container");
        canvasEl.classList.add("c8a11y-canvas");
        videoEl.classList.add("c8a11y-video");
        peepholeEl.classList.add("c8a11y-peephole");

        bodyEl.appendChild(containerEl);
        containerEl.appendChild(canvasEl);
        containerEl.appendChild(videoEl);
        containerEl.appendChild(peepholeEl);

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

      /* SECTION: DRAW FACIAL LANDMARKS ON THE CANVAS / VIDEO */

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

          // Draw frame around user's eyes
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

          if (state.isFirstLoad) {
            state.isFirstLoad = false;
            clearInfoBanner();
          }
        }

        requestAnimationFrame(() => {
          renderKeypoints(ctx, canvasEl, videoEl);
        });
      }

      /* SECTION: MAIN */

      async function main() {
        // First add the info banner to tell use that the video stream is loading
        initInfoBanner();
        // Fire modal with instructions of how to use the application / extension
        initInfoModal(
          "<p>You’ve successfully activated <i>c8a11y</i>!</p><p><i>c8a11y</i> is an application that determines where you are looking in the browser, but first, the application needs to be trained. To do so, simply </p><ol><li>Ensure <b>access to your camera is enabled</b></li><li>Click away all of the <div class='c8a11y-dot inline'></div> green dots on the screen</li><li>Keep looking at the <div class='c8a11y-mouse-target inline'></div> bright blue dot whilst moving the cursor around</li></ol><p>Though not essential, the bright blue rectangle can be used as a guide for positioning your head and might result in more accurate results.</p><p><i>Ready?</i>",
          "Let's go!",
          () => {
            state.shouldRenderOverlay = true;
          }
        );

        // Setup the video stream
        await initVideoStream();

        if (videoEl && canvasEl) {
          videoEl.play();
          videoEl.width = VIDEO_WIDTH;
          videoEl.height = VIDEO_HEIGHT;
          canvasEl.width = VIDEO_WIDTH;
          canvasEl.height = VIDEO_HEIGHT;

          const ctx = initCanvas(canvasEl);
          await renderKeypoints(ctx, canvasEl, videoEl);
          if (state.shouldRenderOverlay) {
            // Add overlay, only once video stream and canvas has been configured
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
