chrome.extension.sendMessage({}, function (response) {
  var readyStateCheckInterval = setInterval(function () {
    if (document.readyState === "complete") {
      clearInterval(readyStateCheckInterval);

      const videoEl = document.createElement("video");
      videoEl.setAttribute("id", "videoEl");
      videoEl.setAttribute("autoplay", "true");
      document.querySelector("body").appendChild(videoEl);

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
    }
  }, 10);
});
