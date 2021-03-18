const inputVideo = document.getElementById("inputVideo");
const outputCanvas = document.getElementById("outputCanvas");
const outputCanvasContext = outputCanvas.getContext("2d");
const backendOutput = document.getElementById("backendOutput");


async function main() {
    load_settings();


    const faceDetModel = await tf.loadGraphModel("https://tfhub.dev/tensorflow/tfjs-model/blazeface/1/default/1", { fromTFHub: true });
                                             
    const webcam = await tf.data.webcam(inputVideo);

    const videoWidth = inputVideo.videoWidth,
          videoHeight = inputVideo.videoHeight;
    outputCanvas.width = videoWidth;
    outputCanvas.height = videoHeight;

    const faceDetSize = videoHeight > videoWidth ? [128, Math.round(videoWidth * 128 / videoHeight)] :
      (videoWidth > videoHeight ?[Math.round(videoHeight * 128 / videoWidth), 128] : [128, 128]);

    console.log(faceDetSize);
    
    const faceDetPadding = [[Math.ceil((128 - faceDetSize[0]) / 2), Math.floor((128 - faceDetSize[0]) / 2)],
                            [Math.ceil((128 - faceDetSize[1]) / 2), Math.floor((128 - faceDetSize[1]) / 2)],
                            [0, 0]]
    console.log(faceDetPadding);

    outputCanvasContext.font = "18px Arial MS";

    let fps_ema = -1,
        prev_frame_time = -1;
    while (true) {
      const img = await webcam.capture();

      const faceRect = await get_face_rect(faceDetModel, img, faceDetSize, faceDetPadding);
      //console.log(faceRect);

      outputCanvasContext.drawImage(inputVideo, 0, 0);
      //plot_landmarks(predictions);

      let curr_frame_time = Date.now();
      if (prev_frame_time >= 0) {
        fps_ema = calc_fps(prev_frame_time, curr_frame_time, fps_ema);
      }
      outputCanvasContext.fillStyle = "red";
      outputCanvasContext.fillText(Math.round(fps_ema) + " FPS", 5, 20);
      prev_frame_time = curr_frame_time;

      img.dispose();
  
      await tf.nextFrame();
    }
}

function load_settings() {
  let url = new URL(window.location.href);

  let backend = url.searchParams.get("back") ?? "webgl";
  tf.setBackend(backend);
  backendOutput.innerText = "Backend: " + tf.getBackend();
}

async function get_face_rect(faceDetModel, img, size, padding) {
  const faceDetInput = tf.tidy(() => {
    const scale = tf.scalar(127.5);
    const offset = tf.scalar(1);
    const result = img.resizeBilinear(size).div(scale).sub(offset).pad(padding, 0).reshape([1, 128, 128, 3]);
    return result;
  });
    
  let predictions = faceDetModel.predict(faceDetInput);
  
  const bestRect = tf.tidy(() => {
    predictions = predictions.squeeze();
    const logits = predictions.slice([0, 0], [-1, 1]).squeeze();
    const bestRectIDX = logits.argMax();
    const bestPred = predictions.gather(bestRectIDX).squeeze();
    const bestRect = bestPred.slice(0, 5);
    return bestRect;
  });
  return await bestRect.data();
}

function plot_landmarks(predictions) {
  outputCanvasContext.fillStyle = "green";
  if (predictions.length > 0) {
    for (let i = 0; i < predictions.length; i++) {
      const keypoints = predictions[i].scaledMesh;
      for (let i = 0; i < keypoints.length; i++) {
        const [x, y, z] = keypoints[i];
        
        outputCanvasContext.beginPath();
        outputCanvasContext.arc(x, y, 2, 0, 2 * Math.PI);
        outputCanvasContext.fill();
      }
    }
  }
}

function calc_fps(prev_frame_time, curr_frame_time, fps_ema) {
    let curr_fps = 1000 / (curr_frame_time - prev_frame_time);
    if (fps_ema >= 0)
    {
        fps_ema = 0.05 * curr_fps + (1 - 0.05) * fps_ema;
    }
    else
    {
        fps_ema = curr_fps;
    }
    return fps_ema;
}

main();
