const inputVideo = document.getElementById("inputVideo");
const outputCanvas = document.getElementById("outputCanvas");
const outputCanvasContext = outputCanvas.getContext("2d");
const backendOutput = document.getElementById("backendOutput");


async function main(blazefaceAnchors) {
    load_settings();

    const faceDetModel = await tf.loadGraphModel("https://tfhub.dev/tensorflow/tfjs-model/blazeface/1/default/1", { fromTFHub: true });
                                             
    const webcam = await tf.data.webcam(inputVideo);

    const sourceSize = [inputVideo.videoHeight, inputVideo.videoWidth]
    outputCanvas.height = sourceSize[0];
    outputCanvas.width = sourceSize[1];

    const faceDetSize = sourceSize[0] > sourceSize[1] ? [128, Math.round(sourceSize[1] * 128 / sourceSize[0])] :
      (sourceSize[1] > sourceSize[0] ?[Math.round(sourceSize[0] * 128 / sourceSize[1]), 128] : [128, 128]);

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

      const faceRect = await get_face_rect(faceDetModel, img, sourceSize, faceDetSize, faceDetPadding, blazefaceAnchors);
      outputCanvasContext.drawImage(inputVideo, 0, 0);
      if (faceRect[0] > 0.9) {
        plotFaceRect(faceRect);
        //plot_landmarks(predictions);
      }


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

async function get_face_rect(faceDetModel, img, sourceSize, targetSize, padding, anchors) {
  const faceDetInput = tf.tidy(() => {
    const scale = tf.scalar(127.5);
    const offset = tf.scalar(1);
    const result = img.resizeBilinear(targetSize).div(scale).sub(offset).pad(padding, 0).reshape([1, 128, 128, 3]);
    return result;
  });
    
  let predictions = faceDetModel.predict(faceDetInput);
  
  const result = tf.tidy(() => {
    predictions = predictions.squeeze();
    const logits = predictions.slice([0, 0], [-1, 1]).squeeze();
    const bestRectIDX = logits.argMax();
    const bestPred = predictions.gather(bestRectIDX).squeeze();
    const bestRect = bestPred.slice(0, 5);
    const anchor = anchors.gather(bestRectIDX).squeeze();
    return [bestRect, anchor];
  });
  
  const bestRect = await result[0].data();
  const anchor = await result[1].data();

  bestRect[0] = 1 / (1 + Math.exp(-bestRect[0]));

  bestRect[1] += anchor[0] * 128 - padding[1][0];
  bestRect[2] += anchor[1] * 128 - padding[0][0];
  bestRect[3] *= anchor[2];
  bestRect[4] *= anchor[3];
    
  scale = sourceSize[0] / targetSize[0];
  for (let i = 1; i < 5; i++) {
    bestRect[i] *= scale;
  }

  bestRect[1] -= bestRect[3] / 2;
  bestRect[2] -= bestRect[4] / 2;
  bestRect[3] += bestRect[1];
  bestRect[4] += bestRect[2];

  for (let i = 1; i < 5; i++) {
    bestRect[i] = Math.round(bestRect[i]);
  }

  return bestRect;
}

function plotFaceRect(faceRect) {
  outputCanvasContext.strokeStyle = "green";
  outputCanvasContext.lineWidth = 2;
  outputCanvasContext.beginPath();
  outputCanvasContext.rect(
    faceRect[1], faceRect[2], faceRect[3] - faceRect[1], faceRect[4] - faceRect[2]);
  outputCanvasContext.stroke();
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
