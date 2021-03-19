const FACE_DETECT_SIZE = 128;


const inputVideo = document.getElementById("inputVideo");
const outputCanvas = document.getElementById("outputCanvas");
const outputCanvasContext = outputCanvas.getContext("2d");
const backendOutput = document.getElementById("backendOutput");


async function main(blazefaceAnchors) {
    loadSettings();

    const faceDetModel = await tf.loadGraphModel("https://tfhub.dev/tensorflow/tfjs-model/blazeface/1/default/1", { fromTFHub: true });
                                             
    const webcam = await tf.data.webcam(inputVideo);

    const sourceSize = [inputVideo.videoHeight, inputVideo.videoWidth]
    outputCanvas.height = sourceSize[0];
    outputCanvas.width = sourceSize[1];

    const faceDetSize = sourceSize[0] > sourceSize[1] ?
      [FACE_DETECT_SIZE, Math.round(sourceSize[1] * FACE_DETECT_SIZE / sourceSize[0])] :
      (sourceSize[1] > sourceSize[0] ?
        [Math.round(sourceSize[0] * FACE_DETECT_SIZE / sourceSize[1]), FACE_DETECT_SIZE] :
        [FACE_DETECT_SIZE, FACE_DETECT_SIZE]);
    
    const faceDetPadding = [[Math.ceil((FACE_DETECT_SIZE - faceDetSize[0]) / 2), Math.floor((FACE_DETECT_SIZE - faceDetSize[0]) / 2)],
                            [Math.ceil((FACE_DETECT_SIZE - faceDetSize[1]) / 2), Math.floor((FACE_DETECT_SIZE - faceDetSize[1]) / 2)],
                            [0, 0]]
    console.log(faceDetPadding);
    
    outputCanvasContext.font = "18px Arial MS";

    let fpsEMA = -1,
        prevFrameTime = -1;
    while (true) {
      const img = await webcam.capture();

      const faceRect = await getFaceRect(faceDetModel, img, sourceSize, faceDetSize, faceDetPadding, blazefaceAnchors);
      outputCanvasContext.drawImage(inputVideo, 0, 0);
      if (faceRect[0] > 0.9) {
        plotFaceRect(faceRect);
        // plotLandmarks(predictions);
      }

      let currFrameTime = Date.now();
      if (prevFrameTime >= 0) {
        fpsEMA = calcFPS(prevFrameTime, currFrameTime, fpsEMA);
      }
      outputCanvasContext.fillStyle = "red";
      outputCanvasContext.fillText(Math.round(fpsEMA) + " FPS", 5, 20);
      prevFrameTime = currFrameTime;

      img.dispose();
      
      await tf.nextFrame();
    }
}

function loadSettings() {
  let url = new URL(window.location.href);

  let backend = url.searchParams.get("back") ?? "webgl";
  tf.setBackend(backend);
  backendOutput.innerText = "Backend: " + tf.getBackend();
}

async function getFaceRect(faceDetModel, img, sourceSize, targetSize, padding, anchors) {
  const faceDetInput = tf.tidy(() => preprocessFaceDet(img, targetSize, padding));
    
  let predictions = await faceDetModel.predict(faceDetInput);
  faceDetInput.dispose();
  
  const result = tf.tidy(() => getBestRect(predictions, anchors));
  predictions.dispose();
  
  const faceRect = await result[0].data();
  result[0].dispose();
  const anchor = await result[1].data();
  result[1].dispose();

  postprocessFaceRect(faceRect, anchor, sourceSize, targetSize, padding);

  return faceRect;
}

function preprocessFaceDet(img, targetSize, padding) {
  /* Transforms the input image into a (128 x 128) tensor while keeping the aspect
   * ratio (what is expected by the corresponding face detection model), resulting
   * in potential letterboxing in the transformed image.
   */

  const scale = tf.scalar(127.5);
  const offset = tf.scalar(1);
  const result = img.resizeBilinear(targetSize)
    .div(scale)
    .sub(offset)
    .pad(padding, 0)
    .reshape([1, FACE_DETECT_SIZE, FACE_DETECT_SIZE, 3]);
  return result;
}

function getBestRect(predictions, anchors) {
  const squeezedPred = predictions.squeeze();
  const logits = squeezedPred.slice([0, 0], [-1, 1]).squeeze();
  const bestRectIDX = logits.argMax();
  const bestPred = squeezedPred.gather(bestRectIDX).squeeze();
  const bestRect = bestPred.slice(0, 5);
  const anchor = anchors.gather(bestRectIDX).squeeze();
  return [bestRect, anchor];
}

function postprocessFaceRect(rect, anchor, sourceSize, targetSize, padding) {
  rect[0] = 1 / (1 + Math.exp(-rect[0]));

  rect[1] += anchor[0] * FACE_DETECT_SIZE - padding[1][0];
  rect[2] += anchor[1] * FACE_DETECT_SIZE - padding[0][0];
  rect[3] *= anchor[2];
  rect[4] *= anchor[3];
    
  scale = sourceSize[0] / targetSize[0];
  for (let i = 1; i < 5; i++) {
    rect[i] *= scale;
  }

  rect[1] -= rect[3] / 2;
  rect[2] -= rect[4] / 2;
  rect[3] += rect[1];
  rect[4] += rect[2];

  for (let i = 1; i < 5; i++) {
    rect[i] = Math.round(rect[i]);
  }

  rect[1] = Math.max(0, rect[1]);
  rect[2] = Math.max(0, rect[2]);
  rect[3] = Math.min(sourceSize[1] - 1, rect[3]);
  rect[4] = Math.min(sourceSize[0] - 1, rect[4]);
}

function plotFaceRect(faceRect) {
  outputCanvasContext.strokeStyle = "green";
  outputCanvasContext.lineWidth = 2;
  outputCanvasContext.beginPath();
  outputCanvasContext.rect(
    faceRect[1], faceRect[2], faceRect[3] - faceRect[1], faceRect[4] - faceRect[2]);
  outputCanvasContext.stroke();
}

function plotLandmarks(predictions) {
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

function calcFPS(prevFrameTime, currFrameTime, fpsEMA) {
    let currFPS = 1000 / (currFrameTime - prevFrameTime);
    if (fpsEMA >= 0)
    {
        fpsEMA = 0.05 * currFPS + (1 - 0.05) * fpsEMA;
    }
    else
    {
        fpsEMA = currFPS;
    }
    return fpsEMA;
}
