const FACE_MESH_SIZE = 192;


const inputVideo = document.getElementById("inputVideo");
const outputCanvas = document.getElementById("outputCanvas");
const outputCanvasContext = outputCanvas.getContext("2d");
const backendOutput = document.getElementById("backendOutput");


async function main(blazefaceAnchors) {
  loadSettings();

  const faceMeshModel = await tf.loadGraphModel("https://tfhub.dev/mediapipe/tfjs-model/facemesh/1/default/1", { fromTFHub: true });

  const webcam = await tf.data.webcam(inputVideo);

  const sourceSize = [inputVideo.videoHeight, inputVideo.videoWidth]

  outputCanvas.height = sourceSize[0];
  outputCanvas.width = sourceSize[1];
  outputCanvasContext.font = "18px Arial MS";

  const faceRectSize = Math.min(...sourceSize) / 2;
  const faceRect = [1,
                    Math.round(sourceSize[1] / 2 - faceRectSize / 2),
                    Math.round(sourceSize[0] / 2 - faceRectSize / 2),
                    Math.round(sourceSize[1] / 2 + faceRectSize / 2),
                    Math.round(sourceSize[0] / 2 + faceRectSize / 2)];
  console.log(faceRect);


  let fpsEMA = -1,
      prevFrameTime = -1;
  while (true) {
    const img = await webcam.capture();

    faceMesh = await detectFaceMesh(faceMeshModel, img, faceRect);

    outputCanvasContext.drawImage(inputVideo, 0, 0);
    plotFaceRect(faceRect);
    if (faceMesh[0] > 0.5) {
      plotLandmarks(faceMesh[1]);
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

function plotFaceRect(faceRect) {
  outputCanvasContext.strokeStyle = "purple";
  outputCanvasContext.lineWidth = 2;
  outputCanvasContext.beginPath();
  outputCanvasContext.rect(
    faceRect[1], faceRect[2], faceRect[3] - faceRect[1], faceRect[4] - faceRect[2]);
  outputCanvasContext.stroke();
}


async function detectFaceMesh(faceMeshModel, img, faceRect) {
  const faceMeshInput = tf.tidy(() => preprocessFaceMesh(img, faceRect));

  const predictions = await faceMeshModel.predict(faceMeshInput);
  faceMeshInput.dispose();

  const result = tf.tidy(() => postprocessFaceMesh(predictions, faceRect));

  const faceProb = (await result[0].data())[0];
  result[0].dispose();
  const faceMesh = await result[1].array();
  result[1].dispose();

  faceRect = [1]
  for (let i = 0; i < result[2].length; ++i) {
    const coord = await result[2][i].data();
    faceRect.push(coord[0]);
    result[2][i].dispose();
  }

  for (let i = 0; i < predictions.length; i++) {
    predictions[i].dispose();
  }

  return [faceProb, faceMesh, faceRect];
}

function preprocessFaceMesh(img, faceRect) {
  const scale = tf.scalar(255);
  const result = img
    .slice([faceRect[2], faceRect[1], 0],
      [faceRect[4] - faceRect[2], faceRect[3] - faceRect[1], -1])
    .resizeBilinear([FACE_MESH_SIZE, FACE_MESH_SIZE])
    .div(scale)
    .reshape([1, FACE_MESH_SIZE, FACE_MESH_SIZE, 3]);
  return result;
}

function postprocessFaceMesh(predictions, faceRect) {
  const prob = predictions[1];
  const scale = tf.tensor([(faceRect[3] - faceRect[1]) / FACE_MESH_SIZE, (faceRect[4] - faceRect[2]) / FACE_MESH_SIZE, 1])
  const offset = tf.tensor([faceRect[1], faceRect[2], 0])
  const predMesh = predictions[2].reshape([-1, 3]);
  const mesh = predMesh.mul(scale).add(offset);
  const x = mesh.slice([0, 0], [-1, 1]);
  const y = mesh.slice([0, 1], [-1, 1]);
  return [prob, mesh, [x.min(), y.min(), x.max(), y.max()]];
}

function plotLandmarks(predictions) {
  outputCanvasContext.fillStyle = "green";
  for (let i = 0; i < predictions.length; i++) {
    outputCanvasContext.beginPath();
    outputCanvasContext.arc(Math.round(predictions[i][0]), Math.round(predictions[i][1]), 2, 0, 2 * Math.PI);
    outputCanvasContext.fill();
  }
}

function calcFPS(prevFrameTime, currFrameTime, fpsEMA) {
  let currFPS = 1000 / (currFrameTime - prevFrameTime);
  if (fpsEMA >= 0) {
    fpsEMA = 0.05 * currFPS + (1 - 0.05) * fpsEMA;
  }
  else {
    fpsEMA = currFPS;
  }
  return fpsEMA;
}
