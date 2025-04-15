let canvas
let context
let paint
let clickX = []
let clickY = []
let clickDrag = []
const CANVAS_SIZE = 250

function startCanvas() {
  canvas = document.getElementById("canvas")
  context = canvas.getContext("2d")

  context.strokeStyle = "#ffffff"
  context.lineJoin = "round"
  context.lineWidth = 8

  canvas.addEventListener("touchstart", function (e) {
    e.preventDefault()
    var touch = e.touches[0]
    var mouseEvent = new MouseEvent("mousedown", {
      clientX: touch.clientX,
      clientY: touch.clientY,
    })
    canvas.dispatchEvent(mouseEvent)
  })

  canvas.addEventListener("touchmove", function (e) {
    e.preventDefault()
    var touch = e.touches[0]
    var mouseEvent = new MouseEvent("mousemove", {
      clientX: touch.clientX,
      clientY: touch.clientY,
    })
    canvas.dispatchEvent(mouseEvent)
  })

  canvas.addEventListener("touchend", function (e) {
    e.preventDefault()
    var mouseEvent = new MouseEvent("mouseup")
    canvas.dispatchEvent(mouseEvent)
  })

  $('#canvas').mousedown(function (e) {
    paint = true
    addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop, false)
    drawCanvas()
  })

  $('#canvas').mousemove(function (e) {
    if (paint) {
      addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop, true)
      drawCanvas()
    }
  })

  $('#canvas').mouseup(function (e) {
    paint = false
    drawCanvas()
  })

  $('#canvas').mouseleave(function (e) {
    paint = false
  })
}

function addClick(x, y, dragging) {
  clickX.push(x)
  clickY.push(y)
  clickDrag.push(dragging)
}

function clearCanvas() {
  context.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE)
}

function resetCanvas() {
  clickX = []
  clickY = []
  clickDrag = []
  clearCanvas()
}

function drawCanvas() {
  clearCanvas()

  for(let i=0; i<clickX.length; i++) {
    context.beginPath()
    if (clickDrag[i] && i) {
      context.moveTo(clickX[i - 1], clickY[i - 1])
    } else {
      context.moveTo(clickX[i] - 1, clickY[i])
    }
    context.lineTo(clickX[i], clickY[i])
    context.closePath()
    context.stroke()
  }
}

function getPixels() {
  // Get the raw pixel data from the canvas
  let rawPixels = context.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE).data
  let _pixels = []
  let pixels = []

  // Extract alpha channel values (transparency)
  for (let i=0; i < rawPixels.length; i += 4) {
    _pixels.push(rawPixels[i + 3])
  }

  // Downsample from 250x250 to 50x50 for the model
  // We need to sample every 5th pixel in both dimensions
  const step = 5
  const pixelsPerRow = CANVAS_SIZE
  
  for (let y = 0; y < CANVAS_SIZE; y += step) {
    for (let x = 0; x < CANVAS_SIZE; x += step) {
      const index = y * pixelsPerRow + x
      pixels.push(_pixels[index])
    }
  }

  return pixels
}

function practiceAction() {
  let pixels = getPixels()
  document.getElementById("pixels").value = pixels
  document.getElementById("practice-form").submit()
}

function addDataAction() {
  let pixels = getPixels()
  document.getElementById("pixels").value = pixels 
  document.getElementById("add-data-form").submit()
}
