import std/os
import std/random
import std/streams
import std/strformat
import std/math

const BIAS = 20
const SAMPLE_SIZE = 75
const TRAIN_PASSES = 2000

const PPM_SCALER = 25
const PPM_COLOR_INTENSITY = 255
const PPM_RANGE = 10

const TRAIN_SEED = 71
const CHECK_SEED = 411

const DATA_FOLDER = "data"

const WIDTH = 20
const HEIGHT = 20

type Layer = array[WIDTH, array[HEIGHT, float]]

var inputs: Layer
var weights: Layer

proc layerFillRectangle(layer: var Layer, x, y, w, h: int, value: float) =
  let x0 = clamp(x, 0, WIDTH - 1)
  let y0 = clamp(y, 0, HEIGHT - 1)
  let x1 = clamp(x0 + w - 1, 0, WIDTH - 1)
  let y1 = clamp(y0 + h - 1, 0, HEIGHT - 1)

  for y in y0 .. y1:
    for x in x0 .. x1:
      layer[y][x] = value


proc layerFillCircle(layer: var Layer, cx, cy, r: int, value: float) =
  let x0 = clamp(cx - r, 0, WIDTH - 1)
  let y0 = clamp(cy - r, 0, HEIGHT - 1)
  let x1 = clamp(cx + r, 0, WIDTH - 1)
  let y1 = clamp(cy + r, 0, HEIGHT - 1)

  for y in y0 .. y1:
    for x in x0 .. x1:
      let dx = x - cx
      let dy = y - cy
      if dx * dx + dy * dy <= r * r:
        layer[y][x] = value

proc layerSaveAsPPM(layer: Layer, filePath: string) =
  let file = newFileStream(filePath, fmWrite)
  if not isNil(file):
    const wppm = WIDTH * PPM_SCALER
    const hppm = HEIGHT * PPM_SCALER
    file.writeLine("P6")
    file.writeLine(fmt"{wppm} {hppm} 255")
    var buffer = newString(wppm * 3)
    for y in 0 ..< hppm:
      for x in 0 ..< wppm:
        let s = (layer[y div PPM_SCALER][x div PPM_SCALER] + PPM_RANGE) / (2.0 * PPM_RANGE)
        buffer[x * 3] = char(PPM_COLOR_INTENSITY * (1 - s))
        buffer[x * 3 + 1] = char(PPM_COLOR_INTENSITY * s)
        buffer[x * 3 + 2] = char(PPM_COLOR_INTENSITY * (1 - s))
      file.write(buffer)
  file.close()

func feedForward(inputs, weights: Layer): float =
  var output: float = 0.0

  for y in 0 ..< HEIGHT:
    for x in 0 ..< WIDTH:
      output += inputs[y][x] * weights[y][x]

  output

proc addInputsFromWeights(inputs: Layer, weights: var Layer) =
  for y in 0 ..< HEIGHT:
    for x in 0 ..< WIDTH:
      weights[y][x] += inputs[y][x]

proc subInputsFromWeights(inputs: Layer, weights: var Layer) =
  for y in 0 ..< HEIGHT:
    for x in 0 ..< WIDTH:
      weights[y][x] -= inputs[y][x]

proc layerRandomRectangle(layer: var Layer, seed: var Rand) =
  layerFillRectangle(layer, 0, 0, WIDTH, HEIGHT, 0.0)
  let x = seed.rand(0 ..< WIDTH)
  let y = seed.rand(0 ..< HEIGHT)

  var w = WIDTH - x
  if w < 2:
    w = 2
  w = seed.rand(1 ..< w)

  var h = HEIGHT - y
  if h < 2:
    h = 2
  h = seed.rand(1 ..< h)

  layerFillRectangle(layer, x, y, w, h, 1.0)

proc layerRandomCircle(layer: var Layer, seed: var Rand) =
  layerFillRectangle(layer, 0, 0, WIDTH, HEIGHT, 0.0)
  let x = seed.rand(0 ..< WIDTH)
  let y = seed.rand(0 ..< HEIGHT)

  var r = high(int)
  if r > x:
    r = x
  if r > y:
    r = y
  if r > (WIDTH - x):
    r = WIDTH - x
  if r > (HEIGHT - y):
    r = HEIGHT - y
  if r < 2:
    r = 2
  r = seed.rand(1 ..< r)

  layerFillCircle(layer, x, y, r, 1.0)

var count: int = 0
proc trainPass(inputs, weights: var Layer): int =
  var seed = initRand(TRAIN_SEED)
  var filePath = ""

  var adjusted: int = 0
  for i in 0 .. SAMPLE_SIZE:
    layerRandomRectangle(inputs, seed)
    if feedForward(inputs, weights) > BIAS:
      subInputsFromWeights(inputs, weights)
      count += 1
      filePath = fmt"{DATA_FOLDER}/weights-{count:03}.ppm"
      layerSaveAsPPM(weights, filePath)
      adjusted += 1

    layerRandomCircle(inputs, seed)
    if feedForward(inputs, weights) < BIAS:
      addInputsFromWeights(inputs, weights)
      count += 1
      filePath = fmt"{DATA_FOLDER}/weights-{count:03}.ppm"
      layerSaveAsPPM(weights, filePath)
      adjusted += 1

  return adjusted

proc checkPass(inputs, weights: var Layer): int =
  var seed = initRand(CHECK_SEED)

  var adjusted: int = 0
  for i in 0 .. SAMPLE_SIZE:
    layerRandomRectangle(inputs, seed)
    if feedForward(inputs, weights) > BIAS:
      adjusted += 1

    layerRandomCircle(inputs, seed)
    if feedForward(inputs, weights) < BIAS:
      adjusted += 1
  return adjusted

proc main() =
  removeDir(DATA_FOLDER)
  createDir(DATA_FOLDER)

  discard checkPass(inputs, weights)

  for i in 0 ..< TRAIN_PASSES:
    var adjusted: int = trainPass(inputs, weights)
    echo adjusted
    if adjusted <= 0:
      break

  discard checkPass(inputs, weights)

main()
