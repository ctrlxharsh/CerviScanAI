import 'dart:io';
import 'dart:typed_data';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart' show rootBundle;

class OnnxService {
  OrtSession? _session;

  Future<void> init() async {
    try {
      OrtEnv.instance.init();
      final sessionOptions = OrtSessionOptions();
      final rawAssetFile = await rootBundle.load('assets/model.onnx');
      final bytes = rawAssetFile.buffer.asUint8List();
      _session = OrtSession.fromBuffer(bytes, sessionOptions);
    } catch (e) {
      print('Error initializing ONNX: $e');
    }
  }

  Future<String> predict(File imageFile) async {
    if (_session == null) return "Model not loaded";

    try {
      final imageBytes = await imageFile.readAsBytes();
      final image = img.decodeImage(imageBytes);
      if (image == null) return "Invalid image";

      // Resize to 224x224
      final resized = img.copyResize(image, width: 224, height: 224);

      final mean = [0.485, 0.456, 0.406];
      final std = [0.229, 0.224, 0.225];

      var inputData = Float32List(1 * 3 * 224 * 224);
      int pixelIndex = 0;
      for (var y = 0; y < 224; y++) {
        for (var x = 0; x < 224; x++) {
          final pixel = resized.getPixel(x, y);
          final r = pixel.r / 255.0;
          final g = pixel.g / 255.0;
          final b = pixel.b / 255.0;

          inputData[0 * 224 * 224 + pixelIndex] = (r - mean[0]) / std[0]; // R
          inputData[1 * 224 * 224 + pixelIndex] = (g - mean[1]) / std[1]; // G
          inputData[2 * 224 * 224 + pixelIndex] = (b - mean[2]) / std[2]; // B
          pixelIndex++;
        }
      }

      final shape = [1, 3, 224, 224];
      final inputTensor = OrtValueTensor.createTensorWithDataList(inputData, shape);

      final inputName = _session!.inputNames[0];
      final inputs = {inputName: inputTensor};

      final runOptions = OrtRunOptions();
      final outputs = await _session!.runAsync(runOptions, inputs);

      final outputValue = outputs?.first?.value;
      if (outputs != null && outputValue != null) {
        // Output from MLP classification head might be nested
        List<dynamic> logits;
        if (outputValue is List && outputValue.isNotEmpty) {
          if (outputValue[0] is List) {
            logits = outputValue[0] as List<dynamic>;
          } else {
            logits = outputValue;
          }
        } else {
          return "Unexpected output format";
        }

        int maxIdx = 0;
        double maxVal = (logits[0] as num).toDouble();
        for (int i = 1; i < logits.length; i++) {
          final val = (logits[i] as num).toDouble();
          if (val > maxVal) {
            maxVal = val;
            maxIdx = i;
          }
        }

        // Clean up
        inputTensor.release();
        for (var o in outputs) {
          o?.release();
        }
        runOptions.release();

        return "Output Index: $maxIdx";
      }

      return "Output processing error";
    } catch (e) {
      return "Inference Error: $e";
    }
  }

  void dispose() {
    _session?.release();
    OrtEnv.instance.release();
  }
}
