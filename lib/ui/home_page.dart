import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../ml/onnx_service.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final OnnxService _onnxService = OnnxService();
  final ImagePicker _picker = ImagePicker();
  
  File? _image;
  String _result = '';
  bool _isProcessing = false;

  @override
  void initState() {
    super.initState();
    _initModel();
  }

  Future<void> _initModel() async {
    await _onnxService.init();
  }

  @override
  void dispose() {
    _onnxService.dispose();
    super.dispose();
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final pickedFile = await _picker.pickImage(source: source);
      if (pickedFile != null) {
        setState(() {
          _image = File(pickedFile.path);
          _result = '';
        });
        _processImage();
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to pick image: $e')),
        );
      }
    }
  }

  Future<void> _processImage() async {
    if (_image == null) return;

    setState(() {
      _isProcessing = true;
      _result = 'Analyzing image...';
    });

    final prediction = await _onnxService.predict(_image!);

    if (mounted) {
      setState(() {
        _isProcessing = false;
        _result = prediction;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Image.asset(
              'assets/logo.png',
              height: 32,
            ),
            const SizedBox(width: 12),
            const Text(
              'CerviScan',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
          ],
        ),
        centerTitle: true,
      ),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
            if (_image != null) ...[
              Card(
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    Image.file(
                      _image!,
                      height: 300,
                      width: double.infinity,
                      fit: BoxFit.cover,
                    ),
                    if (_isProcessing)
                      const Positioned.fill(
                        child: DecoratedBox(
                          decoration: BoxDecoration(
                            color: Colors.black45,
                          ),
                          child: Center(
                            child: CircularProgressIndicator(
                              color: Colors.white,
                            ),
                          ),
                        ),
                      ),
                  ],
                ),
              ),
              const SizedBox(height: 24),
              Card(
                elevation: 0,
                color: Theme.of(context).colorScheme.primaryContainer,
                child: Padding(
                  padding: const EdgeInsets.all(24.0),
                  child: Column(
                    children: [
                      Text(
                        'Analysis Result',
                        style: Theme.of(context).textTheme.titleMedium?.copyWith(
                              color: Theme.of(context).colorScheme.onPrimaryContainer,
                              fontWeight: FontWeight.bold,
                            ),
                      ),
                      const SizedBox(height: 12),
                      Text(
                        _result,
                        style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                              color: Theme.of(context).colorScheme.onPrimaryContainer,
                            ),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
              ),
            ] else ...[
              Card(
                elevation: 0,
                color: Theme.of(context).colorScheme.surfaceContainerHighest,
                child: Container(
                  height: 300,
                  alignment: Alignment.center,
                  child: Icon(
                    Icons.image_search,
                    size: 80,
                    color: Theme.of(context).colorScheme.onSurfaceVariant.withOpacity(0.5),
                  ),
                ),
              ),
            ],
            const SizedBox(height: 32),
            Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                FilledButton.icon(
                  style: FilledButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 20),
                  ),
                  onPressed: _isProcessing ? null : () => _pickImage(ImageSource.camera),
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('Camera', style: TextStyle(fontSize: 16)),
                ),
                const SizedBox(height: 16),
                FilledButton.tonalIcon(
                  style: FilledButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 20),
                  ),
                  onPressed: _isProcessing ? null : () => _pickImage(ImageSource.gallery),
                  icon: const Icon(Icons.photo_library),
                  label: const Text('Gallery', style: TextStyle(fontSize: 16)),
                ),
              ],
            ),
          ],
        ),
      ),
      ),
    );
  }
}
