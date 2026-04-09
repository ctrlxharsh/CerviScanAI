import 'package:flutter/material.dart';
import 'ui/home_page.dart';
import 'ui/theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const CerviScanApp());
}

class CerviScanApp extends StatelessWidget {
  const CerviScanApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'CerviScanAI',
      theme: CerviScanTheme.lightTheme,
      home: const HomePage(),
      debugShowCheckedModeBanner: false,
    );
  }
}
