import 'package:cloud_firestore/cloud_firestore.dart';
import '../../domain/entities/detection_event_entity.dart';

class DetectionEventModel extends DetectionEventEntity {
  const DetectionEventModel({
    required super.id,
    required super.anomalyType,
    required super.confidence,
    required super.gpsLat,
    required super.gpsLng,
    required super.timestamp,
    required super.imageUrl,
    required super.storagePath,
    required super.status,
    super.diseaseName,
  });

  factory DetectionEventModel.fromFirestore(DocumentSnapshot doc) {
    final d = doc.data() as Map<String, dynamic>;
    return DetectionEventModel(
      id: doc.id,
      anomalyType: d['anomaly_type'] as String? ?? '',
      confidence: (d['confidence'] as num?)?.toDouble() ?? 0.0,
      gpsLat: (d['gps_lat'] as num?)?.toDouble() ?? 0.0,
      gpsLng: (d['gps_lng'] as num?)?.toDouble() ?? 0.0,
      timestamp: DateTime.tryParse(d['timestamp'] as String? ?? '') ?? DateTime.now(),
      imageUrl: d['image_url'] as String? ?? '',
      storagePath: d['storage_path'] as String? ?? '',
      status: d['status'] as String? ?? 'new',
      diseaseName: d['disease_name'] as String?,
    );
  }
}
