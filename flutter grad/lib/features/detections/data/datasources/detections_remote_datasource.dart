import 'package:cloud_firestore/cloud_firestore.dart';
import '../models/detection_event_model.dart';

abstract class DetectionsRemoteDataSource {
  Stream<List<DetectionEventModel>> watchEvents({int limit = 50});
}

class DetectionsRemoteDataSourceImpl implements DetectionsRemoteDataSource {
  final FirebaseFirestore _firestore;

  DetectionsRemoteDataSourceImpl({FirebaseFirestore? firestore})
      : _firestore = firestore ?? FirebaseFirestore.instance;

  @override
  Stream<List<DetectionEventModel>> watchEvents({int limit = 50}) {
    return _firestore
        .collection('events')
        .orderBy('timestamp', descending: true)
        .limit(limit)
        .snapshots()
        .map((snap) =>
            snap.docs.map(DetectionEventModel.fromFirestore).toList());
  }
}
