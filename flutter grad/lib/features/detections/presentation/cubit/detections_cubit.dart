import 'dart:async';
import 'package:flutter_bloc/flutter_bloc.dart';
import '../../domain/repositories/detections_repository.dart';
import 'detections_state.dart';

class DetectionsCubit extends Cubit<DetectionsState> {
  final DetectionsRepository _repository;
  StreamSubscription? _sub;

  DetectionsCubit(this._repository) : super(const DetectionsInitial());

  void watch() {
    emit(const DetectionsLoading());
    _sub?.cancel();
    _sub = _repository.watchEvents().listen(
      (events) => emit(DetectionsLoaded(events)),
      onError: (e) => emit(DetectionsError(e.toString())),
    );
  }

  @override
  Future<void> close() {
    _sub?.cancel();
    return super.close();
  }
}
