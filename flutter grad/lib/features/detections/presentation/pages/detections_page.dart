import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:cached_network_image/cached_network_image.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:intl/intl.dart';
import '../../../../core/constants/app_colors.dart';
import '../../domain/entities/detection_event_entity.dart';
import '../cubit/detections_cubit.dart';
import '../cubit/detections_state.dart';

class DetectionsPage extends StatefulWidget {
  const DetectionsPage({super.key});

  @override
  State<DetectionsPage> createState() => _DetectionsPageState();
}

class _DetectionsPageState extends State<DetectionsPage> {
  @override
  void initState() {
    super.initState();
    context.read<DetectionsCubit>().watch();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.backgroundDeep,
      appBar: AppBar(
        backgroundColor: AppColors.backgroundBase,
        title: const Text('Detections',
            style: TextStyle(color: AppColors.textPrimary)),
        iconTheme: const IconThemeData(color: AppColors.textPrimary),
      ),
      body: BlocBuilder<DetectionsCubit, DetectionsState>(
        builder: (context, state) {
          if (state is DetectionsLoading) {
            return const Center(
                child: CircularProgressIndicator(color: AppColors.neonGreen));
          }
          if (state is DetectionsError) {
            return Center(
              child: Text(state.message,
                  style: const TextStyle(color: AppColors.alertFire)),
            );
          }
          if (state is DetectionsLoaded) {
            if (state.events.isEmpty) {
              return const Center(
                child: Text('No detections yet.',
                    style: TextStyle(color: AppColors.textSecondary)),
              );
            }
            return ListView.separated(
              padding: const EdgeInsets.all(16),
              itemCount: state.events.length,
              separatorBuilder: (_, __) => const SizedBox(height: 12),
              itemBuilder: (_, i) => _EventCard(event: state.events[i]),
            );
          }
          return const SizedBox.shrink();
        },
      ),
    );
  }
}

const _kStorageBucket = 'gs://agrosentinel-storage-2026';

class _EventCard extends StatefulWidget {
  final DetectionEventEntity event;

  const _EventCard({required this.event});

  @override
  State<_EventCard> createState() => _EventCardState();
}

class _EventCardState extends State<_EventCard> {
  Future<String>? _imageUrlFuture;

  @override
  void initState() {
    super.initState();
    if (widget.event.storagePath.isNotEmpty) {
      _imageUrlFuture = FirebaseStorage.instanceFor(bucket: _kStorageBucket)
          .ref(widget.event.storagePath)
          .getDownloadURL();
    }
  }

  Widget _imagePlaceholder({Widget? child}) => Container(
        height: 180,
        color: AppColors.backgroundElevated,
        child: Center(child: child),
      );

  Widget _brokenImage() => Container(
        height: 180,
        color: AppColors.backgroundElevated,
        child: const Icon(Icons.broken_image_outlined,
            color: AppColors.textMuted, size: 40),
      );

  @override
  Widget build(BuildContext context) {
    final event = widget.event;
    final formatted =
        DateFormat('MMM d, y · HH:mm').format(event.timestamp.toLocal());
    final confidencePct = (event.confidence * 100).toStringAsFixed(1);

    return Container(
      decoration: BoxDecoration(
        color: AppColors.backgroundCard,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.borderDim),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (_imageUrlFuture != null)
            ClipRRect(
              borderRadius:
                  const BorderRadius.vertical(top: Radius.circular(12)),
              child: FutureBuilder<String>(
                future: _imageUrlFuture,
                builder: (context, snapshot) {
                  if (snapshot.hasError) {
                    debugPrint(
                        '🔥 IMAGE getDownloadURL FAILED for path "${event.storagePath}" '
                        'error=${snapshot.error}');
                    return _brokenImage();
                  }
                  if (snapshot.connectionState == ConnectionState.waiting) {
                    return _imagePlaceholder(
                        child: const CircularProgressIndicator(
                            color: AppColors.neonGreen));
                  }
                  if (snapshot.hasData) {
                    return CachedNetworkImage(
                      imageUrl: snapshot.data!,
                      height: 180,
                      width: double.infinity,
                      fit: BoxFit.cover,
                      placeholder: (_, __) => _imagePlaceholder(
                          child: const CircularProgressIndicator(
                              color: AppColors.neonGreen)),
                      errorWidget: (_, url, error) {
                        debugPrint(
                            '🔥 CachedNetworkImage FAILED url=$url error=$error');
                        return _brokenImage();
                      },
                    );
                  }
                  return _brokenImage();
                },
              ),
            ),
          Padding(
            padding: const EdgeInsets.all(14),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 8, vertical: 3),
                      decoration: BoxDecoration(
                        color: AppColors.alertDiseaseFaint,
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Text(
                        event.anomalyType.toUpperCase(),
                        style: const TextStyle(
                            color: AppColors.alertDisease,
                            fontSize: 10,
                            fontWeight: FontWeight.w700,
                            letterSpacing: 0.8),
                      ),
                    ),
                    const Spacer(),
                    Text('$confidencePct%',
                        style: const TextStyle(
                            color: AppColors.neonGreen,
                            fontWeight: FontWeight.w600)),
                  ],
                ),
                if (event.diseaseName != null && event.diseaseName!.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.only(top: 8),
                    child: Text(event.diseaseName!,
                        style: const TextStyle(
                            color: AppColors.textPrimary,
                            fontSize: 15,
                            fontWeight: FontWeight.w600)),
                  ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    const Icon(Icons.location_on_outlined,
                        size: 14, color: AppColors.textSecondary),
                    const SizedBox(width: 4),
                    Text(
                      '${event.gpsLat.toStringAsFixed(5)}, '
                      '${event.gpsLng.toStringAsFixed(5)}',
                      style: const TextStyle(
                          color: AppColors.textSecondary, fontSize: 12),
                    ),
                  ],
                ),
                const SizedBox(height: 4),
                Row(
                  children: [
                    const Icon(Icons.access_time_outlined,
                        size: 14, color: AppColors.textMuted),
                    const SizedBox(width: 4),
                    Text(formatted,
                        style: const TextStyle(
                            color: AppColors.textMuted, fontSize: 12)),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
