%% Actualizar los tracks asignados
% La función updateAssignedTracks actualiza cada track asignado con la 
% detección correspondiente. Llama al método correcto de vision.KalmanFilter 
% para corregir la estimación de la localización. A continuación, almacena 
% el nuevo bounding box, y aumenta la edad del track y el recuento total 
% visible en 1. Por último, la función establece el recuento invisible a 0. 
    function [tracks trackssum] = updateAssignedTracks(tracks, trackssum, assignments, detections)
    
        numAssignedTracks = size(assignments, 1);
        
        for i = 1:numAssignedTracks
            
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);
            centroid = detections.centroids(detectionIdx, :);
            bbox = detections.bboxes(detectionIdx, :);
            
            % Corregir la estimación de la posición del objeto utilizando la nueva detección
            correct(tracks(trackIdx).kalmanFilter, centroid);
            
            % Sustituye el bounding box de la predicción por el detectado.
            tracks(trackIdx).bbox = bbox;
            
            % Actualiza la edad del track
            tracks(trackIdx).age = tracks(trackIdx).age + 1;
            
            % Actualiza la visibilidad.
            tracks(trackIdx).totalVisibleCount = tracks(trackIdx).totalVisibleCount + 1;
            tracks(trackIdx).consecutiveInvisibleCount = 0;
                       
            trackdetection = struct(...
                'bbox', bbox, ...
                'centroid', centroid, ...
                'visibility', 1);

            trackIdxsum = tracks(trackIdx).id;
            trackssum(trackIdxsum).age = tracks(trackIdx).age;
            trackssum(trackIdxsum).totalVisibleCount = tracks(trackIdx).totalVisibleCount;
            trackssum(trackIdxsum).consecutiveInvisibleCount = 0;
            trackssum(trackIdxsum).detections = [trackssum(trackIdxsum).detections; trackdetection]; 
        end
        
    end