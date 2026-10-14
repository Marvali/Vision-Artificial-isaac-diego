%% Crear nuevos Tracks
% Cree nuevas pistas a partir de detecciones no asignadas. Asume que cualquier 
% detección no asignada es el inicio de una nuevo track. En la práctica, puede 
% utilizar otras tracks para eliminar las detecciones ruidosas, como el tamaño, 
% la ubicación o la apariencia.
    function [tracks trackssum nextId] = createNewTracks(tracks, trackssum, detections, unassignedDetections, MotionModel, nextId, frameId)
    
        detections.centroids = detections.centroids(unassignedDetections, :);
        detections.bboxes = detections.bboxes(unassignedDetections, :);
        
        for i = 1:size(detections.centroids, 1)
            
            centroid = detections.centroids(i,:);
            bbox = detections.bboxes(i, :);
            
            % Create a Kalman filter object.
            % (MotionModel, InitialLocation, InitialEstimateError, MotionNoise, MeasurementNoise
            if strcmp(MotionModel,'ConstantAcceleration')
                kalmanFilter = configureKalmanFilter('ConstantAcceleration', ...
                    centroid, [200, 50, 50], [100, 25, 25], 100);
            else
                kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                    centroid, [200, 50], [100, 25], 100);
            end
            % Create a new track.
            newTrack = struct(...
                'id', nextId, ...
                'bbox', bbox, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0);
            
            % Add it to the array of tracks.
            tracks(end + 1) = newTrack;
            
            % Create a new tracksum.
            trackdetection = struct(...
                'bbox', bbox, ...
                'centroid', centroid, ...
                'visibility', 1);
            
            newTracksum = struct(...
                'id', nextId, ...
                'frame_i', frameId, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0, ...
                'detections', trackdetection);
            
            % Add it to the array of trackssum.
            trackssum(end + 1) = newTracksum;  
            
            % Increment the next id.
            nextId = nextId + 1;
        end
    end