%% Actualizar los tracks no asignados
% Marca cada track no asignado como invisible, y aumenta su edad en 1.
    function [tracks trackssum] = updateUnassignedTracks(tracks, trackssum, unassignedTracks)
    
        for i = 1:length(unassignedTracks)
            
            ind = unassignedTracks(i);
            
            tracks(ind).age = tracks(ind).age + 1;
            
            tracks(ind).consecutiveInvisibleCount = tracks(ind).consecutiveInvisibleCount + 1;
            
            trackdetection = struct(...
                'bbox', [], ...
                'centroid', [], ...
                'visibility', 0);

            trackssum(tracks(ind).id).age = tracks(ind).age;
            trackssum(tracks(ind).id).consecutiveInvisibleCount = tracks(ind).consecutiveInvisibleCount;
            trackssum(tracks(ind).id).detections = [trackssum(tracks(ind).id).detections; trackdetection]; 

        end
        
    end