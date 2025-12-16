% La función plotCentroids dibuja los centoides de los tracks cuyos 
% identificadores se pasan en track_IDs, si track_IDs < 0 se dibujan todos.
function image = plotCentroids(image, trackssum, track_IDs, display)

    % identificadores.
    ids = int32([trackssum(:).id]);
    
    % si trackId == -1 se muestran todos los tracks
    if track_IDs == -1
        % que cumplan totalVisibleCount > ageThreshold
        reliableTrackIDs = [trackssum(:).totalVisibleCount] > display.ageThreshold;
        track_IDs = int32([trackssum(reliableTrackIDs).id])';          
    else
        
        for i=1:length(track_IDs)
            if track_IDs(i) ~= ids
                disp(['Error infoTracks function! ' track_IDs ' no es un identificador correcto']);
                return;
            end
        end
    end
     
    for i=1:length(track_IDs)
        ID = track_IDs(i);
        centroids = cat(1,  trackssum(ID).detections.centroid);
        id_color = mod(ID,size(display.colors, 1)) + 1;
        image = insertMarker(image, centroids, display.marker, MarkerColor=display.colors(id_color,:),Size=6);

    end
    

end