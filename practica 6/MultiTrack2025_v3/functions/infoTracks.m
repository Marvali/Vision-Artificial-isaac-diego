% La función infoTracks().
    function T = infoTracks(trackssum, invisibleForTooLong, track_IDs, display)
    
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
        

        frame_i = zeros(length(track_IDs),1);
        consecutiveInvisibleCount = zeros(length(track_IDs),1);
        ages = zeros(length(track_IDs),1);
        totalVisibleCounts = zeros(length(track_IDs),1);
        visibility = zeros(length(track_IDs),1);

        
        for i=1:length(track_IDs)
            
            ID = track_IDs(i);
            frame_i(i) = trackssum(ID).frame_i;
            ages(i) = length(trackssum(ID).detections);
            totalVisibleCounts(i) = sum(cat(1,  trackssum(ID).detections.visibility));
            consecutiveInvisibleCount(i) = trackssum(ID).consecutiveInvisibleCount;
            visibility(i) =  totalVisibleCounts(i)./ages(i);
            
            if ages(i) > invisibleForTooLong

            lastVisibleCount = sum(cat(1,trackssum(ID).detections(end-invisibleForTooLong+1:end).visibility));
            
                if  lastVisibleCount > 0
                    ages(i) = ages(i) - invisibleForTooLong;
                    totalVisibleCounts(i) = totalVisibleCounts(i) - invisibleForTooLong;
                    visibility(i) =  totalVisibleCounts(i)./ages(i);  
                end 
            end
            
        end
        
        if size(track_IDs,1) == 1
            track_IDs = track_IDs';
        end
        
        T = table(track_IDs, frame_i, ages, totalVisibleCounts, consecutiveInvisibleCount, visibility)

    end