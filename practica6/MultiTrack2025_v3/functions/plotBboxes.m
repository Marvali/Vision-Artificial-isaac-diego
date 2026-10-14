% La función plotBboxes dibuja el primer y el último bounding box de los 
% tracks cuyos identificadores se pasan en track_IDs, si track_IDs < 0 
% se dibujan todos.
    function image = plotBboxes(image, trackssum, track_IDs, display)

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
            % frame_i = trackssum(ID).frame_i;
            bboxes = cat(1,  trackssum(ID).detections.bbox);
            % frame_f = frame_i + length(bboxes);

            id_color = mod(ID,size(display.colors, 1)) + 1;

            if display.video_mode == 1
                image = insertShape(image, 'Rectangle', bboxes(end,:), 'Color', display.colors(id_color,:), 'LineWidth', 2);
                position = bboxes(end,1:2); 
                image = insertText(image, position, sprintf('%d.i',ID), "TextBoxColor",display.colors(id_color,:), 'FontColor', "black","FontSize",15);
            else
                image = insertObjectAnnotation(image, 'rectangle', bboxes(1,:), sprintf('ID%d.i',ID),"Color",uint8(display.colors(id_color,:)*255));
                image = insertObjectAnnotation(image, 'rectangle', bboxes(end,:), sprintf('ID%d.f',ID),"Color",uint8(display.colors(id_color,:)*255));
            end
            % BoundingBoxes (first and last)

            % Centroids (track record)
            % image = insertMarker(image, centroids, display.marker, MarkerColor=display.colors(1,:),Size=6);

            % display.colors = circshift(display.colors,1);
        end
        

    end