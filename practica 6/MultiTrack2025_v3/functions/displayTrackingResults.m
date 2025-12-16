%% Visualización de los resultados del seguimiento
% La función displayTrackingResults dibuja un cuadro delimitador y una 
% etiqueta ID para cada pista en el frame de vídeo y en la máscara de primer 
% plano. A continuación, muestra el frame y la máscara en sus respectivos 
% reproductores de vídeo. 
    function displayTrackingResults(obj, frame, mask, tracks, trackssum, display)
    
    % display.colors= [0         0    0.7500;
    %              0         0    1.0000;
    %              0    0.2500    1.0000;
    %              0    0.5000    1.0000;
    %              0    0.7500    1.0000;
    %              0    1.0000    1.0000;
    %         0.2500    1.0000    0.7500;
    %         0.5000    1.0000    0.5000;
    %         0.7500    1.0000    0.2500;
    %         1.0000    1.0000         0;
    %         1.0000    0.7500         0;
    %         1.0000    0.5000         0;
    %         1.0000    0.2500         0;
    %         1.0000         0         0;
    %         0.7500         0         0];

    % % Set the color for each track based on its ID
    % trackColors = display.colors(mod(ids, size(display.colors, 1)) + 1, :);
    % 
    % % Draw bounding boxes on the frame
    % frame = insertShape(frame, 'Rectangle', bboxes, 'Color', trackColors, 'LineWidth', 2);

        % 
        % Convertir la máscara en single
        mask = im2single(repmat(mask, [1, 1, 3])) .* 255;

            
        if ~isempty(tracks)
              
            % Noisy detections tend to result in short-lived tracks.
            % Only display tracks that have been visible for more than 
            % a minimum number of frames.
            reliableTrackInds = [tracks(:).totalVisibleCount] > display.ageThreshold;
            reliableTracks = tracks(reliableTrackInds);
           
            % Display the objects. If an object has not been detected
            % in this frame, display its predicted bounding box.
            if ~isempty(reliableTracks)
                % Get bounding boxes.
                bboxes = cat(1, reliableTracks.bbox);
                
                % Get ids.
                ids = int32([reliableTracks(:).id]);
                
                % Create labels for objects indicating the ones for 
                % which we display the predicted rather than the actual 
                % location.
                labels = cellstr(int2str(ids'));
                predictedTrackInds = [reliableTracks(:).consecutiveInvisibleCount] > 0;
                isPredicted = cell(size(labels));
                isPredicted(predictedTrackInds) = {' predicted'};
                labels = strcat(labels, isPredicted);
                
                % Draw the objects on the frame.
                frame = insertObjectAnnotation(frame, 'rectangle', ...
                    bboxes, labels);
                % frame = plotBboxes(frame, trackssum, -1, display);
                
                mask = plotCentroids(mask, trackssum, -1, display);
                mask = plotBboxes(mask, trackssum, -1, display);
                % Draw the objects on the mask.
                % mask = insertObjectAnnotation(mask, 'rectangle', ...
                %     bboxes, labels);
                % plotTracks(trackssum, -1)


            end
        end
        
        % Display the mask and the frame.
        obj.maskPlayer(mask); 
        obj.videoPlayer(frame);


        pause(display.speed);

    end
    