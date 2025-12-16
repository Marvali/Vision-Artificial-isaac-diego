%% Visualización de los resultados de la detección
% La función displayTrackingResults dibuja un cuadro delimitador y una 
% etiqueta ID para cada pista en el frame de vídeo y en la máscara de primer 
% plano. A continuación, muestra el frame y la máscara en sus respectivos 
% reproductores de vídeo. 
function displayDetections(obj, frame, mask, detections, display)
    
    mask = im2single(repmat(mask, [1, 1, 3])) .* 255;
    labels = strcat('Area=', cellstr(int2str(detections.area)));
    mask = insertObjectAnnotation(mask, 'rectangle', detections.bboxes, labels);
    % Display the mask and the frame.
    obj.maskPlayer(mask); 
        
    labels = strcat('Detection ',cellstr(int2str((1:1:size(detections.bboxes,1))')));
    frame = insertObjectAnnotation(frame, 'rectangle', detections.bboxes, labels);
    % Display the mask and the frame.
    obj.videoPlayer(frame);

    pause(display.speed);

    end
    