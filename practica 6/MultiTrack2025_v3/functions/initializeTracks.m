%% Inicializa Tracks
%
% La función initializeTracks crea una matriz de tracks, donde cada track es una estructura que representa un objeto en movimiento en el vídeo. El propósito de la estructura es mantener el estado de un objeto rastreado. El estado consiste en la información utilizada para la detección, la finalización, y la visualización del track. 
% La estructura contiene los siguientes campos
%   id : el ID entero de la pista
%                      bbox : el marco delimitador actual del objeto (bounding box);
%                             se utiliza para la visualización
%              kalmanFilter : un objeto del tipo filtro de Kalman utilizado para el 
%                             seguimiento basado en el movimiento
%                       age : el número de fotogramas desde que se detectó el track 
%                             por primera vez
%         totalVisibleCount : el número total de fotogramas en los que se ha detectado 
%                             el track  (visible)
% consecutiveInvisibleCount : el número de fotogramas consecutivos en los que no se ha 
%                             detectado el track  (invisible).
% Las detecciones ruidosas tienden a dar lugar a tracks de corta duración. 
% Por esta razón, el ejemplo sólo muestra un objeto después de haber sido 
% rastreado durante un cierto número de fotogramas. Esto ocurre cuando el 
% totalVisibleCount supera un umbral especificado. 
% Cuando no hay detecciones asociadas a un seguimiento durante varios 
% fotogramas consecutivos, el ejemplo asume que el objeto ha salido del 
% campo de visión y borra el seguimiento. 
% Esto sucede cuando consecutiveInvisibleCount supera un umbral especificado. 
% Una pista también puede ser borrada como ruido si fue rastreada durante un 
% corto periodo de tiempo, y marcada como invisible durante la mayoría de 
% los fotogramas. 
function [tracks trackssum]= initializeTracks()
    
        % Crea una estructura vacía de tracks
        tracks = struct(...
            'id', {}, ...
            'bbox', {}, ...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount', {});
        
        % Crea una estructura vacía de resumen de los tracks
        trackssum = struct(...
            'id', {}, ...
            'frame_i', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount', {}, ...
            'detections', {});

    end