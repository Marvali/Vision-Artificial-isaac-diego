%% Predecir las nuevas ubicaciones de los tracks existentes
% Utiliza el filtro Kalman para predecir el centroide de cada track en el 
% frame actual, y actualiza su cuadro delimitador en consecuencia.
    function [tracks] = predictNewLocationsOfTracks(tracks)
    
        for i = 1:length(tracks)
            
            bbox = int32(tracks(i).bbox);
            
            % predicción del filtro del kalman
            predictedCentroid = predict(tracks(i).kalmanFilter);
            
            % Desplazar el bounding box para que su centro esté en 
            % el centroide de la predicción.
            predictedCentroid = int32(predictedCentroid) - bbox(3:4) / 2;
            tracks(i).bbox = [predictedCentroid, bbox(3:4)];
            
        end
    end